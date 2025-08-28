# ----------------------------
# MC Dropout & PI sanity tests
# ----------------------------

library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}


# small helper to build a tiny dropout model
.build_dropout_model <- function(output_dim = 1L, rate = 0.2) {
  keras::keras_model_sequential() |>
    keras::layer_dense(units = 8, activation = "relu", input_shape = 1) |>
    keras::layer_dropout(rate = rate) |>
    keras::layer_dense(units = as.integer(output_dim), activation = "linear") |>
    keras::compile(optimizer = "adam", loss = "mse")
}

test_that(".mc_dropout_forward returns [passes, n_obs, output_dim] (odim=1)", {
  skip_if_no_keras()

  set.seed(123)
  n <- 7
  passes <- 15L
  x <- matrix(seq(-1, 1, length.out = n), ncol = 1)

  mdl <- .build_dropout_model(output_dim = 1L, rate = 0.2)
  arr <- neuralGAM:::.mc_dropout_forward(mdl, x, passes, 1L)

  expect_true(is.array(arr))
  expect_equal(dim(arr), c(passes, n, 1L))
  # Variance across passes should be non-zero for at least some obs
  y_mat <- arr[, , 1]
  v <- var(y_mat)
  expect_true(any(v > 0))
})

test_that(".mc_dropout_forward returns [passes, n_obs, output_dim] (odim=3)", {
  skip_if_no_keras()

  set.seed(123)
  n <- 5
  passes <- 12L
  x <- matrix(runif(n, -1, 1), ncol = 1)

  mdl <- .build_dropout_model(output_dim = 3L, rate = 0.2)
  arr <- neuralGAM:::.mc_dropout_forward(mdl, x, passes, 3L)

  expect_true(is.array(arr))
  expect_equal(dim(arr), c(passes, n, 3L))
})

test_that("Quantile/mean sanity with MC dropout samples", {
  skip_if_no_keras()

  set.seed(321)
  n <- 9
  passes <- 40L
  x <- matrix(seq(-2, 2, length.out = n), ncol = 1)

  mdl <- .build_dropout_model(output_dim = 3L, rate = 0.25)
  arr <- neuralGAM:::.mc_dropout_forward(mdl, x, passes, 3L)
  y_mat <- arr[, , 1]  # [passes, n]

  lower_q <- 0.025
  upper_q <- 0.975
  low  <- matrixStats::colQuantiles(y_mat, probs = lower_q)
  up   <- matrixStats::colQuantiles(y_mat, probs = upper_q)
  mean <- matrixStats::colMeans2(y_mat)

  # monotonicity and containment
  expect_true(all(low <= up + 1e-12))
  expect_true(all(low <= mean + 1e-12))
  expect_true(all(mean <= up + 1e-12))
  # some spread present
  expect_true(any(up - low > 0))
})

test_that(".combine_uncertainties_sampling: decomposition and bounds look sane", {
  skip_if_no_keras()

  set.seed(999)
  n <- 11
  passes <- 50L
  alpha <- 0.05
  z <- qnorm(1 - alpha/2)

  # Create synthetic MC means (epistemic) and fixed aleatoric widths per obs
  mean_mat <- matrix(rnorm(passes * n, sd = 0.6), nrow = passes, ncol = n)
  sigma_ale <- runif(n, 0.1, 0.4)                         # per-observation aleatoric sd
  lwr_mat <- sweep(mean_mat, 2, z * sigma_ale, FUN = "-") # same width each pass
  upr_mat <- sweep(mean_mat, 2, z * sigma_ale, FUN = "+")

  out <- .combine_uncertainties_sampling(lwr_mat, upr_mat, mean_mat, alpha)

  expect_true(is.data.frame(out))
  expect_equal(nrow(out), n)
  expect_true(all(c("lwr","upr","var_epistemic","var_aleatoric","var_total") %in% names(out)))

  # variance additivity (within numerical tolerance)
  expect_lt(max(abs(out$var_total - (out$var_epistemic + out$var_aleatoric))), 1e-8)

  # bounds contain fit
  expect_true(all(out$lwr <= out$fit + 1e-12))
  expect_true(all(out$fit <= out$upr + 1e-12))

  # aleatoric variance ~ sigma_ale^2
  expect_lt(mean(abs(out$var_aleatoric - sigma_ale^2)), 1e-2)
})
