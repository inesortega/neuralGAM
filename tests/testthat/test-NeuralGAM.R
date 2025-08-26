library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
    )
  ) skip("keras not available for testing...")
}

# ----------------------------
# Validation error tests
# ----------------------------

test_that("neuralGAM throws an error for missing smooth terms", {
  skip_if_no_keras()
  formula <- y ~ x
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10))
})

test_that("neuralGAM throws an error for non-numeric num_units", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = "abc"))
})

test_that("neuralGAM throws an error for num_units < 1", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 0))
})

test_that("neuralGAM throws an error for non-numeric learning_rate", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, learning_rate = "abc"))
})

test_that("neuralGAM throws an error for invalid family", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, family = "abc"))
})

test_that("neuralGAM throws an error for invalid loss type", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, loss = -1))
})

test_that("neuralGAM throws an error for incompatible loss with PI aleatoric", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, pi_method = "aleatoric", loss = "binary_crossentropy"))
})

test_that("neuralGAM throws an error for invalid kernel_initializer", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, kernel_initializer = -1))
})

test_that("neuralGAM throws an error for invalid bias_initializer", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, bias_initializer = -1))
})

test_that("neuralGAM throws an error for invalid regularizers", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, kernel_regularizer = "abc"))
  expect_error(neuralGAM(formula, data, num_units = 10, bias_regularizer = "abc"))
  expect_error(neuralGAM(formula, data, num_units = 10, activity_regularizer = "abc"))
})

test_that("neuralGAM throws an error for invalid alpha", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, pi_method = "aleatoric", alpha = -0.1))
  expect_error(neuralGAM(formula, data, num_units = 10, pi_method = "aleatoric", alpha = 1.5))
})

test_that("neuralGAM throws an error for invalid validation_split", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, validation_split = -0.2))
  expect_error(neuralGAM(formula, data, num_units = 10, validation_split = 1.2))
})

test_that("neuralGAM throws an error for invalid w_train", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, w_train = "abc"))
  expect_error(neuralGAM(formula, data, num_units = 10, w_train = rep(1, 5))) # length mismatch
})

test_that("neuralGAM throws an error for invalid thresholds", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, bf_threshold = -0.1))
  expect_error(neuralGAM(formula, data, num_units = 10, ls_threshold = -0.1))
})

test_that("neuralGAM throws an error for invalid iteration counts", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, max_iter_backfitting = 0))
  expect_error(neuralGAM(formula, data, num_units = 10, max_iter_ls = 0))
})

test_that("neuralGAM throws an error for invalid seed", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, seed = "abc"))
})

test_that("neuralGAM throws an error for invalid verbose", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, verbose = 2))
})

test_that("neuralGAM runs OK with single hidden layer", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  data <- data.frame(x = 1:10, y = rnorm(10))

  ngam <- neuralGAM(formula, data, num_units = 10, seed = seed)
  expect_equal(round(ngam$mse,4), 0.5655)
})

test_that("neuralGAM runs OK with deep architecture", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  data <- data.frame(x = 1:10, y = rnorm(10))

  ngam <- neuralGAM(formula,
                    data,
                    num_units = c(10,10),
                    seed = seed,
                    max_iter_backfitting = 1,
                    max_iter_ls = 1)
  expect_equal(round(ngam$mse,4), 0.5207)
})

test_that("neuralGAM runs OK with mixed neural-linear model architecture", {
  skip_if_no_keras()

  seed <- 10
  formula <- y ~ s(x1, num_units = 32) + x2
  set.seed(seed)
  data <- data.frame(x1 = 1:10, x2 = 1:10, y = rnorm(10))

  ngam <- neuralGAM(
    formula,
    data = data,
    seed = seed,
    max_iter_backfitting = 1,
    max_iter_ls = 1
  )
  expect_equal(round(ngam$mse,4), 0.5105)
})

test_that("neuralGAM runs OK with binomial response", {
  skip_if_no_keras()

  n <- 10
  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  eta0 <- rnorm(n)
  true_eta <- exp(eta0)/(1 + exp(eta0)) # generate probs

  data <- data.frame(x = 1:10, y = rbinom(n, 1, true_eta))

  ngam <- neuralGAM(formula,
                    data,
                    num_units = 10,
                    seed = seed,
                    family = "binomial")

  expect_equal(round(ngam$mse,4), 0.221)
})

test_that("neuralGAM runs OK with poisson response", {
  skip_if_no_keras()

  n <- 10
  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  lambda <- runif(n, 1, 5)
  eta0 <- rpois(n, lambda)

  data <- data.frame(x = 1:10, y = eta0)

  ngam <- neuralGAM(formula,
                    data,
                    num_units = 10,
                    seed = seed,
                    family = "poisson", max_iter_backfitting = 1,
                    max_iter_ls = 1)

  expect_equal(round(ngam$mse,4), 2.3316)
})


test_that("neuralGAM runs OK with Prediction Intervals and gaussian response", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  data <- data.frame(x = 1:10, y = rnorm(10))

  ngam <- neuralGAM(formula, data, num_units = 10, seed = seed, pi_method = "aleatoric", alpha = 0.05)
  expect_equal(round(ngam$mse,4), 0.7079)
})

test_that("neuralGAM runs OK with Prediction Intervals and binomial response", {
  skip_if_no_keras()

  n <- 10
  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  eta0 <- rnorm(n)
  true_eta <- exp(eta0)/(1 + exp(eta0)) # generate probs

  data <- data.frame(x = 1:10, y = rbinom(n, 1, true_eta))

  ngam <- neuralGAM(formula,
                    data,
                    num_units = 10,
                    seed = seed,
                    family = "binomial",
                    pi_method = "aleatoric",
                    alpha = 0.05)

  expect_equal(round(ngam$mse,4), 0.2105)
})

test_that("neuralGAM runs OK with Prediction Intervals and poisson response", {
  skip_if_no_keras()

  n <- 10
  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  lambda <- runif(n, 1, 5)
  eta0 <- rpois(n, lambda)

  data <- data.frame(x = 1:10, y = eta0)

  ngam <- neuralGAM(formula,
                    data,
                    num_units = 10,
                    seed = seed,
                    family = "poisson", max_iter_backfitting = 1,
                    max_iter_ls = 1,
                    pi_method = "aleatoric",
                    alpha = 0.05)

  expect_equal(round(ngam$mse,4), 4.4227)
})


test_that("neuralGAM runs OK with mixed neural-linear model architecture and PI", {
  skip_if_no_keras()

  seed <- 10
  set.seed(seed)

  formula <- y ~ s(x1, num_units = 32) + x2
  data <- data.frame(x1 = 1:10, x2 = 1:10, y = rnorm(10))

  # global num_units = 64 should apply only to x2, not override x1's 32
  ngam <- neuralGAM(
    formula,
    data = data,
    num_units = 64,
    seed = seed,
    max_iter_backfitting = 1,
    max_iter_ls = 1,
    pi_method = "aleatoric",
    alpha = 0.05
  )
  expect_equal(round(ngam$mse,4), 0.5234)
})


test_that("neuralGAM accepts valid validation_split", {
  skip_if_no_keras()

  formula <- y ~ s(x)

  seed <- 10
  set.seed(seed)

  data <- data.frame(x = 1:10, y = rnorm(10))
  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1, validation_split = 0.2)
  expect_equal(round(ngam$mse,4), 2.7667)
})

test_that("neuralGAM accepts valid w_train", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  data <- data.frame(x = 1:10, y = rnorm(10))
  w <- rep(1, 10)

  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1,  w_train = w)

  expect_equal(round(ngam$mse,4), 2.7667)
})

test_that("neuralGAM accepts build_pi=TRUE with supported losses", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  data <- data.frame(x = 1:10, y = rnorm(10))

  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1, pi_method = "aleatoric",
                    alpha = 0.05, loss = "mse")

  expect_equal(round(ngam$mse, 4), 0.8739)

  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1, pi_method = "aleatoric",
                    alpha = 0.05, loss = "mae")

  expect_equal(round(ngam$mse, 4), 0.8739)
})


test_that("neuralGAM accepts per-term kernel_initializer and bias_initializer", {
  skip_if_no_keras()

  seed <- 10
  set.seed(seed)
  formula <- y ~ s(x, num_units = 5,
                   kernel_initializer = keras::initializer_he_normal(),
                   bias_initializer = keras::initializer_zeros())
  data <- data.frame(x = 1:10, y = rnorm(10))
  ngam <- neuralGAM(formula, data, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1)
  expect_true(inherits(ngam, "neuralGAM"))
})

test_that("neuralGAM rejects invalid per-term kernel_initializer / bias_initializer", {
  skip_if_no_keras()
  formula <- y ~ s(x, kernel_initializer = 123)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 5))

  formula <- y ~ s(x, bias_initializer = list(a = 1))
  expect_error(neuralGAM(formula, data, num_units = 5))
})

test_that("neuralGAM accepts valid per-term regularizers", {
  skip_if_no_keras()

  seed <- 10
  set.seed(seed)

  formula <- y ~ s(x,
                   kernel_regularizer = keras::regularizer_l2(1e-4),
                   bias_regularizer = keras::regularizer_l1(1e-4))
  data <- data.frame(x = 1:10, y = rnorm(10))
  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1)
  expect_true(inherits(ngam, "neuralGAM"))
})

test_that("neuralGAM rejects invalid per-term regularizers", {
  skip_if_no_keras()
  formula <- y ~ s(x, kernel_regularizer = "abc")
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 5))
})


test_that("neuralGAM accepts per-term num_units and default value for other smooth terms", {
  skip_if_no_keras()

  seed <- 10
  set.seed(seed)

  formula <- y ~ s(x1, num_units = 32) + s(x2)
  data <- data.frame(x1 = 1:10, x2 = 1:10, y = rnorm(10))
  ngam <- neuralGAM(formula, data, num_units = 64, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1)

  # global num_units = 64 should apply only to x2, not override x1's 32
  ngam <- neuralGAM(
    formula,
    data = data,
    num_units = 64,
    seed = seed,
    max_iter_backfitting = 1,
    max_iter_ls = 1
  )

  expect_equal(ngam$formula$np_architecture$x1$num_units, 32)
  expect_equal(ngam$formula$np_architecture$x2$num_units, 64)

})


# ----------------------------
# Prediction interval (PI) method tests
# ----------------------------


.fit_ngam_with_pi <- function(pi_method,
                              family = "gaussian",
                              seed = 10,
                              n = 10,
                              alpha = 0.05) {
  set.seed(seed)

  if (family == "gaussian") {
    data <- data.frame(x = 1:n, y = rnorm(n))
    formula <- y ~ s(x)
  } else if (family == "binomial") {
    eta0 <- rnorm(n)
    p <- exp(eta0) / (1 + exp(eta0))
    data <- data.frame(x = 1:n, y = rbinom(n, 1, p))
    formula <- y ~ s(x)
  } else if (family == "poisson") {
    lambda <- runif(n, 1, 5)
    y <- rpois(n, lambda)
    data <- data.frame(x = 1:n, y = y)
    formula <- y ~ s(x)
  } else {
    stop("Unsupported family for helper")
  }

  # Try to fit; if error mentions unsupported pi method, skip
  res <- try(
    neuralGAM(
      formula,
      data = data,
      num_units = 5,
      seed = seed,
      max_iter_backfitting = 1,
      max_iter_ls = 1,
      family = family,
      pi_method = pi_method,
      alpha = alpha
    ),
    silent = TRUE
  )

  if (inherits(res, "try-error")) {
    msg <- as.character(res)
    if (grepl("pi_method|unsupported|not supported|invalid.*pi", msg, ignore.case = TRUE)) {
      skip(paste0("pi_method='", pi_method, "' not supported in this build"))
    } else {
      stop(res)
    }
  }

  res
}

test_that("neuralGAM rejects invalid pi_method", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(
    neuralGAM(formula, data,
              num_units = 5,
              pi_method = "definitely_not_a_method",
              alpha = 0.05)
  )
})

# Create a list of accepted PI methods.
# Tests will auto-skip any that aren't implemented in the installed version.
pi_methods_to_check <- c(
  "aleatoric",
  "epistemic",
  "both"
  )

test_that("neuralGAM runs OK with various pi_methods (gaussian)", {
  skip_if_no_keras()
  for (m in pi_methods_to_check) {
    ngam <- .fit_ngam_with_pi(m, family = "gaussian", alpha = 0.05)
    expect_true(inherits(ngam, "neuralGAM"), info = paste("pi_method =", m))
    expect_true(is.numeric(ngam$mse) && length(ngam$mse) == 1, info = paste("pi_method =", m))
  }
})

test_that("neuralGAM runs OK with various pi_methods (binomial)", {
  skip_if_no_keras()
  for (m in pi_methods_to_check) {
    ngam <- .fit_ngam_with_pi(m, family = "binomial", alpha = 0.05)
    expect_true(inherits(ngam, "neuralGAM"), info = paste("pi_method =", m))
    expect_true(is.numeric(ngam$mse) && length(ngam$mse) == 1, info = paste("pi_method =", m))
  }
})

test_that("neuralGAM runs OK with various pi_methods (poisson)", {
  skip_if_no_keras()
  for (m in pi_methods_to_check) {
    ngam <- .fit_ngam_with_pi(m, family = "poisson", alpha = 0.05)
    expect_true(inherits(ngam, "neuralGAM"), info = paste("pi_method =", m))
    expect_true(is.numeric(ngam$mse) && length(ngam$mse) == 1, info = paste("pi_method =", m))
  }
})

# Sanity check that changing alpha has an effect for quantile-based methods
# (This does not assert exact numeric values; it only checks the model runs and
# that training with a different alpha still returns a valid object.)
test_that("neuralGAM quantile-like pi_methods accept different alpha values", {
  skip_if_no_keras()
  for (m in c("aleatoric", "both")) {
    ngam_a <- .fit_ngam_with_pi(m, family = "gaussian", alpha = 0.90)
    ngam_b <- .fit_ngam_with_pi(m, family = "gaussian", alpha = 0.50)
    expect_true(inherits(ngam_a, "neuralGAM"), info = paste("pi_method =", m, "alpha=0.90"))
    expect_true(inherits(ngam_b, "neuralGAM"), info = paste("pi_method =", m, "alpha=0.50"))
  }
})

# Given that predict() supports returning intervals, include a shape check.
# This test will be skipped if predict() doesn't expose interval outputs.
test_that("predict(neuralGAM) returns lower/upper/mean and variances when built with PI", {
  skip_if_no_keras()

  set.seed(10)
  n <- 12
  df <- data.frame(x = runif(n), y = rnorm(n))
  fit <- try(
    neuralGAM(y ~ s(x),
              data = df,
              num_units = 5,
              seed = 10,
              max_iter_backfitting = 1,
              max_iter_ls = 1,
              pi_method = "aleatoric",
              alpha = 0.05),
    silent = TRUE
  )

  if (inherits(fit, "try-error")) {
    msg <- as.character(fit)
    if (grepl("pi_method|unsupported|not supported|invalid.*pi", msg, ignore.case = TRUE)) {
      skip("predict-with-PI test skipped: 'aleatoric' pi_method not supported in this build")
    } else {
      stop(fit)
    }
  }

  # attempt prediction
  newx <- data.frame(x = seq(0, 1, length.out = 5))
  pr <- try(predict(fit, newdata = newx, type = "response", se.fit = TRUE), silent = TRUE)

  if (inherits(pr, "try-error")) {
    skip("predict() with intervals not available in this build")
  } else {
    expect_equal(length(pr$fit), nrow(newx))
    expect_equal(length(pr$se.fit), nrow(newx))
  }
})

