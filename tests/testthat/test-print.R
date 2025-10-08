# tests/testthat/test-print-neuralGAM.R
library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}

test_that("print.neuralGAM: validation and output format on a real model", {
  skip_if_no_keras()

  # --- error on wrong class
  expect_error(print.neuralGAM(list()), "must be a neuralGAM")

  # --- fit a tiny, fast model
  set.seed(2025)
  n <- 200
  d <- data.frame(x1 = runif(n, -2, 2))
  f <- sin(2 * pi * d$x1)
  d$y <- 1.5 + f + rnorm(n, sd = 0.2)

  m <- neuralGAM(
    y ~ s(x1),
    data = d,
    family = "gaussian",
    num_units = 32,
    learning_rate = 0.01,
    max_iter_backfitting = 2,
    max_iter_ls = 2,
    seed = 1,
    verbose = 0
  )

  # --- capture printed summary
  out <- testthat::capture_output(print.neuralGAM(m))

  # Core lines present
  expect_match(out, "^Class: neuralGAM", perl = TRUE)
  expect_match(out, "Family\\s*: gaussian")
  expect_match(out, "Formula\\s*: y ~ s\\(x1\\)")
  expect_match(out, sprintf("Observations\\s*: %d", n))

  # Intercept and MSE are numeric-looking
  expect_match(out, "Intercept \\(eta0\\)\\s*:\\s*[-+0-9\\.eE]+")
  expect_match(out, "Train MSE\\s*:\\s*[-+0-9\\.eE]+")
  # Deviance explained percentage with a '%' sign
  expect_match(out, "Deviance explained\\s*:\\s*[0-9]+\\.[0-9]{2}%")

  # Intervals line: either ENABLED(...) or 'disabled'
  if (isTRUE(m$build_pi) && !is.null(m$alpha) && !is.null(m$uncertainty_method)) {
    expect_match(
      out,
      "Pred\\. / Conf\\. Int\\. \\s*:\\s*ENABLED \\(alpha =\\s*[-+0-9\\.eE]+, method =\\s*.+\\)",
      perl = TRUE
    )
  } else {
    expect_match(out, "Pred\\. / Conf\\. Int\\. \\s*:\\s*disabled", perl = TRUE)
  }

  # --- returns the object invisibly
  vis <- withVisible(print.neuralGAM(m))
  expect_false(vis$visible)
  expect_identical(vis$value, m)
})
