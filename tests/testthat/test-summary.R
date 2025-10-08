# tests/testthat/test-summary-neuralGAM.R


library(testthat)
library(reticulate)

skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("keras"))      skip("keras not available")
  if (!reticulate::py_module_available("tensorflow")) skip("tensorflow not available")
}
test_that("summary.neuralGAM basic printing and invisibility (smooth-only, fast)", {
  skip_if_no_keras()

  set.seed(1234)
  n  <- 160
  x1 <- runif(n, -2, 2)
  y  <- 1 + sin(x1) + rnorm(n, sd = 0.3)
  dat <- data.frame(x1 = x1, y = y)

  # Tiny model for speed
  m <- neuralGAM(
    y ~ s(x1),
    data = dat,
    family = "gaussian",
    num_units = 16,
    activation = "relu",
    validation_split = 0.1,
    dropout_rate = 0.1,
    verbose = 0,
    seed = 1234
  )

  # Capture printed output; also confirm invisible return
  out <- capture.output({
    ret <- expect_invisible(summary.neuralGAM(m))
    expect_s3_class(ret, "neuralGAM")
  })
  txt <- paste(out, collapse = "\n")

  # Key sections/fields should be present
  expect_match(txt, "neuralGAM summary")
  expect_match(txt, "Family\\s*:\\s*gaussian")
  expect_match(txt, "Formula\\s*:")
  expect_match(txt, "Observations\\s*:")
  expect_match(txt, "Intercept \\(eta0\\)\\s*:")
  expect_match(txt, "Deviance explained\\s*:")
  expect_match(txt, "Train MSE\\s*:")
  expect_match(txt, "Pred\\. / Conf\\. Int\\.")
  expect_match(txt, "Per-term configuration \\(parsed from s\\(\\.\\.\\.\\)\\)")
  expect_match(txt, "Neural network layer configuration per smooth term")
})

test_that("summary.neuralGAM prints linear component coefficients when present", {
  skip_if_no_keras()

  set.seed(5678)
  n  <- 150
  x1 <- runif(n, -2, 2)
  x2 <- rnorm(n)                # linear term
  y  <- 0.5 + 1.2 * x2 + cos(x1) + rnorm(n, sd = 0.25)
  dat <- data.frame(x1, x2, y)

  m_lin <- neuralGAM(
    y ~ s(x1) + x2,             # includes linear component
    data = dat,
    family = "gaussian",
    num_units = 16,
    activation = "relu",
    validation_split = 0.1,
    dropout_rate = 0.1,
    verbose = 0,
    seed = 5678
  )

  txt <- paste(capture.output(summary.neuralGAM(m_lin)), collapse = "\n")

  # Expect explicit linear component section
  expect_match(txt, "Linear component coefficients")
  # And that coefficients actually printed (at least an intercept-like number)
  expect_true(any(grepl("\\b[[:digit:]]+\\.?[[:digit:]]*\\b", txt)))
})

test_that("summary.neuralGAM errors on wrong object type", {
  # Call the method directly to bypass S3 dispatch to base::summary()
  expect_error(summary.neuralGAM(list()), "must be a neuralGAM object")
})
