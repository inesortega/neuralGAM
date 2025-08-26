library(testthat)

# --- Gaussian ---------------------------------------------------------------

test_that("dev for gaussian family is sum of squared residuals (with weights)", {
  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  y <- c(0.2, 0.6, 0.8)

  # unweighted -> weights default to 1
  expected_unw <- sum((y - muhat)^2)
  actual_unw <- dev(muhat = muhat, y = y, family = family)
  expect_equal(actual_unw, expected_unw, tolerance = 1e-12)

  # with weights
  w <- c(1, 2, 3)
  expected_w <- sum(w * (y - muhat)^2)
  actual_w <- dev(muhat = muhat, y = y, family = family, w = w)
  expect_equal(actual_w, expected_w, tolerance = 1e-12)
})

# --- Binomial ---------------------------------------------------------------

test_that("dev for binomial family matches analytic form and is finite", {
  family <- "binomial"
  muhat <- c(0.2, 0.7, 0.99)
  y <- c(0, 1, 1)

  # Expected using the same stabilized formula as implementation
  eps <- 1e-12
  mu_c <- pmin(pmax(muhat, eps), 1 - eps)
  y_c  <- pmin(pmax(y,     eps), 1 - eps)
  expected <- sum(2 * ( y_c * log(y_c / mu_c) + (1 - y_c) * log((1 - y_c) / (1 - mu_c)) ))

  actual <- dev(muhat = muhat, y = y, family = family)
  expect_equal(actual, expected, tolerance = 1e-10)
  expect_true(is.finite(actual))
})

test_that("binomial deviance handles extreme fits and zeros safely", {
  family <- "binomial"
  y <- c(0, 1, 0, 1)
  muhat <- c(1e-20, 1 - 1e-20, 1e-8, 1 - 1e-8)

  res <- dev(muhat, y, family = family)
  expect_true(is.numeric(res) && length(res) == 1L && is.finite(res))
})

# --- Poisson ----------------------------------------------------------------

test_that("dev for poisson family returns correct value for known inputs", {
  y <- c(1, 2, 0)
  muhat <- c(1, 2, 1)
  w <- c(1, 1, 1)
  family <- "poisson"

  # Manually: obs1=0, obs2=0, obs3=2
  expected <- 2.0
  result <- dev(muhat, y, family = family, w = w)
  expect_equal(result, expected, tolerance = 1e-8)
})

test_that("poisson deviance is finite for small fits and zeros", {
  y <- c(0, 5, 0)
  muhat <- c(1e-12, 5, 1e-9)
  res <- dev(muhat, y, family = "poisson")
  expect_true(is.finite(res))
})

# --- Argument validation ----------------------------------------------------

test_that("dev() errors on missing required args but not on missing weights", {
  y <- c(0.2, 0.6, 0.8)
  muhat <- c(0.1, 0.5, 0.9)

  expect_error(dev(y = y, family = "gaussian"))
  expect_error(dev(muhat = muhat, family = "gaussian"))
  expect_error(dev(muhat = muhat, y = y))

  # Missing w should be OK (defaults to 1's)
  expect_silent({
    res <- dev(muhat = muhat, y = y, family = "gaussian")
    expect_true(is.numeric(res) && length(res) == 1L)
  })
})

test_that("dev() errors on unsupported family", {
  muhat <- c(0.1, 0.5, 0.9)
  y <- c(0.2, 0.6, 0.8)
  expect_error(dev(muhat, y, "gamma"))
})

# --- Deviance explained -----------------------------------------------------

test_that(".deviance_explained.neuralGAM returns sensible value and attributes", {
  # toy perfectly-fit gaussian model -> dev explained = 1
  y <- c(1, 2, 3, 4)
  obj <- structure(list(
    y = y,
    muhat = y,                 # perfect fit
    family = "gaussian"
  ), class = "neuralGAM")

  out <- .deviance_explained.neuralGAM(obj)
  expect_true(!is.na(attr(out, "percent")))
  expect_true(!is.na(attr(out, "dev_model")))
  expect_true(!is.na(attr(out, "dev_null")))
  expect_identical(attr(out, "family"), "gaussian")
  expect_s3_class(out, "neuralGAM_devexp")
})
