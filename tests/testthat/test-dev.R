library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}

# Check the neuralGAM:::deviance for gaussian family
test_that("neuralGAM:::deviance for gaussian family should be correctly calculated", {
  skip_if_no_keras()

  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  w <- c(1, 1, 1)
  y <- c(0.2, 0.6, 0.8)
  expected_output <- mean((y - muhat)^2)
  actual_output <- neuralGAM:::dev(muhat, y, w, family)
  expect_equal(actual_output, expected_output)
})

# Check the neuralGAM:::deviance for binomial family
test_that("neuralGAM:::deviance for binomial family should be correctly calculated", {
  skip_if_no_keras()

  family <- "binomial"
  muhat <- c(0.2, 0.7, 0.99)
  w <- c(1, 1, 1)
  y <- c(0, 1, 1)

  muhat[muhat < 0.0001] <- 0.0001
  muhat[muhat > 0.9999] <- 0.9999

  entrop <- rep(0, length(y))
  ii <- (1 - y) * y > 0
  if (sum(ii, na.rm = TRUE) > 0) {
    entrop[ii] <- 2 * (y[ii] * log(y[ii])) +
      ((1 - y[ii]) * log(1 - y[ii]))
  } else {
    entrop <- 0
  }
  entadd <- 2 * (y * log(muhat)) + ((1 - y) * log(1 - muhat))
  expected_output <- sum(entrop - entadd, na.rm = TRUE)

  actual_output <- neuralGAM:::dev(muhat, y, w, family)
  expect_equal(actual_output, expected_output)
})

# Check the neuralGAM:::deviance for poisson family
test_that("neuralGAM:::deviance for poisson family returns correct value for known inputs", {
  # Test case 1
  y <- c(1, 2, 0)
  muhat <- c(1, 2, 1)
  w <- c(1, 1, 1)
  family <- "poisson"

  # Manually compute deviance
  # obs 1: 2 * (1 * log(1/1) - (1 - 1)) = 0
  # obs 2: 2 * (2 * log(2/2) - (2 - 2)) = 0
  # obs 3: 2 * (-0 * log(1) - (0 - 1)) + 0 = 2 * (0 - (-1)) = 2
  expected <- 2.0

  result <- neuralGAM:::dev(muhat, y, w, family)
  expect_equal(result, expected, tolerance = 1e-8)
})

test_that("poisson_deviance handles small fits and zeros safely", {
  y <- c(0, 5)
  muhat <- c(0.00001, 5)
  w <- c(1, 1)
  family <- "binomial"

  # Should not return NA/Inf
  result <- neuralGAM:::dev(muhat, y, w, family)
  expect_true(is.finite(result))
})

# Check for missing 'muhat' argument
test_that("Function should throw an error for missing 'muhat' argument", {
  skip_if_no_keras()

  family <- "gaussian"
  y <- c(0.2, 0.6, 0.8)
  expect_error(neuralGAM:::dev(y = y, family = family))
})

# Check for missing 'y' argument
test_that("Function should throw an error for missing 'y' argument", {
  skip_if_no_keras()

  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  expect_error(neuralGAM:::dev(muhat = muhat, family = family))
})

# Check for missing 'family' argument
test_that("Function should throw an error for missing 'family' argument", {
  skip_if_no_keras()

  muhat <- c(0.1, 0.5, 0.9)
  y <- c(0.2, 0.6, 0.8)
  expect_error(neuralGAM:::dev(muhat = muhat, y = y))
})

# Check for missing 'w' argument
test_that("Function should throw an error for missing 'w' argument", {
  skip_if_no_keras()

  muhat <- c(0.1, 0.5, 0.9)
  y <- c(0.2, 0.6, 0.8)
  expect_error(neuralGAM:::dev(muhat = muhat, y = y, family = "poisson"))
})

# Check for unsupported family
test_that("Function should throw an error for unsupported 'family'", {
  skip_if_no_keras()

  family <- "gamma"
  muhat <- c(0.1, 0.5, 0.9)
  y <- c(0.2, 0.6, 0.8)
  expect_error(neuralGAM:::dev(muhat, y, family))
})
