library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}

# Test case 1: Check the neuralGAM:::deviance for gaussian family
test_that("neuralGAM:::deviance for gaussian family should be correctly calculated", {
  skip_if_no_keras()

  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  y <- c(0.2, 0.6, 0.8)
  expected_output <- mean((y - muhat)^2)
  actual_output <- neuralGAM:::dev(muhat, y, family)
  expect_equal(actual_output, expected_output)
})

# Test case 2: Check the neuralGAM:::deviance for binomial family
test_that("neuralGAM:::deviance for binomial family should be correctly calculated", {
  skip_if_no_keras()

  family <- "binomial"
  muhat <- c(0.2, 0.7, 0.99)
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

  actual_output <- neuralGAM:::dev(muhat, y, family)
  expect_equal(actual_output, expected_output)
})

# Test case 3: Check for missing 'muhat' argument
test_that("Function should throw an error for missing 'muhat' argument", {
  skip_if_no_keras()

  family <- "gaussian"
  y <- c(0.2, 0.6, 0.8)
  expect_error(neuralGAM:::dev(y = y, family = family))
})

# Test case 4: Check for missing 'y' argument
test_that("Function should throw an error for missing 'y' argument", {
  skip_if_no_keras()

  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  expect_error(neuralGAM:::dev(muhat = muhat, family = family))
})

# Test case 5: Check for missing 'family' argument
test_that("Function should throw an error for missing 'family' argument", {
  skip_if_no_keras()

  muhat <- c(0.1, 0.5, 0.9)
  y <- c(0.2, 0.6, 0.8)
  expect_error(neuralGAM:::dev(muhat = muhat, y = y))
})

# Test case 6: Check for unsupported family
test_that("Function should throw an error for unsupported 'family'", {
  skip_if_no_keras()

  family <- "poisson"
  muhat <- c(0.1, 0.5, 0.9)
  y <- c(0.2, 0.6, 0.8)
  expect_error(neuralGAM:::dev(muhat, y, family))
})
