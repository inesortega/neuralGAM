library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}
# Test case 1: Check the weights for gaussian family
test_that("Weights for gaussian family should be equal to input weights", {
  skip_if_no_keras()

  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  w <- c(0.2, 0.6, 0.8)
  expected_output <- w
  actual_output <- weight(w, muhat, family)
  expect_equal(actual_output, expected_output)
})

# Test case 2: Check the weights for binomial family
test_that("Weights for binomial family should be correctly calculated", {
  skip_if_no_keras()

  family <- "binomial"
  muhat <- c(0.2, 0.7, 0.99)
  w <- c(0.2, 0.6, 0.8)

  # Calculate the expected weights
  diriv_values <- diriv(family, muhat)
  aux <- muhat * (1 - muhat) * (diriv_values^2)
  aux[aux <= 0.001] <- 0.001
  expected_output <- w / aux

  actual_output <- weight(w, muhat, family)
  expect_equal(actual_output, expected_output)
})

# Test case 3: Check for missing 'muhat' argument
test_that("Function should throw an error for missing 'muhat' argument", {
  skip_if_no_keras()

  family <- "gaussian"
  w <- c(0.2, 0.6, 0.8)
  expect_error(weight(w, family))
})

# Test case 4: Check for missing 'w' argument
test_that("Function should throw an error for missing 'w' argument", {
  skip_if_no_keras()

  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  expect_error(weight(muhat, family))
})

# Test case 5: Check for missing 'family' argument
test_that("Function should throw an error for missing 'family' argument", {
  skip_if_no_keras()

  muhat <- c(0.1, 0.5, 0.9)
  w <- c(0.2, 0.6, 0.8)
  expect_error(weight(w, muhat))
})

# Test case 6: Check for unsupported family
test_that("Function should throw an error for unsupported 'family'", {
  skip_if_no_keras()

  family <- "poisson"
  muhat <- c(0.1, 0.5, 0.9)
  w <- c(0.2, 0.6, 0.8)
  expect_error(weight(w, muhat, family))
})

# Test case 7: Check that weights calculation handles extreme muhat values for binomial family
test_that("Weights calculation should handle extreme muhat values for binomial family", {
  skip_if_no_keras()

  family <- "binomial"
  muhat <- c(0.0001, 0.9999)
  w <- c(0.2, 0.6)

  # Calculate the expected weights
  muhat[muhat <= 0.001] <- 0.001
  muhat[muhat >= 0.999] <- 0.999
  diriv_values <- neuralGAM:::diriv(family, muhat)
  aux <- muhat * (1 - muhat) * (diriv_values**2)
  aux[aux <= 0.001] <- 0.001
  expected_output <- w / aux

  actual_output <- neuralGAM:::weight(w, muhat, family)
  expect_equal(actual_output, expected_output)
})
