library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}

# Test case 1: Check the derivative for gaussian family
test_that("Derivative for gaussian family should be 1", {
  skip_if_no_keras()

  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  expected_output <- 1
  actual_output <- neuralGAM:::diriv(family, muhat)
  expect_equal(actual_output, expected_output)
})

# Test case 2: Check the derivative for binomial family
test_that("Derivative for binomial family should be correctly calculated", {
  family <- "binomial"
  muhat <- c(0.2, 0.7, 0.99)
  expected_output <- 1 / (muhat * (1 - muhat))
  actual_output <- neuralGAM:::diriv(family, muhat)
  expect_equal(actual_output, expected_output)
})

# Test case 3: Check for missing 'muhat' argument
test_that("Function should throw an error for missing 'muhat' argument", {
  skip_if_no_keras()

  family <- "gaussian"
  expect_error(neuralGAM:::diriv(family))
})

# Test case 4: Check for missing 'family' argument
test_that("Function should throw an error for missing 'family' argument", {
  skip_if_no_keras()

  muhat <- c(0.1, 0.5, 0.9)
  expect_error(neuralGAM:::diriv(muhat = muhat))
})

# Test case 5: Check for unsupported family
test_that("Function should throw an error for unsupported 'family'", {
  skip_if_no_keras()
  family <- "unknown"
  muhat <- c(0.1, 0.5, 0.9)
  expect_error(neuralGAM:::diriv(family, muhat))
})

# Test case 6: Check that derivative calculation handles extreme muhat values for binomial family
test_that("Derivative calculation should handle extreme muhat values for binomial family", {
  skip_if_no_keras()

  family <- "binomial"
  muhat <- c(0.001, 0.999)
  expected_output <- c(1 / (0.001 * (1 - 0.001)), 1 / (0.999 * (1 - 0.999)))
  actual_output <- neuralGAM:::diriv(family, muhat)
  expect_equal(actual_output, expected_output)
})
