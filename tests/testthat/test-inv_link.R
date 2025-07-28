library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}

test_that("Inverse link for gaussian family should be equal to muhat", {
  skip_if_no_keras()

  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  expected_output <- muhat
  actual_output <- neuralGAM:::inv_link(family, muhat)
  expect_equal(actual_output, expected_output)
})

test_that("Inverse link for binomial family should be correctly calculated", {
  skip_if_no_keras()

  family <- "binomial"
  muhat <- c(0.2, 0.7, 0.99)
  expected_output <- log(muhat / (1 - muhat))
  actual_output <- neuralGAM:::inv_link(family, muhat)
  expect_equal(actual_output, expected_output)
})

test_that("Function should throw an error for missing 'muhat' argument", {
  skip_if_no_keras()

  family <- "gaussian"
  expect_error(inv_link(family))
})

test_that("Function should throw an error for missing 'family' argument", {
  skip_if_no_keras()

  muhat <- c(0.1, 0.5, 0.9)
  expect_error(inv_link(muhat = muhat))
})


test_that("Function should throw an error for unsupported 'family'", {
  skip_if_no_keras()

  family <- "unknown"
  muhat <- c(0.1, 0.5, 0.9)
  expect_error(inv_link(family, muhat))
})
