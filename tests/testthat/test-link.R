library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}

test_that("Link function for gaussian family should be equal to muhat", {
  skip_if_no_keras()

  family <- "gaussian"
  muhat <- c(0.1, 0.5, 0.9)
  expected_output <- muhat
  actual_output <- link(family, muhat)
  expect_equal(actual_output, expected_output)
})

# Check the link function for binomial family
test_that("Link function for binomial family should be correctly calculated", {
  skip_if_no_keras()

  family <- "binomial"
  muhat <- c(0.2, 0.7, 0.99)
  expected_output <- exp(muhat) / (1 + exp(muhat))
  actual_output <- link(family, muhat)
  expect_equal(actual_output, expected_output)
})

# Check for missing 'muhat' argument
test_that("Function should throw an error for missing 'muhat' argument", {
  skip_if_no_keras()

  family <- "gaussian"
  expect_error(link(family))
})

# Check for missing 'family' argument
test_that("Function should throw an error for missing 'family' argument", {
  skip_if_no_keras()

  muhat <- c(0.1, 0.5, 0.9)
  expect_error(link(muhat = muhat))
})

# Check for unsupported family
test_that("Function should throw an error for unsupported 'family'", {
  family <- "unknown"
  muhat <- c(0.1, 0.5, 0.9)
  expect_error(link(family, muhat))
})

# Check that large positive / negative values of muhat are capped for binomial family
test_that("Function should cap large positive values of muhat at 10 for binomial family", {
  skip_if_no_keras()

  family <- "binomial"
  muhat <- c(500, 300, -300)
  expected_cap_muhat <- pmin(pmax(muhat, -300), 300)
  exp_eta <- exp(expected_cap_muhat)
  expected_output <- exp_eta / (1 + exp_eta)
  actual_output <- link(family, muhat)
  testthat::expect_equal(expected_output, actual_output)
})
