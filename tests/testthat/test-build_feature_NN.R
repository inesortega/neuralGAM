library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}

# Test case 1: function throws error for missing num_units
test_that("build_feature_NN throws an error for missing num_units", {
  skip_if_no_keras()

  # Missing num_units
  expect_error(neuralGAM:::build_feature_NN())
})

# num_units is not numeric or vector of integers
test_that("Invalid num_units argument", {
  skip_if_no_keras()
  expect_error(neuralGAM:::build_feature_NN(num_units = "string"))
})

# num_units is not numeric or vector of integers
test_that("Invalid num_units argument", {
  skip_if_no_keras()
  expect_error(neuralGAM:::build_feature_NN(num_units = c("string", "string2")))
})

# learning_rate is not numeric
test_that("Invalid learning_rate argument", {
  skip_if_no_keras()
  expect_error(neuralGAM:::build_feature_NN(num_units = 10, learning_rate = "string"))
})

# name is not NULL or character string
test_that("Invalid name argument", {
  skip_if_no_keras()
  expect_error(neuralGAM:::build_feature_NN(num_units = 10, name = 123))
})

# Valid function call
test_that("Valid function call", {
  skip_if_no_keras()
  testthat::expect_no_error(neuralGAM:::build_feature_NN(num_units = 10, learning_rate = 0.001, name = "test"))
})

# Valid function call
test_that("Valid function call with deep layer", {
  skip_if_no_keras()
  testthat::expect_no_error(neuralGAM:::build_feature_NN(num_units = c(10,20), learning_rate = 0.001, name = "test"))
})
