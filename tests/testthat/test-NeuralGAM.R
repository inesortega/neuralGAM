library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if(!findpython::can_find_python_cmd(minimum_version = "3.8")){
    skip("Python not found on system...")
  }

  if (!reticulate::py_module_available("keras"))
    skip("keras not available for testing")
}

# Test if function throws error for missing smooth terms
test_that("neuralGAM throws an error for missing smooth terms", {
  skip_if_no_keras()

  formula <- y ~ x
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = 10))
})

# Test if function throws error for non-numeric num_units
test_that("neuralGAM throws an error for non-numeric num_units", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = "abc"))
})

# Test if function throws error for num_units < 1
test_that("neuralGAM throws an error for num_units < 1", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = 0))
})

# Test if function throws error for non-numeric learning_rate
test_that("neuralGAM throws an error for non-numeric learning_rate", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = 10,
    learning_rate = "abc"
  ))
})

# Test if function throws error for invalid family
test_that("neuralGAM throws an error for invalid family", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = 10,
    family = "abc"))
})

# Test if function throws error for invalid loss
test_that("neuralGAM throws an error for invalid loss", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, loss = -1))
})

# Test if function throws error for invalid kernel initializer
test_that("neuralGAM throws an error for invalid kernel_initializer", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = 10,
    kernel_initializer = -1
  ))
})

# Test if function throws error for invalid bias initializer
test_that("neuralGAM throws an error for invalid bias_initializer", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = 10,
    bias_initializer = -1
  ))
})

# Test if function runs OK main example
test_that("neuralGAM runs OK", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  data <- data.frame(x = 1:10, y = rnorm(10))

  ngam <- neuralGAM(formula, data, num_units = 10, seed = seed)
  expect_equal(round(ngam$mse,4), 0.5655)
})

