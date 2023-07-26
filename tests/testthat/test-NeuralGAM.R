library(testthat)
library(neuralGAM)

# Test if function throws error for missing smooth terms
test_that("neuralGAM throws an error for missing smooth terms", {
  formula <- y ~ x
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10))
})

# Test if function throws error for non-numeric num_units
test_that("neuralGAM throws an error for non-numeric num_units", {
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = "abc"))
})

# Test if function throws error for num_units < 1
test_that("neuralGAM throws an error for num_units < 1", {
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 0))
})

# Test if function throws error for non-numeric learning_rate
test_that("neuralGAM throws an error for non-numeric learning_rate", {
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
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, family = "abc"))
})

# Test if function throws error for invalid loss
test_that("neuralGAM throws an error for invalid loss", {
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, loss = -1))
})

# Test if function throws error for invalid kernel initializer
test_that("neuralGAM throws an error for invalid kernel_initializer", {
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
  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  data <- data.frame(x = 1:10, y = rnorm(10))
  ngam <- neuralGAM(formula, data, num_units = 10, seed = seed)
  expect_equal(ngam$mse, 0.56552945)
})

