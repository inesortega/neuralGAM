library(testthat)
library(NeuralGAM)

# Test if function throws error for missing smooth terms
test_that("NeuralGAM throws an error for missing smooth terms", {
  formula <- y ~ x
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(NeuralGAM(formula, data, num_units = 10))
})

# Test if function throws error for non-numeric num_units
test_that("NeuralGAM throws an error for non-numeric num_units", {
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(NeuralGAM(formula, data, num_units = "abc"))
})

# Test if function throws error for num_units < 1
test_that("NeuralGAM throws an error for num_units < 1", {
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(NeuralGAM(formula, data, num_units = 0))
})

# Test if function throws error for non-numeric learning_rate
test_that("NeuralGAM throws an error for non-numeric learning_rate", {
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(NeuralGAM(
    formula,
    data,
    num_units = 10,
    learning_rate = "abc"
  ))
})

# Test if function throws error for invalid family
test_that("NeuralGAM throws an error for invalid family", {
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(NeuralGAM(formula, data, num_units = 10, family = "abc"))
})