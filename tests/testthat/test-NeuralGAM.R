library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
    )
  ) skip("keras not available for testing...")
}

test_that("neuralGAM throws an error for missing smooth terms", {
  skip_if_no_keras()

  formula <- y ~ x
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = 10))
})

test_that("neuralGAM throws an error for non-numeric num_units", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = "abc"))
})

test_that("neuralGAM throws an error for num_units < 1", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(
    formula,
    data,
    num_units = 0))
})

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

test_that("neuralGAM throws an error for invalid loss", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, loss = -1))
})

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

test_that("neuralGAM runs OK with single hidden layer", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  data <- data.frame(x = 1:10, y = rnorm(10))

  ngam <- neuralGAM(formula, data, num_units = 10, seed = seed)
  expect_equal(round(ngam$mse,4), 0.5655)
})

test_that("neuralGAM runs OK with deep architecture", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  data <- data.frame(x = 1:10, y = rnorm(10))

  ngam <- neuralGAM(formula,
                    data,
                    num_units = c(10,10),
                    seed = seed,
                    max_iter_backfitting = 1,
                    max_iter_ls = 1)
  expect_equal(round(ngam$mse,4), 0.5207)
})


test_that("neuralGAM runs OK with binomial response", {
  skip_if_no_keras()

  n <- 10
  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  eta0 <- rnorm(n)
  true_eta <- exp(eta0)/(1 + exp(eta0)) # generate probs

  data <- data.frame(x = 1:10, y = rbinom(n, 1, true_eta))

  ngam <- neuralGAM(formula,
                    data,
                    num_units = 10,
                    seed = seed,
                    family = "binomial")

  expect_equal(round(ngam$mse,4), 0.221)
})

