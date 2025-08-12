library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
    )
  ) skip("keras not available for testing...")
}

# ----------------------------
# Validation error tests
# ----------------------------

test_that("neuralGAM throws an error for missing smooth terms", {
  skip_if_no_keras()
  formula <- y ~ x
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10))
})

test_that("neuralGAM throws an error for non-numeric num_units", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = "abc"))
})

test_that("neuralGAM throws an error for num_units < 1", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 0))
})

test_that("neuralGAM throws an error for non-numeric learning_rate", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, learning_rate = "abc"))
})

test_that("neuralGAM throws an error for invalid family", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, family = "abc"))
})

test_that("neuralGAM throws an error for invalid loss type", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, loss = -1))
})

test_that("neuralGAM throws an error for incompatible loss when build_pi=TRUE", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, build_pi = TRUE, loss = "binary_crossentropy"))
})

test_that("neuralGAM throws an error for invalid kernel_initializer", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, kernel_initializer = -1))
})

test_that("neuralGAM throws an error for invalid bias_initializer", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, bias_initializer = -1))
})

test_that("neuralGAM throws an error for invalid regularizers", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, kernel_regularizer = "abc"))
  expect_error(neuralGAM(formula, data, num_units = 10, bias_regularizer = "abc"))
  expect_error(neuralGAM(formula, data, num_units = 10, activity_regularizer = "abc"))
})

test_that("neuralGAM throws an error for invalid alpha", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, build_pi = TRUE, alpha = -0.1))
  expect_error(neuralGAM(formula, data, num_units = 10, build_pi = TRUE, alpha = 1.5))
})

test_that("neuralGAM throws an error for invalid validation_split", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, validation_split = -0.2))
  expect_error(neuralGAM(formula, data, num_units = 10, validation_split = 1.2))
})

test_that("neuralGAM throws an error for invalid w_train", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, w_train = "abc"))
  expect_error(neuralGAM(formula, data, num_units = 10, w_train = rep(1, 5))) # length mismatch
})

test_that("neuralGAM throws an error for invalid thresholds", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, bf_threshold = -0.1))
  expect_error(neuralGAM(formula, data, num_units = 10, ls_threshold = -0.1))
})

test_that("neuralGAM throws an error for invalid iteration counts", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, max_iter_backfitting = 0))
  expect_error(neuralGAM(formula, data, num_units = 10, max_iter_ls = 0))
})

test_that("neuralGAM throws an error for invalid seed", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, seed = "abc"))
})

test_that("neuralGAM throws an error for invalid verbose", {
  skip_if_no_keras()
  formula <- y ~ s(x)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 10, verbose = 2))
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

test_that("neuralGAM runs OK with mixed neural-linear model architecture", {
  skip_if_no_keras()

  seed <- 10
  formula <- y ~ s(x1, num_units = 32) + x2
  set.seed(seed)
  data <- data.frame(x1 = 1:10, x2 = 1:10, y = rnorm(10))

  ngam <- neuralGAM(
    formula,
    data = data,
    seed = seed,
    max_iter_backfitting = 1,
    max_iter_ls = 1
  )
  expect_equal(round(ngam$mse,4), 0.5105)
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

test_that("neuralGAM runs OK with poisson response", {
  skip_if_no_keras()

  n <- 10
  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  lambda <- runif(n, 1, 5)
  eta0 <- rpois(n, lambda)

  data <- data.frame(x = 1:10, y = eta0)

  ngam <- neuralGAM(formula,
                    data,
                    num_units = 10,
                    seed = seed,
                    family = "poisson", max_iter_backfitting = 1,
                    max_iter_ls = 1)

  expect_equal(round(ngam$mse,4), 2.3316)
})


test_that("neuralGAM runs OK with Prediction Intervals and gaussian response", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  data <- data.frame(x = 1:10, y = rnorm(10))

  ngam <- neuralGAM(formula, data, num_units = 10, seed = seed, build_pi = TRUE, alpha = 0.95)
  expect_equal(round(ngam$mse,4), 0.7079)
})

test_that("neuralGAM runs OK with Prediction Intervals and binomial response", {
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
                    family = "binomial",
                    build_pi = TRUE,
                    alpha = 0.95)

  expect_equal(round(ngam$mse,4), 0.2105)
})

test_that("neuralGAM runs OK with Prediction Intervals and poisson response", {
  skip_if_no_keras()

  n <- 10
  formula <- y ~ s(x)
  seed <- 10
  set.seed(seed)
  lambda <- runif(n, 1, 5)
  eta0 <- rpois(n, lambda)

  data <- data.frame(x = 1:10, y = eta0)

  ngam <- neuralGAM(formula,
                    data,
                    num_units = 10,
                    seed = seed,
                    family = "poisson", max_iter_backfitting = 1,
                    max_iter_ls = 1,
                    build_pi = TRUE,
                    alpha = 0.95)

  expect_equal(round(ngam$mse,4), 4.4227)
})


test_that("neuralGAM runs OK with mixed neural-linear model architecture and PI", {
  skip_if_no_keras()

  seed <- 10
  set.seed(seed)

  formula <- y ~ s(x1, num_units = 32) + x2
  data <- data.frame(x1 = 1:10, x2 = 1:10, y = rnorm(10))

  # global num_units = 64 should apply only to x2, not override x1's 32
  ngam <- neuralGAM(
    formula,
    data = data,
    num_units = 64,
    seed = seed,
    max_iter_backfitting = 1,
    max_iter_ls = 1,
    build_pi = TRUE,
    alpha = 0.95
  )
  expect_equal(round(ngam$mse,4), 0.5234)
})


test_that("neuralGAM accepts valid validation_split", {
  skip_if_no_keras()

  formula <- y ~ s(x)

  seed <- 10
  set.seed(seed)

  data <- data.frame(x = 1:10, y = rnorm(10))
  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1, validation_split = 0.2)
  expect_equal(round(ngam$mse,4), 2.7667)
})

test_that("neuralGAM accepts valid w_train", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  data <- data.frame(x = 1:10, y = rnorm(10))
  w <- rep(1, 10)

  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1,  w_train = w)

  expect_equal(round(ngam$mse,4), 2.7667)
})

test_that("neuralGAM accepts build_pi=TRUE with supported losses", {
  skip_if_no_keras()

  formula <- y ~ s(x)
  seed <- 10
  data <- data.frame(x = 1:10, y = rnorm(10))

  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1, build_pi = TRUE,
                    alpha = 0.95, loss = "mse")

  expect_equal(round(ngam$mse, 4), 0.8739)

  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1, build_pi = TRUE,
                    alpha = 0.95, loss = "mae")

  expect_equal(round(ngam$mse, 4), 0.8739)
})


test_that("neuralGAM accepts per-term kernel_initializer and bias_initializer", {
  skip_if_no_keras()

  seed <- 10
  set.seed(seed)
  formula <- y ~ s(x, num_units = 5,
                   kernel_initializer = keras::initializer_he_normal(),
                   bias_initializer = keras::initializer_zeros())
  data <- data.frame(x = 1:10, y = rnorm(10))
  ngam <- neuralGAM(formula, data, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1)
  expect_true(inherits(ngam, "neuralGAM"))
})

test_that("neuralGAM rejects invalid per-term kernel_initializer / bias_initializer", {
  skip_if_no_keras()
  formula <- y ~ s(x, kernel_initializer = 123)
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 5))

  formula <- y ~ s(x, bias_initializer = list(a = 1))
  expect_error(neuralGAM(formula, data, num_units = 5))
})

test_that("neuralGAM accepts valid per-term regularizers", {
  skip_if_no_keras()

  seed <- 10
  set.seed(seed)

  formula <- y ~ s(x,
                   kernel_regularizer = keras::regularizer_l2(1e-4),
                   bias_regularizer = keras::regularizer_l1(1e-4))
  data <- data.frame(x = 1:10, y = rnorm(10))
  ngam <- neuralGAM(formula, data, num_units = 5, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1)
  expect_true(inherits(ngam, "neuralGAM"))
})

test_that("neuralGAM rejects invalid per-term regularizers", {
  skip_if_no_keras()
  formula <- y ~ s(x, kernel_regularizer = "abc")
  data <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(neuralGAM(formula, data, num_units = 5))
})


test_that("neuralGAM accepts per-term num_units and default value for other smooth terms", {
  skip_if_no_keras()

  seed <- 10
  set.seed(seed)

  formula <- y ~ s(x1, num_units = 32) + s(x2)
  data <- data.frame(x1 = 1:10, x2 = 1:10, y = rnorm(10))
  ngam <- neuralGAM(formula, data, num_units = 64, seed = seed, max_iter_backfitting = 1, max_iter_ls = 1)

  # global num_units = 64 should apply only to x2, not override x1's 32
  ngam <- neuralGAM(
    formula,
    data = data,
    num_units = 64,
    seed = seed,
    max_iter_backfitting = 1,
    max_iter_ls = 1
  )

  expect_equal(ngam$formula$np_architecture$x1$num_units, 32)
  expect_equal(ngam$formula$np_architecture$x2$num_units, 64)

})
