library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}

test_that("plot_history works with a trained neuralGAM model", {
  skip_if_no_keras()
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("neuralGAM")

  # Simulated data
  set.seed(123)
  n <- 200
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  y <- 2 + x1^2 + sin(x2) + rnorm(n, 0, 0.1)
  df <- data.frame(x1 = x1, x2 = x2, y = y)

  # Train a small neuralGAM with 2 backfitting iterations (fast)
  model <- neuralGAM(
    y ~ s(x1) + s(x2),
    data = df,
    num_units = 8,
    family = "gaussian",
    max_iter_backfitting = 2,
    max_iter_ls = 1,
    learning_rate = 0.01,
    seed = 123,
    verbose = 0
  )

  # Plot for all terms
  p <- plot_history(model$history)
  expect_s3_class(p, "ggplot")

  # Plot specific term
  p_x1 <- plot_history(model$history, select = "x1")
  expect_s3_class(p_x1, "ggplot")
  expect_true(all(p_x1$data$Term == "x1"))

  # Check number of iterations matches
  n_iter_x1 <- length(model$history$x1)
  expect_equal(n_iter_x1, 2)
})
