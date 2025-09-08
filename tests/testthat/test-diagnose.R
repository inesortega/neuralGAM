# tests/testthat/test-diagnose-neuralGAM.R

test_that("diagnose returns a patchwork object (gaussian, fast)", {
  skip_if_not_installed("neuralGAM")
  skip_if_not_installed("keras")
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("patchwork")

  set.seed(2025)

  # small synthetic data
  n <- 200
  x1 <- runif(n, -2, 2)
  f  <- sin(x1)
  y  <- 2 + f + rnorm(n, sd = 0.25)
  dat <- data.frame(x1 = x1, y = y)

  # fit a very small model (keep fast)
  # NOTE: these args are common in neuralGAM; adjust if your API differs.
  expect_no_error({
    m <- neuralGAM(
      y ~ s(x1),
      data = dat,
      family = "gaussian",
      num_units = c(16, 16),
      dropout_rate = 0.1,
      validation_split = 0.1,
      verbose = 1,
      seed = 2025
    )
  })

  # default qq_method = "uniform" with tiny n_uniform for speed
  expect_no_error({
    p <- diagnose(
      m,
      qq_method   = "uniform",
      n_uniform   = 50,     # small, fast
      n_simulate  = 10,     # irrelevant here but harmless
      residual_type = "deviance",
      hist_bins   = 10,
      point_alpha = 0.4
    )
    # Should be a patchwork object
    expect_s3_class(p, "patchwork")
  })
})

test_that("diagnose works with qq_method='simulate' and 'normal'", {
  skip_if_not_installed("neuralGAM")
  skip_if_not_installed("patchwork")

  set.seed(2026)

  n <- 180
  x1 <- runif(n, -2, 2)
  y  <- 1 + cos(1.5 * x1) + rnorm(n, sd = 0.3)
  dat <- data.frame(x1, y)

  m <- neuralGAM(
    y ~ s(x1),
    data = dat,
    family = "gaussian",
    num_units = 16,
    dropout_rate = 0.1,
    validation_split = 0.1,
    seed = 2026
  )

  # qq_method = "simulate" with small n_simulate (fast)
  expect_no_error({
    p_sim <- diagnose(
      m,
      qq_method  = "simulate",
      n_simulate = 20,
      level = 0.90,
      hist_bins = 12
    )
    expect_s3_class(p_sim, "patchwork")
  })

  # qq_method = "normal" (fastest)
  expect_no_error({
    p_norm <- diagnose(
      m,
      qq_method = "normal",
      hist_bins = 12
    )
    expect_s3_class(p_norm, "patchwork")
  })
})

test_that("diagnose accepts external data + response and quantile residuals", {
  skip_if_not_installed("neuralGAM")
  skip_if_not_installed("patchwork")

  set.seed(2027)

  n_train <- 160
  n_test  <- 80
  x_tr <- runif(n_train, -2, 2)
  x_te <- runif(n_test,  -2, 2)
  f_tr <- 0.5 * x_tr^2
  f_te <- 0.5 * x_te^2
  y_tr <- f_tr + rnorm(n_train, sd = 0.25)
  y_te <- f_te + rnorm(n_test,  sd = 0.25)

  dat_tr <- data.frame(x1 = x_tr, y = y_tr)
  dat_te <- data.frame(x1 = x_te, y = y_te)

  m <- neuralGAM(
    y ~ s(x1),
    data = dat_tr,
    family = "gaussian",
    num_units = 16,
    dropout_rate = 0.1,
    validation_split = 0.1,
    seed = 2027
  )

  # Use out-of-sample data and quantile residuals (works for Gaussian too)
  expect_no_error({
    p <- diagnose(
      m,
      data = dat_te,
      response = "y",
      qq_method = "normal",
      residual_type = "quantile",
      hist_bins = 10
    )
    expect_s3_class(p, "patchwork")
  })
})
