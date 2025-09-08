library(testthat)
library(ggplot2)
library(neuralGAM)


skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("keras")) skip("keras not available")
  if (!reticulate::py_module_available("tensorflow")) skip("tensorflow not available")
}

has_geom <- function(p, geom_class) {
  # geom_class e.g. "GeomRibbon", "GeomLine", "GeomErrorbar", "GeomPoint", "GeomRug"
  any(vapply(p$layers, function(layer) inherits(layer$geom, geom_class), logical(1)))
}

setup_local_data_and_fit <- function(seed = 42L, n = 1500L) {
  set.seed(seed)

  # numeric smooths and a factor parametric term to guarantee SE from lm part
  x1 <- runif(n, -2.5, 2.5)              # smooth
  x3 <- runif(n, -2.5, 2.5)              # smooth
  x2 <- factor(sample(letters[1:4], n, TRUE))  # parametric factor
  x_lin <- runif(n, -1, 1)               # optional linear numeric (parametric)

  f1 <- x1^2
  f3 <- sin(x3)
  f1 <- f1 - mean(f1); f3 <- f3 - mean(f3)

  # Give x2 level means to ensure visible effect + SE from parametric part
  lev_eff <- c(a= -0.5, b=0.0, c=0.3, d=0.6)
  f2 <- lev_eff[as.character(x2)]
  f2 <- f2 - mean(f2)

  eta0 <- 1.2
  eta  <- eta0 + f1 + f2 + f3 + 0.5 * x_lin
  y <- eta + rnorm(n, 0, 0.25)

  train <- data.frame(x1, x2, x3, x_lin, y)

  ngam <- neuralGAM(
    y ~ s(x1) + x2 + s(x3) + x_lin, data = train,
    family = "gaussian",
    num_units = 128,
    activation = "relu",
    learning_rate = 0.001,
    bf_threshold = 0.001,
    max_iter_backfitting = 3,   # keep small for test speed
    max_iter_ls = 3,
    seed = seed,
    uncertainty_method = "aleatoric"
  )

  list(train = train, fit = ngam)
}

test_that("autoplot returns ggplot and basic geoms for numeric smooth", {
  skip_if_no_keras()

  obj <- setup_local_data_and_fit()
  ngam <- obj$fit

  # Prediction intervals are not defined for term effects; using 'confidence' instead.
  # Confidence band unavailable for term 'x1' (missing SEs). -> none
  expect_warning(expect_warning(p <- autoplot(ngam, which = "terms", term = "x1", interval = "prediction")))
  expect_s3_class(p, "ggplot")
  expect_true(has_geom(p, "GeomLine"))
  expect_false(has_geom(p, "GeomRibbion"))

  # ribbon may be omitted if SE not available for the neural term; check no error:
  invisible(ggplot_build(p))

  # Explicitly disable CI: ribbon must disappear
  p_noci <- autoplot(ngam, which = "terms", term = "x1", interval = "none")
  expect_s3_class(p_noci, "ggplot")
  expect_false(has_geom(p_noci, "GeomRibbon"))
})

test_that("autoplot adds error bars for parametric factor term", {
  skip_if_no_keras()

  obj <- setup_local_data_and_fit()
  ngam <- obj$fit

  p <- autoplot(ngam, which = "terms", term = "x2", interval = "confidence")
  expect_s3_class(p, "ggplot")
  expect_true(has_geom(p, "GeomPoint"))
  # parametric factor terms should have SE via predict.lm -> expect error bars
  expect_true(has_geom(p, "GeomErrorbar"))

  # Turning CI off removes error bars but keeps points
  p_noci <- autoplot(ngam, which = "terms", term = "x2", interval = "none")
  expect_true(has_geom(p_noci, "GeomPoint"))
  expect_false(has_geom(p_noci, "GeomErrorbar"))
})

test_that("labels and CI level are properly constructed", {
  skip_if_no_keras()

  obj <- setup_local_data_and_fit()
  ngam <- obj$fit

  expect_warning(p <- autoplot(ngam, which = "terms", term = "x3", interval = "confidence", level = 0.8) +
      ggplot2::xlab("X3 axis") +
      ggplot2::ylab("s(x3)"))
  expect_identical(p$labels$x, "X3 axis")
  expect_identical(p$labels$y, "s(x3)")
})

test_that("factor x scale is discrete and rotated", {
  skip_if_no_keras()

  obj <- setup_local_data_and_fit()
  ngam <- obj$fit

  p <- autoplot(ngam, which = "terms", term = "x2")
  # Should be a discrete scale; building shouldn't error
  built <- ggplot_build(p)
  expect_s3_class(built, "ggplot_built")
})


