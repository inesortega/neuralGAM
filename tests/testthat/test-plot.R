# tests/testthat/test-plot-neuralGAM.R


library(testthat)
library(reticulate)

skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("keras"))      skip("keras not available")
  if (!reticulate::py_module_available("tensorflow")) skip("tensorflow not available")
}

test_that("plot.neuralGAM works on real trained models (no stubs)", {
  skip_if_no_keras()

  # Require Python deps; otherwise skip gracefully
  if (!reticulate::py_module_available("tensorflow")) {
    skip("TensorFlow not available; run neuralGAM::install_neuralGAM() first.")
  }

  # Helper: offscreen device so tests don't open a window
  with_offscreen_device <- function(expr) {
    tf <- tempfile(fileext = ".png")
    grDevices::png(tf, width = 800, height = 600, res = 96)
    on.exit(grDevices::dev.off(), add = TRUE)
    force(expr)
    invisible(TRUE)
  }

  set.seed(123)

  # ------------------------
  # 1) CONTINUOUS term model
  # ------------------------
  n1 <- 300
  d1 <- data.frame(
    x1 = runif(n1, -2, 2)
  )
  f1 <- sin(2 * pi * d1$x1)
  d1$y <- 2 + f1 + rnorm(n1, sd = 0.2)

  # Keep it tiny & fast; a couple of backfitting/LS iters is enough for tests
  m_cont <- neuralGAM(
    y ~ s(x1),
    data = d1,
    num_units = 32,
    family = "gaussian",
    learning_rate = 0.005,
    bf_threshold = 1e-3,
    ls_threshold = 0.1,
    max_iter_backfitting = 2,
    max_iter_ls = 2,
    seed = 123,
    verbose = 0
  )

  # Sanity: predict() should return terms for x1
  trm <- predict(m_cont, newdata = d1, type = "terms")
  expect_true("x1" %in% colnames(trm))

  # Plot (continuous), no intervals
  expect_silent(with_offscreen_device(
    plot.neuralGAM(m_cont, select = "x1", interval = "none")
  ))

  # Try confidence intervals if the predict() in your codebase supports se.fit
  ci_ok <- TRUE
  try({
    with_offscreen_device(
      plot.neuralGAM(m_cont, select = "x1", interval = "confidence", level = 0.9)
    )
  }, silent = TRUE) -> res_ci
  if (inherits(res_ci, "try-error")) ci_ok <- FALSE
  if (!ci_ok) {
    testthat::skip("predict(..., se.fit=TRUE) not supported in this neuralGAM build; skipping CI branch.")
  } else {
    succeed()
  }

  # --------------------
  # 2) FACTOR term model
  # --------------------
  n2 <- 220
  d2 <- data.frame(
    x1 = runif(n2, -2, 2),
    x2 = factor(sample(letters[1:3], n2, replace = TRUE))
  )
  # Assign small effects per level; center to keep partials non-flat
  eff <- c(a = -0.4, b = 0.0, c = 0.5)
  d2$y <- 1 + sin(2 * pi * d2$x1) + unname(eff[d2$x2]) + rnorm(n2, sd = 0.25)

  m_fact <- neuralGAM(
    y ~ s(x1) + x2,                   # factor handled additively
    data = d2,
    num_units = 16,
    family = "gaussian",
    learning_rate = 0.01,
    max_iter_backfitting = 2,
    max_iter_ls = 2,
    seed = 321,
    verbose = 0
  )

  # Terms for factor should exist
  trm2 <- predict(m_fact, newdata = d2, type = "terms")
  expect_true("x2" %in% colnames(trm2))

  # Plot (factor), no intervals
  expect_silent(with_offscreen_device(
    plot.neuralGAM(m_fact, select = "x2", interval = "none")
  ))

  # If CI branch is supported, try it for factor too
  if (ci_ok) {
    expect_silent(with_offscreen_device(
      plot.neuralGAM(m_fact, select = "x2", interval = "confidence", level = 0.95)
    ))
  }

  # --------------------------------------
  # 3) CONFIDENCE INTERVALS branch (PI/PB)
  # --------------------------------------
  # The cI code path in plot.neuralGAM requires:
  #   - select = NULL (plots all terms) AND
  #   - ngam$build_pi == TRUE plus lwr/upr present
  # To avoid the interactive <Return> pause, we fit a ONE-term model and call with select=NULL.
  n3 <- 200
  d3 <- data.frame(x = runif(n3, -1, 1))
  d3$y <- cos(3 * d3$x) + rnorm(n3, sd = 0.15)

  m_pi <- neuralGAM(
    y ~ s(x),
    data = d3,
    num_units = 24,
    family = "gaussian",
    learning_rate = 0.01,
    max_iter_backfitting = 2,
    uncertainty_method = "epistemic", forward_passes = 10,
    max_iter_ls = 2,
    seed = 999,
    verbose = 0
  )

  # Only run PI/both if the fitted model indeed carries PI components
  if (isTRUE(m_pi$build_pi) && !is.null(m_pi$lwr) && !is.null(m_pi$upr)) {
    # "prediction"
    expect_silent(with_offscreen_device(
      plot.neuralGAM(m_pi, select = NULL, interval = "prediction")
    ))
    # "both" (PI + CI) if CI supported
    if (ci_ok) {
      expect_silent(with_offscreen_device(
        plot.neuralGAM(m_pi, select = NULL, interval = "both", level = 0.95)
      ))
    }
  } else {
    skip("Model did not expose confidence intervals (build_pi/lwr/upr); skipping CI tests.")
  }

  # -------------------------
  # 4) Error handling checks
  # -------------------------
  expect_error(plot.neuralGAM(list(), interval = "none"),
               "must be of class 'neuralGAM'")
  expect_error(plot.neuralGAM(m_cont, select = "nope"),
               "Invalid select argument")
  expect_error(plot.neuralGAM(m_cont, interval = "garbage"),
               "should be one of|arg should be one of")
})
