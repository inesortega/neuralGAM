# tests/testthat/test-autoplot-neuralGAM.R
# ------------------------------------------------------------
# Coverage:
# - which = response/link/terms
# - intervals = none/confidence/prediction/both
# - warnings for unsupported cases (PI on link; missing bands)
# - term validation (single-term only; unknown term)
# - factor vs continuous term plotting
# - newdata path (no PI ribbons for terms off-cache)
# - PI ribbons when PI/CIs are available (aleatoric path; tiny models)
# ------------------------------------------------------------

library(testthat)
library(reticulate)

skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("keras")) skip("keras not available")
  if (!reticulate::py_module_available("tensorflow")) skip("tensorflow not available")
}

set.seed(123)
n <- 300
x1 <- runif(n, -2.5, 2.5)
x2 <- factor(sample(letters[1:3], n, TRUE))
x3 <- runif(n, -2.5, 2.5)
f1 <- sin(x1)
f2 <- ifelse(x2 == "a", 1, ifelse(x2 == "b", -1, 0))
f3 <- 0.5 * x3
y  <- 2 + f1 + f2 + f3 + rnorm(n, 0.25)
dat <- data.frame(x1, x2, x3, y)

newx_small <- data.frame(
  x1 = seq(-1, 1, length.out = 20),
  x2 = factor(sample(levels(x2), 20, TRUE), levels = levels(x2)),
  x3 = 0
)

test_that("autoplot: response/link basic", {
  skip_if_no_keras()

  # model (with a smooth on x1) -> SEs available via on-demand MC
  ngam_lin <- neuralGAM(
    y ~ s(x1) + x2 + x3,
    data = dat,
    num_units = 128,
    family = "gaussian",
    pi_method = "both"
  )

  # RESPONSE: CI should be drawn, no warning
  expect_no_warning(
    p_resp_ci <- autoplot(ngam_lin, which = "response",
                          interval = "confidence", level = 0.95)
  )
  expect_s3_class(p_resp_ci, "ggplot")

  # LINK: CI should be drawn, no warning
  expect_no_warning(
    p_link_ci <- autoplot(ngam_lin, which = "link",
                          interval = "confidence", level = 0.95)
  )
  expect_s3_class(p_link_ci, "ggplot")

  # TERMS (continuous smooth): CI should be drawn, no warning
  expect_no_warning(
    p_term_ci <- autoplot(ngam_lin, which = "terms",
                          term = "x1", interval = "confidence", level = 0.95)
  )
  expect_s3_class(p_term_ci, "ggplot")

  # TERMS (factor): should not error (SE bars shown if available)
  expect_s3_class(
    autoplot(ngam_lin, which = "terms", term = "x2",
             interval = "confidence", level = 0.95),
    "ggplot"
  )

  # ---- NO PI: PI requests warn but plots are returned ----
  ngam_no_pi <- neuralGAM(
    y ~ s(x1) + x2 + x3,
    data = dat,
    num_units = 128,
    family = "gaussian",
    pi_method = "none"
  )

  # Response PI -> warn (no PI machinery)
  expect_warning(expect_warning(
    p_resp_pi <- autoplot(ngam_no_pi, which = "response",
                          interval = "prediction", level = 0.95)
  ))
  expect_s3_class(p_resp_pi, "ggplot")

  # Link PI -> warn (PI not defined on link; CI drawn)
  expect_warning(
    p_link_pi <- autoplot(ngam_no_pi, which = "link",
                          interval = "prediction", level = 0.95)
  )
  expect_s3_class(p_link_pi, "ggplot")

  # Terms PI -> warn (no PI ribbons for terms without PI)
  expect_warning(
    p_term_pi <- autoplot(ngam_no_pi, which = "terms",
                          term = "x1", interval = "prediction", level = 0.95)
  )
  expect_s3_class(p_term_pi, "ggplot")
})

test_that("autoplot: terms single-term enforcement and unknown term errors", {
  skip_if_no_keras()

  ngam_lin <- neuralGAM(
    y ~ s(x1) + x2,
    data = dat,
    family = "gaussian",
    num_units = 128
  )

  # must provide exactly one term
  expect_error(autoplot(ngam_lin, which = "terms"))
  expect_error(autoplot(ngam_lin, which = "terms", term = c("x1","x2")))

  # unknown term
  expect_error(autoplot(ngam_lin, which = "terms", term = "nope"))
})

test_that("autoplot: terms for factor (boxplot + mean ± z*SE) and continuous (line + CI)", {
  skip_if_no_keras()

  ngam_lin <- neuralGAM(
    y ~ s(x1) + x2 + x3,
    data = dat,
    num_units = 128,
    family = "gaussian"
  )

  # factor term
  p_fac <- autoplot(ngam_lin, which = "terms", term = "x2", interval = "confidence")
  expect_s3_class(p_fac, "ggplot")

  # continuous term (parametric): CI available
  p_cont <- autoplot(ngam_lin, which = "terms", term = "x3",
                     interval = "confidence", rug = TRUE)
  expect_s3_class(p_cont, "ggplot")

  # continuous term (non-parametric): CI not available and no warning
  expect_no_warning(
    p_cont_x1 <- autoplot(ngam_lin, which = "terms", term = "x1",
                          interval = "confidence", rug = TRUE)
  )
  expect_s3_class(p_cont_x1, "ggplot")

  # continuous term (non-parametric): PI not available
  expect_warning(
    p_cont_x1 <- autoplot(ngam_lin, which = "terms", term = "x1",
                          interval = "prediction", rug = TRUE)
  )
})

test_that("autoplot: response prediction intervals (pi method aleatoric)", {
  skip_if_no_keras()

  # include nonparametric terms so PI heads exist
  ngam_ale <- neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = dat,
    family = "gaussian",
    num_units = 64,
    pi_method = "aleatoric",
    alpha = 0.05
  )

  # Response + PI ribbon
  expect_no_warning(
    p_resp_pi <- autoplot(ngam_ale, which = "response",
                          interval = "prediction", level = 0.95)
  )
  expect_s3_class(p_resp_pi, "ggplot")

  # Both CI and PI ribbons (order: PI then CI ribbons)
  expect_no_warning(
    p_resp_both <- autoplot(ngam_ale, which = "response",
                            interval = "both", level = 0.95)
  )
  expect_s3_class(p_resp_both, "ggplot")
  expect_true("GeomRibbon" %in% class(p_resp_both$layers[[1]]$geom))
})

test_that("autoplot: terms (nonparametric continuous) CI & PI ribbons when on training cache", {
  skip_if_no_keras()

  ngam_ale <- neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = dat,
    family = "gaussian",
    num_units = 128,
    pi_method = "aleatoric",
    alpha = 0.05
  )

  # On training cache, PI ribbons for NP term should be available (no warning)
  expect_no_warning(
    p_term_cache <- autoplot(ngam_ale, which = "terms",
                             term = "x1", interval = "prediction")
  )
  expect_s3_class(p_term_cache, "ggplot")

  # With newdata, term-level PI ribbons are unavailable by design -> warning
  expect_warning(
    p_term_new <- autoplot(ngam_ale, newdata = newx_small,
                           which = "terms", term = "x1", interval = "prediction")
  )
  expect_s3_class(p_term_new, "ggplot")

  # Both on newdata should also warn (no term PI off-cache)
  expect_warning(
    p_term_ci <- autoplot(ngam_ale, newdata = newx_small,
                          which = "terms", term = "x1", interval = "both")
  )
})

test_that("autoplot: response CI availability when pi_method is none", {
  skip_if_no_keras()

  ngam_lin <- neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = dat,
    num_units = 128,
    family = "gaussian"
  )

  # response CI should be available now via on-demand SEs (no warning)
  expect_no_warning(
    p_ci <- autoplot(ngam_lin, which = "response", interval = "confidence")
  )
  expect_s3_class(p_ci, "ggplot")
})
