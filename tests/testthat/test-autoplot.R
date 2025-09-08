# tests/testthat/test-autoplot-neuralGAM.R
# ------------------------------------------------------------
# Coverage:
# - which = response/link/terms
# - intervals = none/confidence/prediction/both (+ aleatoric on terms)
# - warnings for coerced cases (PI on link; PI on terms -> aleatoric)
# - term validation (single-term only; unknown term)
# - factor vs continuous term plotting
# - newdata path (CI and aleatoric for terms)
# - behavior when aleatoric bands unavailable
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

test_that("autoplot: response/link/terms with expected interval behavior", {
  skip_if_no_keras()

  # Model with both mechanisms available for response and aleatoric for terms
  ngam_both <- neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = dat,
    num_units = 64,
    family = "gaussian",
    uncertainty_method = "both",
    alpha = 0.05
  )

  # ---------------- RESPONSE ----------------
  # CI
  expect_no_warning(
    p_resp_ci <- autoplot(ngam_both, which = "response",
                          interval = "confidence", level = 0.95)
  )
  expect_s3_class(p_resp_ci, "ggplot")

  # PI
  expect_no_warning(
    p_resp_pi <- autoplot(ngam_both, which = "response",
                          interval = "prediction", level = 0.95)
  )
  expect_s3_class(p_resp_pi, "ggplot")

  # Both
  expect_no_warning(
    p_resp_both <- autoplot(ngam_both, which = "response",
                            interval = "both", level = 0.95)
  )
  expect_s3_class(p_resp_both, "ggplot")

  # ---------------- LINK (CI only) ----------------
  expect_no_warning(
    p_link_ci <- autoplot(ngam_both, which = "link",
                          interval = "confidence", level = 0.95)
  )
  expect_s3_class(p_link_ci, "ggplot")

  # Requesting PI on link -> warning (coerced to CI)
  expect_warning(
    p_link_pi_req <- autoplot(ngam_both, which = "link",
                              interval = "prediction", level = 0.95)
  )
  expect_s3_class(p_link_pi_req, "ggplot")

  # ---------------- TERMS (now supports confidence / aleatoric / both) ----------------
  # continuous smooth: epistemic CI
  expect_no_warning(
    p_term_ci <- autoplot(ngam_both, which = "terms",
                          term = "x1", interval = "confidence", level = 0.95)
  )
  expect_s3_class(p_term_ci, "ggplot")

  # continuous smooth: aleatoric residual band (diagnostic)
  expect_no_warning(
    p_term_ale <- autoplot(ngam_both, which = "terms",
                           term = "x1", interval = "aleatoric", level = 0.95)
  )
  expect_s3_class(p_term_ale, "ggplot")

  # continuous smooth: both overlays
  expect_no_warning(
    p_term_both <- autoplot(ngam_both, which = "terms",
                            term = "x1", interval = "both", level = 0.95)
  )
  expect_s3_class(p_term_both, "ggplot")

  # factor term (boxplot + mean ± z*SE; aleatoric ranges if available)
  expect_s3_class(
    autoplot(ngam_both, which = "terms", term = "x2",
             interval = "both", level = 0.95),
    "ggplot"
  )

  # Requesting PI on terms -> warning (coerced to aleatoric)
  expect_warning(
    p_term_pi_req <- autoplot(ngam_both, which = "terms",
                              term = "x1", interval = "prediction", level = 0.95)
  )
  expect_s3_class(p_term_pi_req, "ggplot")
})

test_that("autoplot: terms single-term enforcement and unknown term errors", {
  skip_if_no_keras()

  ngam_lin <- neuralGAM(
    y ~ s(x1) + x2,
    data = dat,
    family = "gaussian",
    num_units = 64
  )

  # must provide exactly one term
  expect_error(autoplot(ngam_lin, which = "terms"))
  expect_error(autoplot(ngam_lin, which = "terms", term = c("x1","x2")))

  # unknown term
  expect_error(autoplot(ngam_lin, which = "terms", term = "nope"))
})

test_that("autoplot: factor vs continuous term plotting (CI)", {
  skip_if_no_keras()

  ngam_lin <- neuralGAM(
    y ~ s(x1) + x2 + x3,
    data = dat,
    num_units = 64,
    family = "gaussian",
    uncertainty_method = "epistemic",
    forward_passes = 5
  )

  # factor term: CI around level means
  p_fac <- autoplot(ngam_lin, which = "terms", term = "x2", interval = "confidence")
  expect_s3_class(p_fac, "ggplot")

  # continuous parametric term: CI available
  p_cont_par <- autoplot(ngam_lin, which = "terms", term = "x3",
                         interval = "confidence", rug = TRUE)
  expect_s3_class(p_cont_par, "ggplot")

  # continuous nonparametric term: CI via MC Dropout SEs
  expect_no_warning(
    p_cont_np <- autoplot(ngam_lin, which = "terms", term = "x1",
                          interval = "confidence", rug = TRUE)
  )
  expect_s3_class(p_cont_np, "ggplot")

  # Requesting aleatoric on a model without quantile heads -> warning but still returns a plot
  expect_warning(
    p_cont_np_ale <- autoplot(ngam_lin, which = "terms", term = "x1",
                              interval = "aleatoric", rug = TRUE)
  )
  expect_s3_class(p_cont_np_ale, "ggplot")
})

test_that("autoplot: response CI works when uncertainty_method='none'", {
  skip_if_no_keras()

  ngam_none <- neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = dat,
    num_units = 64,
    family = "gaussian",
    uncertainty_method = "none"
  )

  # response CI should not be available (SEs via MC on demand) with a warning
  expect_warning(
    p_ci <- autoplot(ngam_none, which = "response", interval = "confidence")
  )
  expect_s3_class(p_ci, "ggplot")

  # link PI request -> warning & CI drawn
  expect_warning(
    p_link_pi <- autoplot(ngam_none, which = "link", interval = "prediction")
  )
  expect_s3_class(p_link_pi, "ggplot")

  # terms PI request -> warning (coerced to aleatoric) and still returns a plot
  expect_warning(expect_warning(
    p_term_pi <- autoplot(ngam_none, which = "terms", term = "x1", interval = "prediction")
  ))
  expect_s3_class(p_term_pi, "ggplot")

  # explicit aleatoric on terms without quantile heads -> warning
  expect_warning(
    p_term_ale_none <- autoplot(ngam_none, which = "terms", term = "x1", interval = "aleatoric")
  )
  expect_s3_class(p_term_ale_none, "ggplot")
})

test_that("autoplot: newdata path (terms CI and aleatoric)", {
  skip_if_no_keras()

  ngam_both <- neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = dat,
    family = "gaussian",
    num_units = 64,
    uncertainty_method = "both",
    alpha = 0.05
  )

  # terms + newdata: CI OK
  expect_no_warning(
    p_term_new_ci <- autoplot(ngam_both, newdata = newx_small,
                              which = "terms", term = "x1", interval = "confidence")
  )
  expect_s3_class(p_term_new_ci, "ggplot")

  # terms + newdata: aleatoric diagnostic band OK
  expect_no_warning(
    p_term_new_ale <- autoplot(ngam_both, newdata = newx_small,
                               which = "terms", term = "x1", interval = "aleatoric")
  )
  expect_s3_class(p_term_new_ale, "ggplot")

  # terms + newdata: PI request -> warning & coerced to aleatoric
  expect_warning(
    p_term_new_pi <- autoplot(ngam_both, newdata = newx_small,
                              which = "terms", term = "x1", interval = "prediction")
  )
  expect_s3_class(p_term_new_pi, "ggplot")
})
