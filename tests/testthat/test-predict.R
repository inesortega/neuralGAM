# tests/testthat/test-predict-neuralGAM.R
# ------------------------------------------------------------
# Coverage:
# - type = "link", "response", "terms"
# - se.fit behavior & delta mapping
# - interval = "none", "confidence", "prediction", "both"
# - cache path (training data) vs newdata
# - term selection and error handling
# - link-scale PI warning (returns CI)
# - missing columns error
# - helper .row_sum_var NA propagation
# - terms ignore `interval` for return shape (unless diagnostic_bands)
# - response "both" without PI machinery -> CI + NA PIs (+ warning)
# - NEW: terms `diagnostic_bands` returns lwr_ale/upr_ale (when available)
# ------------------------------------------------------------

library(testthat)
library(reticulate)

skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("keras"))      skip("keras not available")
  if (!reticulate::py_module_available("tensorflow")) skip("tensorflow not available")
}

set.seed(42)
n <- 400
x1 <- runif(n, -2.5, 2.5)
x2 <- runif(n, -2.5, 2.5)
x3 <- runif(n, -2.5, 2.5)
f1 <- x1^2
f2 <- 2 * x2
f3 <- sin(x3)
y  <- 2 + f1 + f2 + f3 + rnorm(n, 0.25)
train <- data.frame(x1, x2, x3, y)

newx_ok <- data.frame(
  x1 = seq(-2.5, 2.5, length.out = 50),
  x2 = 0,
  x3 = 0
)
newx_bad <- data.frame(x1 = 1) # missing x2, x3

test_that("predict() basic types without PIs", {

  skip_if_no_keras()

  ngam0 <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 64,
    verbose = 0
  )

  # link
  eta <- predict(ngam0, type = "link")
  expect_type(eta, "double")
  expect_length(eta, nrow(train))

  # response
  mu <- predict(ngam0, type = "response")
  expect_type(mu, "double")
  expect_length(mu, nrow(train))

  # terms
  trm <- predict(ngam0, type = "terms")
  expect_setequal(colnames(trm), c("x1","x2","x3"))

  # sum of terms + intercept equals link
  eta0 <- if (is.null(ngam0$eta0)) 0 else ngam0$eta0
  eta_recon <- eta0 + rowSums(trm)
  expect_equal(eta, eta_recon, tolerance = 1e-6)

  # se.fit on link/response
  pr_link <- predict(ngam0, type = "link", se.fit = TRUE)
  expect_type(pr_link$fit, "double"); expect_type(pr_link$se.fit, "double")
  expect_length(pr_link$fit, nrow(train)); expect_length(pr_link$se.fit, nrow(train))

  pr_resp <- predict(ngam0, type = "response", se.fit = TRUE)
  expect_type(pr_resp$fit, "double"); expect_type(pr_resp$se.fit, "double")
  expect_length(pr_resp$fit, nrow(train)); expect_length(pr_resp$se.fit, nrow(train))

  # delta mapping (gaussian/identity): |dμ/dη| = 1
  expect_equal(pr_resp$se.fit, pr_link$se.fit, tolerance = 1e-6)

  # terms with SEs (mgcv-style for linear + MC-dropout for smooths)
  pr_terms <- predict(ngam0, type = "terms", se.fit = TRUE)
  expect_equal(dim(pr_terms$fit), dim(pr_terms$se.fit))

  # intervals: confidence on link (not available)
  expect_warning(ci_link <- predict(ngam0, type = "link", interval = "confidence"))
  expect_s3_class(ci_link, "data.frame")
  expect_true(all(c("fit","lwr","upr") %in% names(ci_link)))
  expect_true(any(is.na(ci_link$lwr)) && any(is.na(ci_link$upr)))

  # intervals: confidence on response
  expect_warning(ci_resp <- predict(ngam0, type = "response", interval = "confidence"))
  expect_s3_class(ci_resp, "data.frame")
  expect_true(all(c("fit","lwr","upr") %in% names(ci_resp)))
  expect_true(any(is.na(ci_resp$lwr)) && any(is.na(ci_resp$upr)))

  # intervals: prediction unavailable when no PI -> expect warning + NA bands
  expect_warning(pi_resp <- predict(ngam0, type = "response", interval = "prediction"))
  expect_true(all(is.na(pi_resp$lwr) | is.na(pi_resp$upr)))

  # link + prediction -> warning (PI not defined on link), returns CI columns with finite values if SEs exist
  expect_warning(expect_warning(pi_on_link <- predict(ngam0, type = "link", interval = "prediction")))
  expect_true(all(c("fit","lwr","upr") %in% names(pi_on_link)))

})

test_that("term selection, interval ignored for terms (shape), diagnostic_bands optional, and errors", {

  skip_if_no_keras()

  ngam0 <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 32,
    verbose = 0
  )

  # subset of terms
  sel <- predict(ngam0, type = "terms", terms = c("x1","x2"))
  expect_true(all(colnames(sel) %in% c("x1","x2")))
  expect_false("x3" %in% colnames(sel))

  # unknown term should error
  expect_error(predict(ngam0, type = "terms", terms = "nope"))

  # newdata missing columns should error
  expect_error(predict(ngam0, newdata = newx_bad, type = "response"))

  # interval argument is ignored for terms in terms of RETURN SHAPE:
  # matrix returned when interval != "none" and se.fit = FALSE
  pr_terms_none <- predict(ngam0, type = "terms")
  pr_terms_conf <- predict(ngam0, type = "terms", interval = "confidence")
  expect_equal(colnames(pr_terms_none), colnames(pr_terms_conf))

  # With se.fit=TRUE:
  # - interval="none" -> list(fit, se.fit)
  # - interval!="none" -> matrix only (legacy behavior)
  pr_terms_se_none <- predict(ngam0, type = "terms", se.fit = TRUE, interval = "none")
  pr_terms_se_conf <- predict(ngam0, type = "terms", se.fit = TRUE, interval = "confidence")
  expect_equal(colnames(pr_terms_se_none$fit),   colnames(pr_terms_se_conf))
  expect_equal(colnames(pr_terms_se_none$se.fit), colnames(pr_terms_se_conf))
})

test_that("terms diagnostic_bands return lwr_ale/upr_ale when quantile heads exist", {

  skip_if_no_keras()

  ngam_ale_terms <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 64,
    uncertainty_method = "aleatoric",  # enables quantile heads
    alpha = 0.05,
    verbose = 0
  )

  # Request diagnostic bands (per-term residual quantiles)
  pr_diag <- predict(ngam_ale_terms, type = "terms",
                     terms = c("x1","x2","x3"),
                     diagnostic_bands = TRUE)

  # Should be a list with fit + aleatoric bands matrices
  expect_true(is.list(pr_diag))
  expect_true(all(c("fit","lwr_ale","upr_ale") %in% names(pr_diag)))
  expect_equal(colnames(pr_diag$fit),     c("x1","x2","x3"))
  expect_equal(colnames(pr_diag$lwr_ale), c("x1","x2","x3"))
  expect_equal(colnames(pr_diag$upr_ale), c("x1","x2","x3"))

  # Bands should be numeric same nrow as training
  expect_equal(nrow(pr_diag$lwr_ale), nrow(train))
  expect_equal(nrow(pr_diag$upr_ale), nrow(train))
})

test_that("uncertainty_method = 'aleatoric': PIs on response, not link", {

  skip_if_no_keras()

  ngam_ale <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 64,
    uncertainty_method = "aleatoric",
    alpha = 0.05,
    verbose = 0
  )

  # response + prediction interval (from per-term quantile heads)
  pi_df <- predict(ngam_ale, type = "response", interval = "prediction", level = 0.95)
  expect_s3_class(pi_df, "data.frame")
  expect_true(all(c("fit","lwr","upr") %in% names(pi_df)))
  expect_true(any(is.finite(pi_df$lwr)) && any(is.finite(pi_df$upr)))

  # both CI and PI; with aleatoric-only training, CI may be NA -> expect warning
  expect_warning(both_df <- predict(ngam_ale, type = "response", interval = "both", level = 0.95))
  expect_true(all(c("fit","lwr_ci","upr_ci","lwr_pi","upr_pi") %in% names(both_df)))
  expect_true(any(is.na(both_df$lwr_ci)) && any(is.na(both_df$upr_ci)))
  expect_true(any(is.finite(both_df$lwr_pi)) && any(is.finite(both_df$upr_pi)))

  # on link, PI not defined => single warning; returns CI-shaped frame (lwr/upr may be NA if no epistemic SEs)
  expect_warning(expect_warning(link_pred <- predict(ngam_ale, type = "link", interval = "prediction")))
  expect_true(all(c("fit","lwr","upr") %in% names(link_pred)))
  expect_true(any(is.na(link_pred$lwr)) && any(is.na(link_pred$upr)))
})

test_that("uncertainty_method = 'both': MC-dropout path works (tiny passes)", {

  skip_if_no_keras()

  ngam_both <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 32,
    uncertainty_method = "both",
    alpha = 0.10,           # wider bands
    forward_passes = 5,     # tiny for speed
    verbose = 0
  )

  res_both <- predict(ngam_both, type = "response", interval = "both", level = 0.90)
  expect_s3_class(res_both, "data.frame")
  expect_true(all(c("fit","lwr_ci","upr_ci","lwr_pi","upr_pi") %in% names(res_both)))
  expect_true(any(is.finite(res_both$lwr_ci)) && any(is.finite(res_both$upr_ci)))
  expect_true(any(is.finite(res_both$lwr_pi)) && any(is.finite(res_both$upr_pi)))

  # link CI exists; "prediction" on link warns and returns CI-shaped data.frame
  link_ci <- predict(ngam_both, type = "link", interval = "confidence")
  expect_s3_class(link_ci, "data.frame")
  expect_warning(link_pi <- predict(ngam_both, type = "link", interval = "prediction"))
  expect_true(all(c("fit","lwr","upr") %in% names(link_pi)))
})

test_that("newdata path works with/without PI", {

  skip_if_no_keras()

  dat2 <- train
  dat2$z <- with(dat2, 2*x1 - 0.5*x2 + rnorm(nrow(dat2), 0, 0.2))

  # no PI
  ngam_lin <- neuralGAM::neuralGAM(
    z ~ x1 + s(x2),
    data = dat2,
    num_units = 128,
    family = "gaussian",
    verbose = 0
  )
  newdf <- data.frame(x1 = c(-1, 0, 1), x2 = c(0.1, -0.2, 0.3))
  pr_nd <- predict(ngam_lin, newdata = newdf, type = "link", se.fit = TRUE)
  expect_type(pr_nd$fit, "double")
  expect_type(pr_nd$se.fit, "double")
  expect_length(pr_nd$se.fit, nrow(newdf))
  expect_true(any(is.na(pr_nd$se.fit)))

  # with PI machinery (aleatoric)
  ngam_lin_pi <- neuralGAM::neuralGAM(
    z ~ x1 + s(x2),
    data = dat2,
    num_units = 128,
    family = "gaussian",
    uncertainty_method = "aleatoric",
    verbose = 0
  )
  pr_nd2 <- predict(ngam_lin_pi, newdata = newdf, type = "link", se.fit = TRUE)
  expect_type(pr_nd2$fit, "double")
  expect_type(pr_nd2$se.fit, "double")
  expect_length(pr_nd2$fit, nrow(newdf))

  ngam_lin_pi <- neuralGAM::neuralGAM(
    z ~ x1 + s(x2),
    data = dat2,
    num_units = 128,
    family = "gaussian",
    uncertainty_method = "epistemic", forward_passes = 5,
    verbose = 0
  )
  pr_nd2 <- predict(ngam_lin_pi, newdata = newdf, type = "link", se.fit = TRUE)
  expect_type(pr_nd2$fit, "double")
  expect_type(pr_nd2$se.fit, "double")
  expect_length(pr_nd2$fit, nrow(newdf))
})

test_that("response 'both' without PI returns CI + NA PIs (with warning)", {

  skip_if_no_keras()

  ngam0 <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 64,
    verbose = 0
  )
  expect_warning(expect_warning(both_none <- predict(ngam0, type = "response", interval = "both")))
  expect_true(all(c("fit","lwr_ci","upr_ci","lwr_pi","upr_pi") %in% names(both_none)))
  expect_true(any(is.na(both_none$lwr_ci)) && any(is.na(both_none$upr_ci)))
  expect_true(all(is.na(both_none$lwr_pi) | is.na(both_none$upr_pi)))
})

test_that("response 'both' with PI returns CI + PIs (with warning)", {

  skip_if_no_keras()

  ngam0 <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 64,
    uncertainty_method = "both",
    verbose = 0
  )
  both_ngam <- predict(ngam0, type = "response", interval = "both")
  expect_true(all(c("fit","lwr_ci","upr_ci","lwr_pi","upr_pi") %in% names(both_ngam)))
  expect_true(any(is.finite(both_ngam$lwr_ci)) && any(is.finite(both_ngam$upr_ci)))
  expect_true(all(is.finite(both_ngam$lwr_pi) | is.finite(both_ngam$upr_pi)))
})

test_that(".row_sum_var NA-propagates per row", {
  var_mat <- rbind(
    c(0.1, 0.2, 0.3),
    c(NA,  0.2, 0.3),
    c(0.0,  NA, NA)
  )
  rs <- neuralGAM:::`.row_sum_var`(var_mat)
  expect_equal(rs[1], sum(var_mat[1, ]))
  expect_true(is.na(rs[2]))
  expect_true(is.na(rs[3]))
})
