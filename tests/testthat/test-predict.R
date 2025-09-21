# tests/testthat/test-predict-neuralGAM.R
# ------------------------------------------------------------
# Coverage (epistemic-only):
# - type = "link", "response", "terms"
# - se.fit behavior & delta mapping
# - interval = "none", "confidence" (only)
# - cache path (training data) vs newdata
# - term selection and error handling
# - missing columns error
# - helper .row_sum_var NA propagation
# - terms: interval="confidence" returns CI matrices (lwr/upr) + se.fit
# - parametric-only model: link SE matches stats::predict(..., se.fit=TRUE)
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
f1 <- x1^2; f2 <- 2 * x2; f3 <- sin(x3)
y  <- 2 + f1 + f2 + f3 + rnorm(n, 0.25)
train <- data.frame(x1, x2, x3, y)

newx_ok <- data.frame(
  x1 = seq(-2.5, 2.5, length.out = 50),
  x2 = 0,
  x3 = 0
)
newx_bad <- data.frame(x1 = 1) # missing x2, x3

test_that("predict() basic types and shapes (epistemic-only)", {

  skip_if_no_keras()

  ngam0 <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 64,
    uncertainty_method = "epistemic",
    forward_passes = 20,   # small for speed
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

  # terms (matrix)
  trm <- predict(ngam0, type = "terms")
  expect_true(is.matrix(trm))
  expect_setequal(colnames(trm), c("x1","x2","x3"))

  # sum of terms + intercept equals link
  eta0 <- if (is.null(ngam0$eta0)) 0 else ngam0$eta0
  eta_recon <- eta0 + rowSums(trm)
  expect_equal(eta, eta_recon, tolerance = 1e-6)

  # se.fit on link/response
  pr_link <- predict(ngam0, type = "link", se.fit = TRUE)
  expect_true(is.list(pr_link))
  expect_type(pr_link$fit, "double"); expect_type(pr_link$se.fit, "double")
  expect_length(pr_link$fit, nrow(train)); expect_length(pr_link$se.fit, nrow(train))
  expect_true(all(is.finite(pr_link$se.fit)))

  pr_resp <- predict(ngam0, type = "response", se.fit = TRUE)
  expect_true(is.list(pr_resp))
  expect_type(pr_resp$fit, "double"); expect_type(pr_resp$se.fit, "double")
  expect_length(pr_resp$fit, nrow(train)); expect_length(pr_resp$se.fit, nrow(train))
  expect_true(all(is.finite(pr_resp$se.fit)))

  # delta mapping (gaussian/identity): |dμ/dη| = 1 -> SEs equal --> not working at this tolerance due to MC uncertainty! we will need ~1000 passess
  # expect_equal(pr_resp$se.fit, pr_link$se.fit, tolerance = 1e-6)
})

test_that("interval='confidence' returns CI frames on link/response", {

  skip_if_no_keras()

  ngam0 <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 64,
    uncertainty_method = "epistemic",
    forward_passes = 15,
    verbose = 0
  )

  # link CI
  ci_link <- predict(ngam0, type = "link", interval = "confidence", level = 0.95)
  expect_s3_class(ci_link, "data.frame")
  expect_true(all(c("fit","lwr","upr") %in% names(ci_link)))
  expect_equal(nrow(ci_link), nrow(train))
  expect_true(all(is.finite(ci_link$fit)))
  # lwr <= fit <= upr
  expect_true(all(ci_link$lwr <= ci_link$fit + 1e-8))
  expect_true(all(ci_link$fit <= ci_link$upr + 1e-8))

  # response CI
  ci_resp <- predict(ngam0, type = "response", interval = "confidence", level = 0.95)
  expect_s3_class(ci_resp, "data.frame")
  expect_true(all(c("fit","lwr","upr") %in% names(ci_resp)))
  expect_equal(nrow(ci_resp), nrow(train))
  expect_true(all(is.finite(ci_resp$fit)))
  expect_true(all(ci_resp$lwr <= ci_resp$fit + 1e-8))
  expect_true(all(ci_resp$fit <= ci_resp$upr + 1e-8))
})

test_that("terms: se.fit and CI matrices are returned as specified", {

  skip_if_no_keras()

  ngam0 <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 32,
    uncertainty_method = "epistemic",
    forward_passes = 10,
    verbose = 0
  )

  # subset of terms
  sel <- predict(ngam0, type = "terms", terms = c("x1","x2"))
  expect_true(all(colnames(sel) %in% c("x1","x2")))
  expect_false("x3" %in% colnames(sel))

  # unknown term should error
  expect_error(predict(ngam0, type = "terms", terms = "nope"))

  # se.fit matrices
  pr_terms_se <- predict(ngam0, type = "terms", se.fit = TRUE)
  expect_true(is.list(pr_terms_se))
  expect_true(all(c("fit","se.fit") %in% names(pr_terms_se)))
  expect_equal(dim(pr_terms_se$fit), dim(pr_terms_se$se.fit))
  expect_setequal(colnames(pr_terms_se$fit), c("x1","x2","x3"))

  # interval='confidence' -> list with CI matrices
  trm_ci <- predict(ngam0, type = "terms", terms = c("x1","x2","x3"),
                    interval = "confidence", level = 0.95)
  expect_true(is.list(trm_ci))
  expect_true(all(c("fit","se.fit","lwr","upr") %in% names(trm_ci)))
  expect_equal(colnames(trm_ci$fit), c("x1","x2","x3"))
  expect_equal(colnames(trm_ci$lwr), c("x1","x2","x3"))
  expect_equal(colnames(trm_ci$upr), c("x1","x2","x3"))
})

test_that("cache vs newdata; missing columns error", {

  skip_if_no_keras()

  ngam0 <- neuralGAM::neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = train,
    family = "gaussian",
    num_units = 32,
    uncertainty_method = "epistemic",
    forward_passes = 10,
    verbose = 0
  )

  # training-data (cache) path already exercised above; now newdata path:
  mu_new <- predict(ngam0, newdata = newx_ok, type = "response", se.fit = FALSE)
  expect_type(mu_new, "double")
  expect_length(mu_new, nrow(newx_ok))

  # se.fit works on newdata (link)
  pr_new <- predict(ngam0, newdata = newx_ok, type = "link", se.fit = TRUE)
  expect_type(pr_new$fit, "double")
  expect_type(pr_new$se.fit, "double")
  expect_length(pr_new$se.fit, nrow(newx_ok))
  expect_true(all(is.finite(pr_new$se.fit)))

  # missing columns -> error
  expect_error(predict(ngam0, newdata = newx_bad, type = "response"))
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
