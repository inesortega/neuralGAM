# tests/testthat/test-autoplot-neuralGAM.R
# ------------------------------------------------------------
# Coverage (epistemic-only):
# - which = response / link / terms
# - interval = none / confidence
# - term validation (single-term only; unknown term)
# - factor vs continuous term plotting
# - newdata path
# ------------------------------------------------------------

library(testthat)
library(reticulate)

skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("keras"))      skip("keras not available")
  if (!reticulate::py_module_available("tensorflow")) skip("tensorflow not available")
}

set.seed(123)
n <- 300
x1 <- runif(n, -2.5, 2.5)                    # smooth
x2 <- factor(sample(letters[1:3], n, TRUE))  # factor
x3 <- runif(n, -2.5, 2.5)                    # linear
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

test_that("autoplot: response/link/terms with interval none/confidence", {
  skip_if_no_keras()

  ngam <- neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = dat,
    num_units = 64,
    family = "gaussian",
    uncertainty_method = "epistemic",
    forward_passes = 10,
    verbose = 0
  )

  # ---------------- RESPONSE ----------------
  p_resp_none <- autoplot(ngam, which = "response", interval = "none")
  expect_s3_class(p_resp_none, "ggplot")

  p_resp_ci <- autoplot(ngam, which = "response", interval = "confidence", level = 0.95)
  expect_s3_class(p_resp_ci, "ggplot")

  # ---------------- LINK ----------------
  p_link_none <- autoplot(ngam, which = "link", interval = "none")
  expect_s3_class(p_link_none, "ggplot")

  p_link_ci <- autoplot(ngam, which = "link", interval = "confidence", level = 0.95)
  expect_s3_class(p_link_ci, "ggplot")

  # ---------------- TERMS (single term) ----------------
  p_term_none <- autoplot(ngam, which = "terms", term = "x1", interval = "none")
  expect_s3_class(p_term_none, "ggplot")

  p_term_ci <- autoplot(ngam, which = "terms", term = "x1", interval = "confidence", level = 0.95)
  expect_s3_class(p_term_ci, "ggplot")
})

test_that("autoplot: terms single-term enforcement and unknown term errors", {
  skip_if_no_keras()

  ngam <- neuralGAM(
    y ~ s(x1) + x2,
    data = dat,
    family = "gaussian",
    num_units = 64,
    uncertainty_method = "epistemic",
    forward_passes = 5
  )

  # must provide exactly one term
  expect_error(autoplot(ngam, which = "terms"))
  expect_error(autoplot(ngam, which = "terms", term = c("x1","x2")))

  # unknown term
  expect_error(autoplot(ngam, which = "terms", term = "nope"))
})

test_that("autoplot: factor vs continuous term plotting (CI)", {
  skip_if_no_keras()

  ngam <- neuralGAM(
    y ~ s(x1) + x2 + x3,
    data = dat,
    num_units = 64,
    family = "gaussian",
    uncertainty_method = "epistemic",
    forward_passes = 5,
    verbose = 0
  )

  # factor term: CI around level means (whiskers)
  p_fac <- autoplot(ngam, which = "terms", term = "x2", interval = "confidence")
  expect_s3_class(p_fac, "ggplot")

  # continuous parametric term (x3): CI band
  p_cont_par <- autoplot(ngam, which = "terms", term = "x3",
                         interval = "confidence", rug = TRUE)
  expect_s3_class(p_cont_par, "ggplot")

  # continuous smooth term (x1): CI band via MC-dropout SEs
  p_cont_np <- autoplot(ngam, which = "terms", term = "x1",
                        interval = "confidence", rug = TRUE)
  expect_s3_class(p_cont_np, "ggplot")
})

test_that("autoplot: newdata path for terms and panels", {
  skip_if_no_keras()

  ngam <- neuralGAM(
    y ~ s(x1) + x2 + s(x3),
    data = dat,
    family = "gaussian",
    num_units = 64,
    uncertainty_method = "epistemic",
    forward_passes = 8,
    verbose = 0
  )

  # response/link with newdata
  expect_s3_class(autoplot(ngam, which = "response", interval = "confidence", newdata = newx_small), "ggplot")
  expect_s3_class(autoplot(ngam, which = "link",     interval = "confidence", newdata = newx_small), "ggplot")

  # terms with newdata
  expect_s3_class(autoplot(ngam, which = "terms", term = "x1",
                           interval = "confidence", newdata = newx_small), "ggplot")
  expect_s3_class(autoplot(ngam, which = "terms", term = "x1",
                           interval = "none",       newdata = newx_small), "ggplot")
})
