# tests/testthat/test-mu-eta.R

test_that("mu_eta() validates inputs", {
  expect_error(mu_eta(eta = 0), 'family')   # family missing
  expect_error(mu_eta(family = "gaussian"), 'eta') # eta missing
  expect_error(mu_eta(family = "gamma", eta = 0), 'Unsupported family')
})

test_that("mu_eta() returns correct length and is numeric", {
  eta <- c(-1, 0, 1, 2.5)
  out <- mu_eta("gaussian", eta)
  expect_type(out, "double")
  expect_length(out, length(eta))
})

test_that("gaussian: derivative is 1 for all eta (including extremes)", {
  eta <- c(-100, -30, -1, 0, 1, 30, 100)
  expect_equal(mu_eta("gaussian", eta), rep(1, length(eta)))
})

test_that("binomial: derivative equals mu*(1-mu) and is stable at extremes", {
  # Typical values
  eta <- c(-2, -1, 0, 1, 2)
  mu  <- 1 / (1 + exp(-eta))
  expect_equal(mu_eta("binomial", eta), mu * (1 - mu), tolerance = 1e-12)

  # Symmetry: at eta = 0, derivative = 0.25
  expect_equal(mu_eta("binomial", 0), 0.25, tolerance = 1e-15)

  # Extreme values should be clamped internally and stay finite/small
  # Internally: eta_clamped = ±30 -> mu ≈ 1 / (1 + exp(-±30))
  # After additional mu clamp to [1e-12, 1-1e-12], derivative ≈ 1e-12*(1-1e-12)
  d_lo <- mu_eta("binomial", -1e6)
  d_hi <- mu_eta("binomial",  1e6)
  expect_gt(d_lo, 0)
  expect_gt(d_hi, 0)
  expect_lte(d_lo, 1e-12 + 1e-18)
  expect_lte(d_hi, 1e-12 + 1e-18)

  # Vectorization check
  expect_equal(
    mu_eta("binomial", c(-2, 0, 2)),
    c(mu_eta("binomial", -2), mu_eta("binomial", 0), mu_eta("binomial", 2)),
    tolerance = 1e-12
  )
})

test_that("poisson: derivative equals exp(eta) with eta clamped to [-30, 30]", {
  # Typical values
  eta <- c(-2, -1, 0, 1, 2)
  expect_equal(mu_eta("poisson", eta), exp(eta), tolerance = 1e-12)

  # Clamping: very large positive/negative
  expect_equal(mu_eta("poisson",  50), exp(30), tolerance = 1e-12)
  expect_equal(mu_eta("poisson", -50), exp(-30), tolerance = 1e-12)

  # Vectorization check
  eta_vec <- c(-100, -30, -1, 0, 1, 30, 100)
  expect_equal(
    mu_eta("poisson", eta_vec),
    exp(pmin(pmax(eta_vec, -30), 30)),
    tolerance = 1e-12
  )
})
