library(testthat)
library(reticulate)

skip_if_no_keras <- function() {

  if (!tryCatch(
    reticulate::py_module_available("keras"),
    error = function(e) return(FALSE)
  )
  ) skip("keras not available for testing...")
}

# ---- validators -------------------------------------------------------------

test_that("validate_activation() and validate_loss() accept valid names and reject invalid ones", {
  skip_if_no_keras()

  act <- validate_activation("relu")
  los <- validate_loss("mse")
  expect_true(inherits(act, "python.builtin.object"))
  expect_true(inherits(los, "python.builtin.object"))

  expect_error(validate_activation("def-not-an-activation"), "Invalid activation")
  expect_error(validate_loss("not-a-loss"), "Invalid loss")

  # custom functions should pass through unchanged
  my_act <- function(x) x
  my_loss <- function(y, yhat) tensorflow::tf$reduce_mean((y - yhat)^2)
  expect_identical(validate_activation(my_act), my_act)
  expect_identical(validate_loss(my_loss), my_loss)
})

# ---- synthetic data (shared) -----------------------------------------------

.local_synth_data <- function(n = 600L, seed = 123L) {
  set.seed(seed)
  x <- runif(n, -3, 3)
  mu <- sin(x) + 0.2 * x
  sigma <- 0.2 + 0.3 * (x > 0) + 0.1 * abs(x)  # heteroscedastic
  y <- mu + rnorm(n, 0, sigma)
  list(
    x = matrix(x, ncol = 1),
    y = matrix(y, ncol = 1)
  )
}

# ---- helpers ----------------------------------------------------------------

.count_inversions <- function(pred_mat) {
  # pred_mat columns: lwr, upr, mean
  mean(pred_mat[, 1] > pred_mat[, 2])
}

.huber_delta <- function(delta = 0.3) {
  function(y_true, y_pred) {
    tensorflow::tf$keras$losses$Huber(delta = delta)(y_true, y_pred)
  }
}

# ---- PI branch: compile, train, predict -------------------------------------

test_that("PI branch compiles, trains, predicts 3 heads (lwr, upr, mean) with custom mean loss", {
  skip_if_no_keras()

  set.seed(123)
  tf$random$set_seed(123L)

  dat <- .local_synth_data(500L)
  x <- dat$x; y <- dat$y

  model_pi <- build_feature_NN(
    num_units = c(32, 16),
    activation = "relu",
    learning_rate = 1e-3,
    kernel_initializer = "glorot_normal",
    uncertainty_method = "aleatoric",
    alpha = 0.10,                 # 90% PI
    w_mean = 0.2,
    order_penalty_lambda = 1e-3,
    loss = .huber_delta(0.3)      # mean-head loss (custom callable)
  )

  hist <- invisible(model_pi %>% fit(
    x = x, y = y,
    epochs = 6L, batch_size = 64L, verbose = 0L,
    validation_split = 0.2
  ))

  # 3-headed output
  pred <- predict(model_pi, x[1:20,, drop = FALSE], verbose = 0L)
  expect_true(is.matrix(pred))
  expect_identical(ncol(pred), 3L)

  # quick monotone sanity (not strict): inversions should be uncommon after a few epochs
  inv_rate <- .count_inversions(pred)
  expect_true(inv_rate <= 0.40)  # relaxed bound for a tiny training run
})

# ---- Penalty effectiveness (λ > 0 should not worsen inversions) -------------

test_that("Order-penalty (lambda > 0) does not increase inversions vs lambda = 0", {
  skip_if_no_keras()

  set.seed(124)
  tf$random$set_seed(124L)

  dat <- .local_synth_data(600L, seed = 124)
  x <- dat$x; y <- dat$y

  # no-penalty model
  m0 <- build_feature_NN(
    num_units = 32, activation = "relu",
    learning_rate = 1e-3, uncertainty_method = "aleatoric", alpha = 0.10,
    w_mean = 0.2, order_penalty_lambda = 0.0, loss = "mse", seed = 124
  )
  invisible(m0 %>% fit(x, y, epochs = 6L, batch_size = 64L, verbose = 0L))
  p0 <- predict(m0, x, verbose = 0L)
  inv0 <- .count_inversions(p0)

  # penalty model
  m1 <- build_feature_NN(
    num_units = 32, activation = "relu",
    learning_rate = 1e-3, uncertainty_method = "aleatoric", alpha = 0.10,
    w_mean = 0.2, order_penalty_lambda = 1e-2, loss = "mse", seed = 124
  )
  invisible(m1 %>% fit(x, y, epochs = 6L, batch_size = 64L, verbose = 0L))
  p1 <- predict(m1, x, verbose = 0L)
  inv1 <- .count_inversions(p1)

  # Because of stochasticity in a very short training, assert "not worse", with small tolerance
  expect_lte(inv1, inv0 + 1e-3)
})

# ---- Point branch with exotic loss ------------------------------------------

test_that("Point branch compiles/trains with exotic loss = 'logcosh' and yields 1 output", {
  skip_if_no_keras()

  set.seed(125)
  tf$random$set_seed(125L)

  dat <- .local_synth_data(400L, seed = 125)
  x <- dat$x; y <- dat$y

  model_pt <- build_feature_NN(
    num_units = c(32, 16),
    activation = "gelu",
    learning_rate = 1e-3,
    uncertainty_method = "none",
    loss = "logcosh"
  )

  hist <- invisible(model_pt %>% fit(
    x, y, epochs = 6L, batch_size = 64L, verbose = 0L,
    validation_split = 0.2
  ))

  pred <- predict(model_pt, x[1:10,, drop = FALSE], verbose = 0L)
  expect_true(is.matrix(pred))
  expect_identical(ncol(pred), 1L)

  # heuristic: last loss not greater than first (allow tiny jitter)
  tr_loss <- as.numeric(hist$metrics$loss)
  expect_lte(tail(tr_loss, 1), tr_loss[1] + 1e-3)
})

# ---- API argument validation ------------------------------------------------

test_that("Argument validation: alpha and dropout_rate bounds enforced", {
  skip_if_no_keras()

  # alpha out of bounds
  expect_error(
    build_feature_NN(num_units = 8, alpha = 0, uncertainty_method = "aleatoric"),
    "strictly between 0 and 1"
  )
  expect_error(
    build_feature_NN(num_units = 8, alpha = 1, uncertainty_method = "aleatoric"),
    "strictly between 0 and 1"
  )

  # dropout bounds
  expect_error(
    build_feature_NN(num_units = 8, dropout_rate = 0),
    "strictly between 0 and 1"
  )
  expect_error(
    build_feature_NN(num_units = 8, dropout_rate = 1),
    "strictly between 0 and 1"
  )
})

# ---- Huber(0.3) as mean-head loss compiles/trains in PI mode ----------------

test_that("PI branch with Huber(delta=0.3) mean head compiles and trains", {
  skip_if_no_keras()

  set.seed(126)
  tf$random$set_seed(126L)

  dat <- .local_synth_data(450L, seed = 126)
  x <- dat$x; y <- dat$y

  model_h <- build_feature_NN(
    num_units = 16,
    activation = "relu",
    learning_rate = 1e-3,
    uncertainty_method = "aleatoric",
    alpha = 0.10,
    w_mean = 0.2,
    order_penalty_lambda = 1e-3,
    loss = .huber_delta(0.3)
  )

  hist <- invisible(model_h %>% fit(x, y, epochs = 5L, batch_size = 64L, verbose = 0L))
  pred <- predict(model_h, x[1:5,, drop = FALSE], verbose = 0L)

  expect_true(is.matrix(pred))
  expect_identical(ncol(pred), 3L)
  # basic loss sanity
  tr_loss <- as.numeric(hist$metrics$loss)
  expect_true(is.finite(tr_loss[1]))
})
