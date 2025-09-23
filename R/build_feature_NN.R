#' @title Build and compile a neural network feature model
#'
#' @description
#' Builds and compiles a `keras` neural network for a single smooth term in a
#' `neuralGAM` model.
#'
#' The network can optionally be configured to output **symmetric prediction intervals**
#' (lower bound, upper bound, and mean prediction) using a custom quantile loss
#' (`make_quantile_loss()`), or a standard single-output point prediction using
#' any user-specified loss function.
#'
#' When `uncertainty_method` is `aleatoric` or `both` the model outputs three units corresponding to the
#' lower bound, upper bound, and mean prediction, and is compiled with
#' `make_quantile_loss(alpha, mean_loss, ...)`. In any other case, the model
#' outputs a single unit (point prediction) and uses the loss function provided in `loss`.
#'
#' @param num_units Integer or vector of integers. Number of units in the hidden
#'   layer(s). If a vector is provided, multiple dense layers are added sequentially.
#' @param learning_rate Numeric. Learning rate for the Adam optimizer.
#' @param activation Character string or function. Activation function to use in hidden layers.
#'   If character, it must be valid for `tf$keras$activations$get()`.
#' @param kernel_initializer Keras initializer object or string. Kernel initializer for dense layers.
#' @param kernel_regularizer Optional Keras regularizer for kernel weights.
#' @param bias_regularizer Optional Keras regularizer for bias terms.
#' @param bias_initializer Keras initializer object or string. Initializer for bias terms.
#' @param activity_regularizer Optional Keras regularizer for layer activations.
#' @param loss Loss function to use.
#'   - When `uncertainty_method` is `aleatoric` or `both`, this is the **mean-head loss** inside
#'     `make_quantile_loss()` and can be any `keras` built-in loss name (e.g., `"mse"`, `"mae"`, `"huber"`,
#'     `"logcosh"`, …) or a custom function.
#'   - In any other case, this is used directly in `compile()`.
#' @param name Optional character string. Name assigned to the model.
#' @param alpha Numeric. Desired significance level for **symmetric** prediction intervals.
#'   Defaults to 0.05 (i.e., 95% PI using quantiles alpha/2 and 1−alpha/2).
#' @param w_mean Non-negative numeric. Weight for the mean-head loss within the composite PI loss.
#' @param order_penalty_lambda Non-negative numeric. Strength of a soft monotonicity penalty
#'   `ReLU(lwr - upr)` to discourage interval inversions.
#' @param uncertainty_method Character string indicating the type of uncertainty to estimate in prediction intervals.
#'   Must be one of `"none"`, `"aleatoric"`, `"epistemic"`, or `"both"`.
#' @param dropout_rate Numeric in (0,1). Dropout rate used when `uncertainty_method %in% c("epistemic","both")`.
#' @param seed Random seed.
#' @param ... Additional arguments passed to `keras::optimizer_adam()`.
#'
#' @return A compiled `keras_model` object ready for training.
#'
#' @details
#' **Prediction interval mode (`uncertainty_method %in% c("aleatoric", "both")`)**:
#' \itemize{
#'   \item Output layer has 3 units:
#'     \itemize{
#'       \item \code{lwr}: lower bound, \eqn{\tau = \alpha/2}
#'       \item \code{upr}: upper bound, \eqn{\tau = 1 - \alpha/2}
#'       \item \code{y_hat}: mean prediction
#'     }
#'   \item Loss function is `make_quantile_loss()` which combines two pinball losses
#'         (for lower and upper quantiles) with the chosen mean prediction loss and an optional
#'         non-crossing penalty.
#' }
#'
#' **Point prediction mode (`uncertainty_method %in% c("none", "epistemic")`)**:
#' \itemize{
#'   \item Output layer has 1 unit: point prediction only.
#'   \item Loss function is the one passed in `loss`.
#' }
#'
#' @references
#' Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
#' arXiv:1412.6980.
#' Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50.
#' @author Ines Ortega-Fernandez, Marta Sestelo
#'
#' @inheritDotParams neuralGAM
#' @importFrom keras keras_model_sequential
#' @importFrom keras layer_dense
#' @importFrom keras optimizer_adam
#' @importFrom magrittr %>%
#' @importFrom keras fit
#' @importFrom keras compile
#' @keywords internal
build_feature_NN <-
  function(num_units,
           learning_rate = 0.001,
           activation = "relu",
           kernel_initializer = "glorot_normal",
           kernel_regularizer = NULL,
           bias_regularizer = NULL,
           bias_initializer = 'zeros',
           activity_regularizer = NULL,
           loss = "mse",
           name = NULL,
           alpha = 0.05,
           w_mean = 0.1,
           order_penalty_lambda = 0.0,
           uncertainty_method = "none",
           dropout_rate = 0.1,
           seed = NULL,
           ...) {

    if (missing(num_units)) stop("Argument 'num_units' is missing, with no default.")
    if (!is.numeric(num_units) || any(num_units <= 0) || any(num_units %% 1 != 0))
      stop("Argument 'num_units' must be a positive integer or vector of positive integers.")

    if (!is.numeric(learning_rate) || learning_rate <= 0)
      stop("Argument 'learning_rate' must be a positive numeric value.")

    # validate activation & loss
    act_callable  <- validate_activation(activation)
    loss_callable <- validate_loss(loss)

    if (!is.null(name) && (!is.character(name) || length(name) != 1))
      stop("Argument 'name' must be NULL or a single character string.")

    if (!is.numeric(alpha) || alpha <= 0 || alpha >= 1)
      stop("Argument 'alpha' must be a numeric value strictly between 0 and 1.")

    if (!is.numeric(dropout_rate) || dropout_rate <= 0 || dropout_rate >= 1)
      stop("Argument 'dropout_rate' must be a numeric value strictly between 0 and 1.")

    # ---- build ----
    model <- keras::keras_model_sequential(name = name)
    model %>% keras::layer_dense(units = 1, input_shape = c(1))

    add_hidden <- function(units) {
      model %>% keras::layer_dense(
        units = units,
        kernel_initializer = kernel_initializer,
        kernel_regularizer = kernel_regularizer,
        bias_regularizer = bias_regularizer,
        bias_initializer = bias_initializer,
        activity_regularizer = activity_regularizer,
        activation = act_callable
      )
      if (uncertainty_method %in% c("epistemic", "both")) {
        model %>% keras::layer_dropout(rate = dropout_rate, seed = seed)
      }
    }

    if (is.vector(num_units)) for (units in num_units) add_hidden(units) else add_hidden(num_units)

    if (uncertainty_method %in% c("aleatoric", "both")) {
      model %>% keras::layer_dense(units = 3, activation = 'linear')  # lwr, upr, mean
    } else {
      model %>% keras::layer_dense(units = 1, activation = 'linear')
    }

    model <- set_compile(model, uncertainty_method, alpha, learning_rate,
                         loss = loss_callable,
                         w_mean = w_mean,
                         order_penalty_lambda = order_penalty_lambda,
                         ...)
  }


set_compile <- function(model, uncertainty_method, alpha, learning_rate,
                        loss, w_mean = 0.1, order_penalty_lambda = 0.0, ...) {

  if (uncertainty_method %in% c("aleatoric", "both")) {
    model %>% keras::compile(
      optimizer = keras::optimizer_adam(learning_rate = learning_rate, ...),
      loss = make_quantile_loss(alpha = alpha,
                                mean_loss = loss,
                                w_mean = w_mean,
                                order_penalty_lambda = order_penalty_lambda),
    )
  } else {
    model %>% keras::compile(
      optimizer = keras::optimizer_adam(learning_rate = learning_rate, ...),
      loss = loss
    )
  }
  model
}

#' @keywords internal
make_quantile_loss <- function(alpha = 0.05,
                               mean_loss,
                               w_mean = 0.1,
                               order_penalty_lambda = 0.0) {

  if (!is.numeric(alpha) || alpha <= 0 || alpha >= 1)
    stop("'alpha' must be in (0,1).")

  tau_low <- alpha / 2
  tau_up  <- 1 - alpha / 2

  pinball_loss <- function(y, q, tau_scalar) {
    e <- y - q
    tau <- tensorflow::tf$cast(tau_scalar, dtype = e$dtype)
    tensorflow::tf$reduce_mean(
      tensorflow::tf$maximum(tau * e, (tau - 1) * e)
    )
  }

  order_penalty <- function(q_lwr, q_upr) {
    # Penalize inverted quantiles
    if (order_penalty_lambda <= 0) {
      return(tensorflow::tf$constant(0, dtype = q_lwr$dtype))
    }
    tensorflow::tf$reduce_mean(tensorflow::tf$nn$relu(q_lwr - q_upr))
  }

  function(y_true, y_pred) {
    # Keep 2D shapes (avoid unwanted broadcasting)
    y_t  <- y_true[, 1, drop = FALSE]

    lwr  <- y_pred[, 1, drop = FALSE]
    upr  <- y_pred[, 2, drop = FALSE]
    yhat <- y_pred[, 3, drop = FALSE]

    # quantile (pinball) losses
    loss_low <- pinball_loss(y_t, lwr, tau_low)
    loss_up  <- pinball_loss(y_t, upr, tau_up)

    # mean-head loss (any keras/custom loss)
    mean_loss_value <- mean_loss(y_t, yhat)

    # total
    total_loss <- loss_low + loss_up + w_mean * mean_loss_value

    # soft non-crossing penalty on predicted quantiles
    if (order_penalty_lambda > 0) {
      total_loss <- total_loss + order_penalty_lambda * order_penalty(lwr, upr)
    }

    total_loss
  }
}

#' Validate/resolve a Keras activation
#'
#' @param activation character or function. If character, must be a valid
#'   tf.keras activation identifier (e.g., "relu", "gelu", "swish", ...).
#' @return a callable activation (Python callable) or the original R function.
#' @examples \dontrun{
#' library(neuralGAM)
#' act <- neuralGAM:::validate_activation("relu")     # ok
#' act <- neuralGAM:::validate_activation(function(x) x)  # custom
#' }
#' @keywords internal
validate_activation <- function(activation) {
  if (is.character(activation)) {
    obj <- tryCatch(
      tensorflow::tf$keras$activations$get(activation),
      error = function(e) NULL
    )
    if (is.null(obj)) {
      stop(sprintf(
        "Invalid activation '%s'. Use a valid tf.keras activation name or an R function.",
        activation
      ))
    }
    return(obj)  # normalized Python callable
  }
  if (is.function(activation)) return(activation)
  stop("`activation` must be a character string or a function.")
}


#' Validate/resolve a Keras loss
#'
#' @param loss character or function. If character, must be a valid
#'   tf.keras loss identifier (e.g., "mse", "mae", "huber", "logcosh", ...).
#' @return a callable loss (Python callable) or the original R function.
#' @examples \dontrun{
#' library(neuralGAM)
#' L <- neuralGAM:::validate_loss("huber")             # ok (Huber with default delta)
#' L <- neuralGAM:::validate_loss(function(y,t) tensorflow::tf$reduce_mean((y-t)^2))  # custom
#' }
#' @keywords internal
validate_loss <- function(loss) {
  if (is.character(loss)) {
    obj <- tryCatch(
      tensorflow::tf$keras$losses$get(loss),
      error = function(e) NULL
    )
    if (is.null(obj)) {
      stop(sprintf(
        "Invalid loss '%s'. Use a valid tf.keras loss name or provide a custom R function.",
        loss
      ))
    }
    return(obj)  # normalized Python callable
  }
  if (is.function(loss)) return(loss)
  stop("`loss` must be a character string or a function.")
}
