#' @title Build and compile a neural network feature model
#'
#' @description
#' Builds and compiles a `keras` neural network for a single smooth term in a
#' `neuralGAM` model.
#' The network can optionally be configured to output **prediction intervals**
#' (lower bound, upper bound, and mean prediction) using a custom quantile loss
#' (`make_quantile_loss()`), or a standard single-output point prediction using
#' any user-specified loss function.
#'
#' When `build_pi = TRUE`, the model outputs three units corresponding to the
#' lower bound, upper bound, and mean prediction, and is compiled with the
#' `make_quantile_loss()` custom loss.
#' When `build_pi = FALSE`, the model outputs a single unit (point prediction)
#' and uses the loss function provided in `loss`.
#'
#' @param num_units Integer or vector of integers. Number of units in the hidden
#'   layer(s). If a vector is provided, multiple dense layers are added
#'   sequentially.
#' @param learning_rate Numeric. Learning rate for the Adam optimizer.
#' @param activation Character string. Activation function to use in hidden layers.
#' @param kernel_initializer Keras initializer object or string. Kernel initializer
#'   for dense layers.
#' @param kernel_regularizer Optional Keras regularizer for kernel weights.
#' @param bias_regularizer Optional Keras regularizer for bias terms.
#' @param bias_initializer Keras initializer object or string. Initializer for
#'   bias terms.
#' @param activity_regularizer Optional Keras regularizer for layer activations.
#' @param loss Loss function to use.
#'   - When `build_pi = TRUE`, this is passed as the mean prediction loss inside
#'     `make_quantile_loss()` (choose from `"mse"`, `"mae"`).
#'   - When `build_pi = FALSE`, this is used directly in `compile()`. Can be any
#'     `keras` built-in loss or custom function.
#' @param name Optional character string. Name assigned to the model.
#' @param alpha Numeric. Desired PI significance level for prediction intervals when
#'   `build_pi = TRUE`. Defaults to 0.05 (i.e., 95\% PI using 2.5\% and 97.5\%
#'   quantiles).
#' @param build_pi Logical. If `TRUE`, builds a model with prediction intervals
#'   (lower bound, upper bound, mean prediction). If `FALSE`, builds a single-output model with the specified loss.
#' @param pi_method Character string indicating the type of uncertainty to estimate in prediction intervals.
#'   Must be one of `"aleatoric"`, `"epistemic"`, or `"both"`:
#'   \itemize{
#'     \item \code{"aleatoric"}: Use quantile regression loss to capture data-dependent (heteroscedastic) noise.
#'     \item \code{"epistemic"}: Use Monte Carlo Dropout with multiple forward passes to capture model uncertainty.
#'     \item \code{"both"}: Combine both quantile estimation and MC Dropout to estimate total predictive uncertainty.
#'   }
#'   Only used when \code{build_pi = TRUE}. Defaults to \code{"aleatoric"}.
#' @param ... Additional arguments passed to `keras::optimizer_adam()`.
#' @return
#' A compiled `keras_model` object ready for training. When `build_pi = TRUE`, the
#' model has three outputs; otherwise, it has one output.
#'
#' @details
#' **Prediction interval mode (`build_pi = TRUE`)**:
#' \itemize{
#'   \item Output layer has 3 units:
#'     \itemize{
#'       \item \code{lwr}: lower bound, \eqn{\tau = \frac{1-\alpha}{2}}
#'       \item \code{upr}: upper bound, \eqn{\tau = 1 - \frac{1-\alpha}{2}}
#'       \item \code{y_hat}: mean prediction
#'     }
#'   \item Loss function is `make_quantile_loss()` which combines two pinball losses
#'         (for lower and upper quantiles) with the chosen mean prediction loss.
#' }
#'
#' **Point prediction mode (`build_pi = FALSE`)**:
#' \itemize{
#'   \item Output layer has 1 unit: point prediction only.
#'   \item Loss function is the one passed in `loss`.
#' }
#'
#' @references
#' Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
#' arXiv preprint arXiv:1412.6980.
#' Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. Econometrica:
#' journal of the Econometric Society, 33–50. *Econometrica*, 46(1), 33–50.
#'
#' @author
#' Ines Ortega-Fernandez, Marta Sestelo
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
           alpha = 0.95,
           build_pi = FALSE,
           pi_method = "none",
           ...) {

    if (missing(num_units)) {
      stop("Argument 'num_units' is missing, with no default.")
    }
    if (!is.numeric(num_units) || any(num_units <= 0) || any(num_units %% 1 != 0)) {
      stop("Argument 'num_units' must be a positive integer or vector of positive integers.")
    }

    if (!is.numeric(learning_rate) || learning_rate <= 0) {
      stop("Argument 'learning_rate' must be a positive numeric value.")
    }

    if (!is.character(activation) || length(activation) != 1) {
      stop("Argument 'activation' must be a single character string.")
    }

    if (!is.null(name) && (!is.character(name) || length(name) != 1)) {
      stop("Argument 'name' must be NULL or a single character string.")
    }

    if (!is.numeric(alpha) || alpha <= 0 || alpha >= 1) {
      stop("Argument 'alpha' must be a numeric value strictly between 0 and 1.")
    }

    if (!is.logical(build_pi) || length(build_pi) != 1) {
      stop("Argument 'build_pi' must be a single logical value (TRUE or FALSE).")
    }

    if (!is.character(loss) && !is.function(loss)) {
      stop("Argument 'loss' must be a character string (keras built-in) or a custom loss function.")
    }

    # ---- Loss compatibility check for PI mode ----

    if (build_pi) {
      if (is.character(loss)) {
        if (!loss %in% c("mse", "mae")) {
          stop("When 'build_pi = TRUE', 'loss' must be either 'mse' or 'mae' to be used in make_quantile_loss().")
        }
      } else {
        stop("When 'build_pi = TRUE', 'loss' must be specified as 'mse' or 'mae' (custom loss functions are not supported for PI mode).")
      }
    }


    # ---- Start model building ----

    model <- keras::keras_model_sequential(name = name)

    model %>% keras::layer_dense(units = 1, input_shape = c(1))

    # if(build_pi) model %>% keras::layer_dropout(rate = 0.2)

    if (is.vector(num_units)) {
      for (units in num_units) {
        model %>% keras::layer_dense(
          units = units,
          kernel_initializer = kernel_initializer,
          kernel_regularizer = kernel_regularizer,
          bias_regularizer = bias_regularizer,
          bias_initializer = bias_initializer,
          activity_regularizer = activity_regularizer,
          activation = activation
        )
        if(build_pi) model %>% keras::layer_dropout(rate = 0.2)
      }
    } else {
      model %>% keras::layer_dense(
        units = num_units,
        kernel_initializer = kernel_initializer,
        kernel_regularizer = kernel_regularizer,
        bias_regularizer = bias_regularizer,
        activity_regularizer = activity_regularizer,
        bias_initializer = bias_initializer,
        activation = activation
      )
      if(build_pi) model %>% keras::layer_dropout(rate = 0.2)
    }

    if (build_pi && pi_method %in% c("aleatoric", "both")) {
      model %>% keras::layer_dense(units = 3, activation = 'linear')  # to return lwr, upr, fit
    } else {
      model %>% keras::layer_dense(units = 1)
    }

    model <- set_compile(model, build_pi, pi_method, alpha, learning_rate, loss, ...)
  }


set_compile <- function(model, build_pi, pi_method, alpha, learning_rate, loss, loss_weights = NULL, ...){
  if(build_pi && pi_method == "aleatoric"){
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = learning_rate, ...),
      loss = make_quantile_loss(alpha = alpha,
                                mean_loss = loss),
      loss_weights = loss_weights
    )
  }
  else{
    model %>% compile(
      optimizer = keras::optimizer_adam(learning_rate = learning_rate, ...),
      loss = loss,   # use provided loss function directly,
      loss_weights = loss_weights
    )
  }

  return(model)
}


make_quantile_loss <- function(alpha = 0.05,
                               mean_loss = "mse") {
  # Quantile values - Convert miscoverage to central-interval quantiles
  tau_low <- alpha / 2
  tau_up  <- 1 - alpha / 2

  function(y_true, y_pred) {
    # Extract outputs: Lower, Upper, Mean
    lwr   <- y_pred[, 1]  # Lower quantile prediction
    upr   <- y_pred[, 2]  # Upper quantile prediction
    y_hat <- y_pred[, 3]  # Mean prediction

    y_t <- y_true[, 1]

    # Pinball (quantile) loss for level tau in (0,1)
    pinball_loss <- function(y, q, tau) {
      e <- y - q
      tensorflow::tf$reduce_mean(
        tensorflow::tf$maximum(tau * e, (tau - 1) * e)
      )
    }

    # Quantile losses
    loss_low <- pinball_loss(y_t, lwr, tau_low)
    loss_up  <- pinball_loss(y_t, upr, tau_up)

    # Mean loss choice
    if (mean_loss == "mse") {
      mean_loss_value <- tensorflow::tf$reduce_mean(tensorflow::tf$square(y_hat - y_t))
    } else if (mean_loss == "mae") {
      mean_loss_value <- tensorflow::tf$reduce_mean(tensorflow::tf$abs(y_hat - y_t))
    } else {
      stop("Unsupported mean_loss: choose 'mse' or 'mae'")
    }

    # Total loss
    total_loss <- loss_low + loss_up + mean_loss_value
    total_loss
  }
}
