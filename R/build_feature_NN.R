#' Build and compile a single Neural Network
#' @description Builds and compiles a neural network using the keras library.
#' The architecture of the neural network is configurable using the
#' @param name Neural Network name.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @inheritParams neuralGAM
#' @inheritDotParams neuralGAM
#' @return compiled Neural Network
#' @importFrom keras keras_model_sequential
#' @importFrom keras layer_dense
#' @importFrom keras optimizer_adam
#' @importFrom magrittr %>%
#' @importFrom keras fit
#' @importFrom keras compile
#' @keywords internal
#' @references
#' Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

build_feature_NN <-
  function(num_units,
           learning_rate = 0.001,
           activation = "relu",
           kernel_initializer = "glorot_normal",
           kernel_regularizer = NULL,
           bias_regularizer = NULL,
           bias_initializer = 'zeros',
           activity_regularizer = NULL,
           loss = "mean_squared_error",
           name = NULL,
           ...) {

    if (missing(num_units)) {
      stop("Argument \"num_units\" is missing, with no default")
    }

    if (!is.numeric(num_units) & !(is.vector(num_units))) {
      stop("Argument \"num_units\" must be an integer or a vector of integers")
    }

    if (!is.numeric(learning_rate)) {
      stop("Error: 'learning_rate' argument should be of numeric type.")
    }

    # Check if the argument 'name' is NULL or a character string
    if (!is.null(name) && !is.character(name)) {
      stop("Error: 'name' argument should be NULL or a character string.")
    }

    model <- keras::keras_model_sequential(name = name)


    model %>% keras::layer_dense(units = 1, input_shape = c(1))

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
    }

    model %>% keras::layer_dense(units = 1)

    adam <-
      keras::optimizer_adam(learning_rate = learning_rate, ...)

    model$compile(loss = loss,
                  optimizer = adam)
    return(model)
  }
