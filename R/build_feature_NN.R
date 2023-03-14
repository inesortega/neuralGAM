#'Build and compile a single Neural Network
#'
#'
#' @description Builds and compiles a neural network using the keras library.
#' The architecture of the neural network is configurable using the
#' \code{"num_units"} parameter. A single integer with the number of hidden
#' units builds a shallow (single layer) neural network. Deep Neural Networks
#' can be built by specifying the size of each hidden layer in the
#'  \code{"num_units"} parameter. For example, \code{"list(32,32,32)"} generates
#'  a DNN with three layers and 32 neurons per layer.
#' @author Ines Ortega-Fernandez, Marta Sestelo and Nora M. Villanueva.
#' @param num_units number of hidden units (for shallow neural networks) or
#' list of hidden units per layer
#' @param learning_rate learning rate for the Adam optimizer (Kingma, 2014).
#' Defaults to \code{0.001}
#' @param kernel_initializer kernel initializer for the Dense layers.
#' Defaults to Xavier Initializer
#' @param \ldots Other options.
#' @return compiled Neural Network
#' @importFrom keras keras_model_sequential
#' @importFrom keras layer_dense
#' @importFrom keras optimizer_adam

#' @examples
#'
#' # Build a Shallow NN with 32 hidden units:
#' model <- build_feature_NN(num_units=32, learning_rate=0.001,
#' kernel_initializer="glorot_normal")
#'
#' # Build a Deep NN with three hidden layers with 32 hidden units on each layer
#' model <- build_feature_NN(num_units=list(32,32,32),
#' learning_rate=0.001, kernel_initializer="glorot_normal")
#'
#' @references
#' Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

build_feature_NN <- function(num_units, learning_rate=0.001, kernel_initializer="glorot_normal", ...){

 if (missing(num_units)){
    stop("Argument \"num_units\" is missing, with no default")
  }

  if (class(num_units) != "numeric" & !(is.list(num_units))) {
    stop("Argument num_units must be an integer or a list of integers")
  }

  model <- keras::keras_model_sequential()

  model %>% keras::layer_dense(units = 1, input_shape = c(1))

  if (is.list(num_units)) {
    for (units in num_units){
      model %>% keras::layer_dense(units = units, kernel_initializer = kernel_initializer, activation = 'relu', ...)
    }
  } else {
    model %>% keras::layer_dense(units = num_units, kernel_initializer = kernel_initializer, activation = 'relu', ...)
  }

  model %>% keras::layer_dense(units = 1)
  adam <- keras::optimizer_adam(learning_rate = learning_rate, ...)


  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = adam
  )
  return(model)
}

