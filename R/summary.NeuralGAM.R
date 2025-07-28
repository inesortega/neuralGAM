#' @title Summary of a \code{neuralGAM} model
#' @description
#' Summarizes a fitted \code{neuralGAM} object. Prints the distribution family, model formula,
#' intercept value, training sample size, mean squared error, as well as neural network architecture
#' and training history.
#'
#' @param object A \code{neuralGAM} object.
#' @param ... Additional arguments (currently unused).
#'
#' @return
#' A printed summary of the fitted model with the following components:
#' \describe{
#'   \item{Distribution family}{The family of the fitted model (\code{"gaussian"}, \code{"binomial"}, or \code{"poisson"}).}
#'   \item{Formula}{The formula used for the model.}
#'   \item{Intercept value}{The fitted intercept (\eqn{\eta_0}).}
#'   \item{Mean Squared Error (MSE)}{The training mean squared error.}
#'   \item{Training sample size}{The number of training observations.}
#'   \item{Training history}{Summary of training/validation losses per backfitting iteration.}
#'   \item{Model architecture}{Description of the architecture of each neural network term.}
#' }
#'
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export
#'
#' @examples
#' \dontrun{
#' n <- 24500
#' seed <- 42
#' set.seed(seed)
#'
#' x1 <- runif(n, -2.5, 2.5)
#' x2 <- runif(n, -2.5, 2.5)
#' x3 <- runif(n, -2.5, 2.5)
#'
#' f1 <- x1**2
#' f2 <- 2 * x2
#' f3 <- sin(x3)
#' f1 <- f1 - mean(f1)
#' f2 <- f2 - mean(f2)
#' f3 <- f3 - mean(f3)
#'
#' eta0 <- 2 + f1 + f2 + f3
#' epsilon <- rnorm(n, 0.25)
#' y <- eta0 + epsilon
#' train <- data.frame(x1, x2, x3, y)
#'
#' library(neuralGAM)
#' ngam <- neuralGAM(
#'   y ~ s(x1) + x2 + s(x3),
#'   data = train,
#'   num_units = 1024,
#'   family = "gaussian",
#'   activation = "relu",
#'   learning_rate = 0.001,
#'   bf_threshold = 0.001,
#'   max_iter_backfitting = 10,
#'   max_iter_ls = 10,
#'   seed = seed
#' )
#' summary(ngam)
#' }
summary.neuralGAM <- function(object, ...) {
  if (inherits(object, "neuralGAM")) {
    # Print the object's contents
    ngam <- object
    print(ngam)
    cat("\n\nTraining History: \n\n")
    print(ngam$stats)
    cat("\n\nModel architecture: \n\n")

    if(is.null(ngam$model$linear)){
      print(ngam$model)
    }
    else{
      condition <- !(names(ngam$model) %in% c("linear"))
      print(ngam$model[condition])
      cat("\nLinear model: \n")
      print(ngam$model$linear$coefficients)
    }

    invisible(object)
  }else{
    stop("Argument object must be a neuralGAM object.")
  }
}
