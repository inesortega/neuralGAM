#' \code{neuralGAM} summary
#' @description Summary of a fitted \code{neuralGAM} object. Prints
#' the distribution family, model formula, intercept value, sample size,
#' as well as neural network architecture and training history.
#' @param object \code{neuralGAM} object.
#' @param \ldots Other options.
#' @return The summary of the object:
#'  \itemize{
#'  \item Distribution family
#'  \item Formula
#'  \item Intercept value
#'  \item Mean Squared Error (MSE)
#'  \item Training sample size
#'  \item Training History
#'  \item Model Architecture
#'}
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export
#' @examples \donttest{
#'
#' n <- 24500
#'
#' seed <- 42
#' set.seed(seed)
#'
#' x1 <- runif(n, -2.5, 2.5)
#' x2 <- runif(n, -2.5, 2.5)
#' x3 <- runif(n, -2.5, 2.5)
#'
#' f1 <-x1**2
#' f2 <- 2*x2
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
#' ngam <- neuralGAM(y ~ s(x1) + x2 + s(x3), data = train,
#'                  num_units = 1024, family = "gaussian",
#'                  activation = "relu",
#'                  learning_rate = 0.001, bf_threshold = 0.001,
#'                  max_iter_backfitting = 10, max_iter_ls = 10,
#'                  seed = seed
#'                  )
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
