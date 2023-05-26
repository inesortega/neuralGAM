#' \code{NeuralGAM} summary
#' @description Summary of a fitted NeuralGAM object. Prints
#' the distribution family, model formula, intercept value, sample size,
#' as well as neural network architecture and training history.
#' @param object \code{NeuralGAM} object.
#' @param \ldots Other options.
#' @return The summary of the object
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export
#' @examples
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
#' library(NeuralGAM)
#' ngam <- NeuralGAM(y ~ s(x1) + x2 + s(x3), data = train,
#'                  num_units = 1024, family = "gaussian",
#'                  activation = "relu",
#'                  learning_rate = 0.001, bf_threshold = 0.001,
#'                  max_iter_backfitting = 10, max_iter_ls = 10,
#'                  seed = seed
#'                  )
#' summary(ngam)
#'

summary.NeuralGAM <- function(object, ...) {
  if (inherits(object, "NeuralGAM")) {
    # Print the object's contents
    ngam <- object
    print(ngam)
    cat("\n\nModel architecture: \n\n")
    print(ngam$model)
    cat("Training History \n\n")
    print(ngam$stats)
    invisible(object)
  }else{
    stop("Argument object must be a NeuralGAM object.")
  }
}
