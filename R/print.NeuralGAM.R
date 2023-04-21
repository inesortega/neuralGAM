#' Short \code{NeuralGAM} summary
#' @description Default print statement for a NeuralGAM object. Prints
#' the distribution family, model formula, intercept value, number of terms
#' (neural networks), as well as neural network architecture.
#' @param x \code{NeuralGAM} object.
#' @param \ldots Other arguments.
#' @return The printed output of the object
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export
#' @examples
#'
#' n <- 24500
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
#' train <- data.frame(x1, x2, x3, y, f1, f2, f3)
#'
#' library(NeuralGAM)
#' ngam <- NeuralGAM(y ~ s(x1) + x2 + s(x3), data = train,
#'                  num_units = 1024, family = "gaussian",
#'                  activation = "relu",
#'                  learning_rate = 0.001, bf_threshold = 0.001,
#'                  max_iter_backfitting = 10, max_iter_ls = 10
#'                  )
#'
#' print(ngam)

print.NeuralGAM <- function(x, ...) {
  if (inherits(x, "NeuralGAM")) {
    # Print the class name
    cat("Class: NeuralGAM \n")

    # Print the object's contents
    ngam <- x

    cat(paste("\nDistribution Family: ", ngam$family))
    cat(paste("\nFormula: ", ngam$formula$formula))
    cat(paste(
      "\nIntercept:",
      round(ngam$model$linear$coefficients["(Intercept)"], 4)
    ))
    cat(paste("\nMean Squared Error:", round(ngam$mse, 4)))
    cat(paste("\nSample size:", nrow(ngam$x)))

    invisible(x)
  } else{
    stop("Argument x must be a NeuralGAM object.")
  }
}

