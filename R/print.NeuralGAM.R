#' Short \code{neuralGAM} summary
#' @description Default print statement for a neuralGAM object.
#' @param x \code{neuralGAM} object.
#' @param \ldots Other arguments.
#' @return The printed output of the object:
#'  \itemize{
#'  \item Distribution family
#'  \item Formula
#'  \item Intercept value
#'  \item Mean Squared Error (MSE)
#'  \item Training sample size
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
#' print(ngam)
#' }

print.neuralGAM <- function(x, ...) {
  if (inherits(x, "neuralGAM")) {
    # Print the class name
    cat("Class: neuralGAM \n")

    # Print the object's contents
    ngam <- x

    cat(paste("\nDistribution Family: ", ngam$family))
    cat(paste("\nFormula: ", ngam$formula$formula))

    if(!is.null(ngam$model$linear)){
      cat(paste(
        "\nIntercept:",
        round(ngam$model$linear$coefficients["(Intercept)"], 4)
      ))
    }
    else{
      cat(paste(
        "\nIntercept:",
        round(ngam$eta0, 4)
      ))
    }

    cat(paste("\nMSE:", round(ngam$mse, 4)))
    cat(paste("\nSample size:", nrow(ngam$x)))

    invisible(x)
  } else{
    stop("Argument x must be a neuralGAM object.")
  }
}

