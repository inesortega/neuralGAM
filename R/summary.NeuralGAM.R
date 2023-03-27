#' \code{NeuralGAM} summary
#' @description Summary of a fitted NeuralGAM object. Prints
#' the distribution family, model formula, intercept value, sample size,
#' as well as neural network architecture and training history.
#' @param x \code{NeuralGAM} object.
#' @param \ldots Other options.
#' @return The summary of the object
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export

summary.NeuralGAM <- function(x = object, ...) {
  if (inherits(x, "NeuralGAM")) {
    # Print the object's contents
    ngam <- x
    print(ngam)
    cat("\n\nTraining History \n\n")
    print(ngam$stats)
    invisible(x)
  }else{
    stop("Argument x must be a NeuralGAM object.")
  }
}
