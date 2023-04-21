#' \code{NeuralGAM} summary
#' @description Summary of a fitted NeuralGAM object. Prints
#' the distribution family, model formula, intercept value, sample size,
#' as well as neural network architecture and training history.
#' @param object \code{NeuralGAM} object.
#' @param \ldots Other options.
#' @return The summary of the object
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export

summary.NeuralGAM <- function(object, ...) {
  if (inherits(object, "NeuralGAM")) {
    # Print the object's contents
    ngam <- object
    print(ngam)
    cat("\n\n Model architecture: \n\n")
    print(ngam$model)
    cat("\n\nTraining History \n\n")
    print(ngam$stats)
    invisible(x)
  }else{
    stop("Argument x must be a NeuralGAM object.")
  }
}
