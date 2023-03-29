#' Short \code{NeuralGAM} summary
#' @description Default print statement for a NeuralGAM object. Prints
#' the distribution family, model formula, intercept value, number of terms
#' (neural networks), as well as neural network architecture.
#' @param x \code{NeuralGAM} object.
#' @param \ldots Other options.
#' @return The printed output of the object
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export

print.NeuralGAM <- function(x = object, ...) {
  if (inherits(x, "NeuralGAM")) {

    # Print the class name
    cat("Class: NeuralGAM \n")

    # Print the object's contents
    ngam <- x

    cat(paste("\nDistribution Family: ", ngam$family))
    cat(paste("\nFormula: ", ngam$formula))
    cat(paste("\nIntercept:", round(ngam$beta0, 4)))
    cat(paste("\nMean Squared Error:", round(ngam$mse, 4)))
    cat(paste("\nSample size:", nrow(ngam$x)))

    invisible(x)
  }else{
    stop("Argument x must be a NeuralGAM object.")
  }
}
