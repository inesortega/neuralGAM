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

    cat(paste("\nNumber of Neural Networks = ", ncol(ngam$x)), sep = " ")
    cat(paste("\nDistribution Family: ", ngam$family))

    fs <- paste("s(", colnames(ngam$x), ")", collapse=" + ", sep="")
    cat("\nFormula: y ~ ", paste(fs, collapse = "+"))
    cat(paste("\nIntercept:", round(ngam$eta0, 5)))
    cat(paste("\nSample size:", nrow(ngam$x)))
    cat("\n\nArchitecture of each Neural Network: \n\n")

    # All the networks have the same architecture, print the first one
    m <- ngam$model[[1]]
    print(m)


    invisible(x)
  }else{
    stop("Argument x must be a NeuralGAM object.")
  }
}
