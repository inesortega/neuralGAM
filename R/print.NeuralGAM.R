#' Short \code{NeuralGAM} summary
#'
#' \code{\link{NeuralGAM}} summary
#' @param x \code{NeuralGAM} object.
#' @param \ldots Other options.
#' @return The printed output of the object
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export
#' @examples
#' library(NeuralGAM)
#' data(train)
#' head(train)
#' X_train <- train[c("X0", "X1", "X2")]
#' y_train <- train$y
#'
#' ngam <- NeuralGAM(
#'   x = X_train, y = y_train, num_units = 1024, family = "gaussian",
#'   learning_rate = 0.001, bf_threshold = 0.001,
#'   max_iter_backfitting = 10, max_iter_ls = 10
#' )
#' ngam


print.NeuralGAM <- function(x = object, ...) {
  if (inherits(x, "NeuralGAM")) {

    # Print the class name
    cat("Class: NeuralGAM \n")

    # Print the object's contents

    cat(paste("\nNumber of Neural Networks = ", ncol(ngam$x)), sep = " ")
    cat(paste("\nDistribution Family: ", ngam$family))

    fs <- paste("f(", colnames(ngam$x), ")", collapse=" + ", sep="")
    cat("\nFormula: y ~ ", paste(fs, collapse = "+"))
    cat(paste("\nIntercept:", ngam$eta0))
    cat("\n\nArchitecture of each Neural Network: \n\n")

    # All the networks have the same architecture, print the first one
    m <- ngam$model[[1]]
    print(m)
    # dims <- list()
    # for(i in 2:(length(m$layers)-1)){
    #   # get dimension of intermediate layers (discarding input and output layer)
    #   dims <- append(dims, as.numeric(m$layers[[i]]$output$shape$as_list()[-1]))
    # }


    invisible(x)
  }else{
    stop("Argument x must be a NeuralGAM object.")
  }
}
