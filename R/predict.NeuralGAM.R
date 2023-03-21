#' Produces predictions from a fitted \code{NeuralGAM} object
#' @description Takes a fitted \code{NeuralGAM} object produced by
#' \code{NeuralGAM()} and produces predictions given a new set of values for the model covariates.
#' @param object a fitted `NeuralGAM` object
#' @param x A data frame or list containing the values of covariates at which
#' predictions are required. If not provided, the function returns the predictions
#' for the original training data.
#' @param type when \code{type="link"} (default), the linear
#' predictor is returned. When \code{type="terms"} each component of the linear
#' predictor is returned separately on each column of a \code{data.frame}. When
#' \code{type="response"} predictions on the scale of the response are returned.
#' @param terms If \code{type="terms"}, then only results for the terms named
#' in this list will be returned. If \code{NULL} then no terms are excluded (default).
#' @param \ldots Other options.
#' @return Predicted values according to \code{type} parameter.
#' @export
#' @examples
#' # Obtain linear predictor
#' predict(object = ngam, x = x_test, type = "link")
#' # Obtain each component of the linear predictor separately on each column of a data.frame
#' predict(object = ngam, x = x_test, type = "terms")
#' # Obtain only some terms
#' predict(object = ngam, x = x_test, type = "terms", terms=c("X0", "X1"))
#' # Obtain predictions on the scale of the response
#' predict(object = ngam, x = x_test, type = "response")
#' Obtain predictions from the training data
#' predict(object = ngam, type = "response")
predict.NeuralGAM <- function(object, x = NULL, type = "link", terms = NULL, ...) {

  ngam <- object

  if (missing(x)) {
    x <- ngam$x

  }

  f <- x * 0

  if (type != "link" && type != "terms" && type != "response"){
    stop("Invalid type. Valid options are {link, terms, response}")
  }

  if(type == "terms" && !is.null(terms)){
    # return only terms present in terms array
    x <- subset(x, select = terms)
    f <- x * 0
  }
  else{
    # All terms considered:

    for (i in 1:ncol(x)) {
      f[, i] <- ngam$model[[i]]$predict(x[, i])
    }
    eta <- rowSums(f) + ngam$eta0

    if(type == "link"){
      # Return the linear predictor
      return(eta)
    }
    if(type == "terms"){
      return(f)
    }
    if(type == "response"){
      y <- link(family=ngam$family, eta)
      return(y)
    }
  }

}
