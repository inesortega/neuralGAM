#' Produces predictions from a fitted \code{NeuralGAM} object
#' @description Takes a fitted \code{NeuralGAM} object produced by
#' \code{NeuralGAM()} and produces predictions given a new set of values for the model covariates.
#' @param object a fitted `NeuralGAM` object
#' @param newdata A data frame or list containing the values of covariates at which
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
predict.NeuralGAM <- function(object, newdata = NULL, type = "link", terms = NULL, ...) {

  ngam <- object

  if (missing(newdata)) {
    x <- ngam$x
  }
  else{
    x <- newdata
  }

  if (type == "terms" && !is.null(terms) && !all(terms %in% colnames(x))){
    stop(paste("Invalid terms. Valid options are: ", paste(colnames(x), collapse=",")))
  }

  f <- x*NA

  if (type != "link" && type != "terms" && type != "response"){
    stop("Invalid type. Valid options are {link, terms, response}")
  }

  for (i in 1:ncol(x)) {

    if (type == "terms" && !is.null(terms)){
      # compute only certain terms
      if(colnames(x)[[i]] %in% terms){
        f[, colnames(x)[[i]]] <- ngam$model[[i]]$predict(x[, i])
      }
      else{
        next
      }
    }
    else{
      # consider all terms
      f[, colnames(x)[[i]]] <- ngam$model[[i]]$predict(x[, i])
    }
  }

  if(type == "terms"){
    if (!is.null(terms)){
      f <- f[,terms]
      colnames(f) <- terms
    }
    return(f)
  }

  eta <- rowSums(f) + ngam$eta0
  if(type == "link"){
    # Return the linear predictor
    return(eta)
  }

  if(type == "response"){
    y <- link(family=ngam$family, eta)
    return(y)
  }
}
