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

  # Check if object argument is missing or NULL
  if (missing(object) || is.null(object)) {
    stop("Please provide a fitted NeuralGAM object as the object argument.")
  }

  # Check if object is of class "NeuralGAM"
  if (!inherits(object, "NeuralGAM")) {
    stop("The object argument must be a fitted NeuralGAM object.")
  }

  # check that all parameters are OK
  if (missing(newdata)) {
    x <- ngam$x
  }
  else{
    if(type != "terms" && ncols(x) != length(ngam$model)){
      stop("newdata needs to have the same components as the fitted ngam model")
    }
    x <- newdata
  }

  # Check if type argument is missing or NULL
  if (missing(type) || is.null(type)) {
    stop("Please provide a value for the type argument.")
  }

  # Check if type argument is valid
  valid_types <- c("link", "terms", "response")
  if (!type %in% valid_types) {
    stop("The value of the type argument is invalid. Valid options are {link, terms, response}.")
  }

  if (type == "terms" && !is.null(terms) && !all(terms %in% colnames(x))){
    stop(paste("Invalid terms. Valid options are: ", paste(colnames(x), collapse=",")))
  }

  # Check if newdata columns match ngam$model columns
  if (type != "terms" && !is.null(newdata) && !all(colnames(newdata) %in% colnames(ngam$x))) {
    stop("The newdata argument does not have the same columns as the fitted ngam model.")
  }

  f <- x*NA
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
