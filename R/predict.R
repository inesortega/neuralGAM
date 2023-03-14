#' predict
#'
#' @description Takes a fitted \code{NeuralGAM} object produced by
#' \code{fit_NeuralGAM()} and produces predictions given a new set of values.
#' @param ngam a fitted `NeuralGAM` object
#' @param x A data frame or list containing the values of covariates at which
#' predictions are required.
#' @param family A description of the link function used in the model:
#' \code{"gaussian"} or \code{"binomial"}
#' @param \ldots Other options.
#'
#' @return model predictions
#' @export
#'
#' @examples
predict.NeuralGAM <- function(ngam, x, family = "gaussian", ...) {

  f <- x*0

  for(i in 1:ncol(x)){
    f[, i] <- ngam$model[[i]] %>% predict(x[, i])
  }

  eta <- rowSums(f) + ngam$eta0
  y <- link(family, eta)

  res <- list(y, eta)
  return(res)
}
