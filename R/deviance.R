#' Deviance of the model
#'
#' @description Computes the deviance of the model according to the distribution
#' family specified in the \code{"family"} parameter (\code{"gaussian"})
#' or (\code{"binomial"})
#'
#' @param fit current estimation of the response variable
#' @param y response variable
#' @param family A description of the link function used in the model: (\code{"gaussian"}) or (\code{"binomial"})
#'
#' @return the deviance of the model
#' @export
#'
#' @examples
#' dev <- deviance(muhat, y_train, "gaussian")
#' dev <- deviance(muhat, y_train, "binomial")
deviance <- function(fit, y, family){

  if (missing(fit)){
    stop("Argument \"fit\" is missing, with no default")
  }

  if (missing(y)){
    stop("Argument \"y\" is missing, with no default")
  }

  if (missing(family)){
    stop("Argument \"family\" is missing, with no default")
  }

  if (family != "gaussian" | family != "binomial"){
    stop("Unsupported distribution family. Supported values are \"gaussian\" and \"binomial\"")
  }

  if(family == "gaussian") {
    dev <- mean((y - fit)**2)
  }

  if(family == "binomial"){
    fit[fit < 0.0001] <- 0.0001
    fit[fit > 0.9999] <- 0.9999

    entrop = rep(0, length(y))
    ii = (1 - y) * y > 0
    if (sum(ii, na.rm = TRUE) > 0) {
      entrop[ii] = 2 * (y[ii] * log(y[ii])) +
        ((1 - y[ii]) * log(1 - y[ii]))
    } else {
      entrop = 0
    }
    entadd <- 2 * (y * log(fit)) + ( (1-y) * log(1-fit))
    dev <- sum(entrop - entadd, na.rm = TRUE)
  }
  return(dev)
}
