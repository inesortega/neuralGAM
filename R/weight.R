#' Weights
#'
#' @description Computes the weights for the Local Scoring Algorithm.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @param w weights
#' @param muhat fitted values
#' @param family A description of the link function used in the model:
#' \code{"gaussian"} or \code{"binomial"}
#' @return computed weights for the Local Scoring algorithm
#' according to the \code{"family"} distribution
#' @keywords internal
weight <- function(w, muhat, family) {

  if (missing(muhat)) {
    stop("Argument \"muhat\" is missing, with no default")
  }

  if (missing(w)) {
    stop("Argument \"w\" is missing, with no default")
  }

  if (missing(family)) {
    stop("Argument \"family\" is missing, with no default")
  }

  if (family != "gaussian" & family != "binomial") {
    stop("Unsupported distribution family. Supported values are \"gaussian\" and \"binomial\"")
  }


  if (family == "gaussian") { # Identity
    wei <- w
  }
  if (family == "binomial") { # Derivative Logit
    muhat[muhat <= 0.001] <- 0.001
    muhat[muhat >= 0.999] <- 0.999
    temp <- diriv(family, muhat)
    aux <- muhat * (1 - muhat) * (temp**2)
    aux[aux <= 0.001] <- 0.001
    wei <- w / aux
  }
  return(wei)
}
