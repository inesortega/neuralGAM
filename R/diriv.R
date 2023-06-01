#' Derivative of the link function
#'
#' @description Computes the derivative of the link function according to
#' the distribution family specified in the \code{"family"} parameter.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @param family A description of the link function used in the model:
#' \code{"gaussian"} or \code{"binomial"}
#' @param muhat fitted values
#' @return derivative of the link function for the fitted values
#' @keywords internal
diriv <- function(family, muhat) {

  if (missing(muhat)) {
    stop("Argument \"muhat\" is missing, with no default")
  }

  if (missing(family)) {
    stop("Argument \"family\" is missing, with no default")
  }

  if (family != "gaussian" & family != "binomial"){
    stop("Unsupported distribution family. Supported values are \"gaussian\" and \"binomial\"")
  }

  if (family == "gaussian") { # Identity
    out <- 1
  }
  if (family == "binomial") { # Derivative Logit
    prob <- muhat
    prob[prob >= 0.999] <- 0.999
    prob[prob <= 0.001] <- 0.001
    prob <- prob * (1.0 - prob)
    out <- 1.0 / prob
  }
  return(out)
}
