#' Link function
#'
#' @description Applies the link function according to the distribution family
#' specified in the \code{"family"} parameter.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @param family A description of the link function used in the model:
#' \code{"gaussian"} or \code{"binomial"}
#' @param muhat fitted values
#' @return the link function specified by the \code{"family"} distribution
#' for the given fitted values
#' @keywords internal

link <- function(family, muhat) {

  if (missing(muhat)) {
    stop("Argument \"muhat\" is missing, with no default")
  }

  if (missing(family)) {
    stop("Argument \"family\" is missing, with no default")
  }

  if (family != "gaussian" & family != "binomial") {
    stop("Unsupported distribution family. Supported values are \"gaussian\" and \"binomial\"")
  }


  if (family == "gaussian") {
    out <- muhat
  }
  if (family == "binomial") {
    muhat[muhat > 10] <- 10
    muhat[muhat < -10] <- -10
    out <- exp(muhat) / (1 + exp(muhat))
  }
  return(out)
}
