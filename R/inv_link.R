#' Inverse of the link functions
#'
#' @description Computes the inverse of the link function according to the
#' distribution family specified in the \code{"family"} parameter.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @param family A description of the link function used in the model:
#' \code{"gaussian"} or \code{"binomial"}
#' @param muhat fitted values
#' @return the inverse link function specified by the \code{"family"}
#' distribution for the given fitted values
#' @keywords internal
inv_link <- function(family, muhat) {

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
    d <- 1 - muhat
    d[d <= 0.001] <- 0.001
    d[d >= 0.999] <- 0.999
    out <- log(muhat / d)
  }

  return(out)
}
