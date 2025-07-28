#' Inverse of the link functions
#'
#' @description Computes the inverse of the link function according to the
#' distribution family specified in the \code{"family"} parameter.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @param family A description of the link function used in the model:
#' \code{"gaussian"}, \code{"poisson"} or \code{"binomial"}
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

  if (!family %in% c("gaussian", "binomial", "poisson")) {
    stop("Unsupported distribution family. Supported values are \"gaussian\", \"binomial\", and \"poisson\"")
  }

  if (family == "gaussian") {
    out <- muhat
  }
  if (family == "binomial") {
    d <- pmax(1 - muhat, 1e-4)
    ratio <- muhat / d
    ratio <- pmin(pmax(ratio, 1e-4), 9999.0)
    out <- log(ratio)
  }
  if (family == "poisson") {
    muhat <- pmax(muhat, 1e-4)
    out <- log(muhat)
  }

  return(out)
}
