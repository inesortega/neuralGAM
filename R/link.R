#' Link function
#'
#' @description Applies the link function according to the distribution family
#' specified in the \code{"family"} parameter.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @param family A description of the link function used in the model:
#' \code{"gaussian"} or \code{"binomial"}
#' @param muhat fitted values
#' @export
#' @return the link function specified by the \code{"family"} distribution
#' for the given fitted values

link <- function(family, muhat) {
  # Applies the link function

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
