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
    eta_clamped <- pmin(pmax(muhat, -300), 300)
    exp_eta <- exp(eta_clamped)
    out <- exp_eta / (1 + exp_eta)
  }

  if (family == "poisson") {
    out <- ifelse(muhat <= 88, exp(muhat), exp(88))
  }

  return(out)
}

inv_link_mat <- function(fam, M) { matrix(inv_link(fam, as.numeric(M)), nrow = nrow(M)) }
