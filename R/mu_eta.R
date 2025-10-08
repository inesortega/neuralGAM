#' Derivative of the Inverse Link Function
#'
#' @description
#' Computes the derivative of the inverse link function \eqn{d\mu/d\eta}
#' for common distributtion families supported by \code{neuralGAM} (\code{"gaussian"}, \code{"binomial"},
#' \code{"poisson"}). This quantity is required when applying the delta method to obtain
#' standard errors on the response scale in \code{predict()}.
#'
#' @details
#' For a neuralGAM with linear predictor \eqn{\eta} and mean response \eqn{\mu}:
#' \deqn{\mu = g^{-1}(\eta),}
#' the derivative \eqn{d\mu/d\eta} depends on the family:
#' \itemize{
#'   \item Gaussian (identity link): \eqn{d\mu/d\eta = 1}.
#'   \item Binomial (logit link): \eqn{d\mu/d\eta = \mu (1-\mu)}.
#'   \item Poisson (log link): \eqn{d\mu/d\eta = \mu}.
#' }
#' Internally, values of \eqn{\eta} are clamped to avoid numerical
#' overflow/underflow in \code{exp()} and \eqn{\mu} is constrained away
#' from \eqn{0} and \eqn{1} for stability.
#'
#' @param family A character string specifying the distribution family:
#'   one of \code{"gaussian"}, \code{"binomial"}, or \code{"poisson"}.
#' @param eta Numeric vector of linear predictor values.
#'
#' @return A numeric vector of the same length as \code{eta}, containing
#'   the derivative \eqn{d\mu/d\eta}.
#'
#' @seealso \code{\link{inv_link}}, \code{\link{link}}
#' @author Ines Ortega-Fernandez, Marta Sestelo
#' @keywords internal
mu_eta <- function(family, eta) {
  if (missing(eta)) stop('Argument "eta" is missing, with no default')
  if (missing(family)) stop('Argument "family" is missing, with no default')
  if (!family %in% c("gaussian","binomial","poisson"))
    stop('Unsupported family')

  eta <- pmin(pmax(eta, -30), 30)

  if (family == "gaussian") {
    out <- rep(1, length(eta))
  } else if (family == "binomial") {
    mu <- 1/(1 + exp(-eta))
    mu <- pmin(pmax(mu, 1e-12), 1 - 1e-12)
    out <- mu*(1 - mu)
  } else {                                # poisson
    out <- exp(eta)   # equals mu
  }
  out
}
