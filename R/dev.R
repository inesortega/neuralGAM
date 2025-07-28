#' Deviance of the model
#'
#' @description Computes the deviance of the model according to the distribution
#' family specified in the \code{"family"} parameter.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @param muhat current estimation of the response variable
#' @param y response variable
#' @param w weight assigned to each observation. Defaults to 1.
#' @param family A description of the link function used in the model:
#' \code{"gaussian"}, \code{"poisson"}, or \code{"binomial"}
#' @return the deviance of the model
#' @keywords internal
dev <- function(muhat, y, w, family) {

  if (missing(muhat)) {
    stop("Argument \"muhat\" is missing, with no default")
  }

  if (missing(y)) {
    stop("Argument \"y\" is missing, with no default")
  }

  if (missing(family)) {
    stop("Argument \"family\" is missing, with no default")
  }

  if (!family %in% c("gaussian", "binomial", "poisson")) {
    stop("Unsupported distribution family. Supported values are \"gaussian\", \"binomial\", and \"poisson\"")
  }

  if (family == "gaussian") {
    dev <- mean((y - muhat)**2)
  }

  if (family == "binomial") {
    muhat[muhat < 0.0001] <- 0.0001
    muhat[muhat > 0.9999] <- 0.9999

    entrop <- rep(0, length(y))
    ii <- (1 - y) * y > 0
    if (sum(ii, na.rm = TRUE) > 0) {
      entrop[ii] <- 2 * (y[ii] * log(y[ii])) +
        ((1 - y[ii]) * log(1 - y[ii]))
    } else {
      entrop <- 0
    }
    entadd <- 2 * (y * log(muhat)) + ((1 - y) * log(1 - muhat))
    dev <- sum(entrop - entadd, na.rm = TRUE)
  }

  if(family == "poisson"){
    # Clamp fitted values to avoid log(0)
    tempf <- pmax(muhat, 1e-4)
    # Initialize deviance
    dev <- 2 * w * (-y * log(tempf) - (y - muhat))

    # Add y * log(y) term where y > 0
    idx <- y > 0
    dev[idx] <- dev[idx] + 2 * w[idx] * y[idx] * log(y[idx])
    dev <- sum(dev)
  }
  return(dev)
}
