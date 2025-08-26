#' @title Deviance of the model
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
dev <- function(muhat, y, family, w = NULL) {

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

  n <- length(y)

  if (is.null(w)) w <- rep(1, n)
  w <- as.numeric(w)

  if (family == "gaussian") {
    return(sum(w * (y - muhat)^2))
  }

  if (family == "binomial") {
    # y in [0,1] (0/1 or proportions). clamp for numerical stability
    eps <- 1e-12
    muhat  <- pmin(pmax(muhat, eps), 1 - eps)
    y   <- pmin(pmax(y,  eps), 1 - eps)
    val <- 2 * ( y * log(y/muhat) + (1 - y) * log((1 - y)/(1 - muhat)) )
    val[!is.finite(val)] <- 0
    return(sum(w * val))
  }

  if(family == "poisson"){
    eps <- 1e-12
    muhat  <- pmax(muhat, eps)
    y   <- pmax(y, 0)
    # convention: y*log(y/mu) is 0 when y==0
    part <- numeric(length(y))
    nz <- y > 0
    part[nz] <- y[nz] * log(y[nz] / muhat[nz])
    val <- 2 * ( part - (y - muhat) )
    return(sum(w * val))
  }
  return(dev)
}

# % deviance explained for a neuralGAM object (training set by default)
.deviance_explained.neuralGAM <- function(object){
  stopifnot(inherits(object, "neuralGAM"))
  y  <- object$y
  mu <- object$muhat
  fam <- object$family
  w <- object$w_train

  # null mean on response scale
  mu0 <- rep(mean(y), length(y))

  dev_model <- dev(muhat = mu, y = y, family = fam, w = w)
  dev_null  <- dev(muhat = mu0, y = y, family = fam, w = w)

  # guard against degenerate null deviance
  if (dev_null <= .Machine$double.eps) return(NA_real_)

  out <- 1 - dev_model / dev_null
  structure(out,
            percent = 100 * out,
            dev_model = dev_model,
            dev_null = dev_null,
            family = fam,
            class = "neuralGAM_devexp")
}

# pretty print
.print.neuralGAM_deve <- function(x, ...){
  cat(sprintf("Deviance explained: %.2f%%\n", attr(x, "percent")))
  invisible(x)
}
