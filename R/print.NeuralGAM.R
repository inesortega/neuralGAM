#' @title Short \code{neuralGAM} summary
#' @description
#' Default print method for a \code{neuralGAM} object.
#'
#' @param x A \code{neuralGAM} object.
#' @param ... Additional arguments (currently unused).
#'
#' @return
#' Prints a brief summary of the fitted model including:
#' \describe{
#'   \item{Distribution family}{The distribution family used (\code{"gaussian"}, \code{"binomial"}, or \code{"poisson"}).}
#'   \item{Formula}{The model formula.}
#'   \item{Intercept value}{The fitted intercept (\eqn{\eta_0}).}
#'   \item{Mean Squared Error (MSE)}{The training MSE of the model.}
#'   \item{Training sample size}{The number of observations used to train the model.}
#' }
#'
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export
#'
#' @examples
#' \dontrun{
#'
#'
#' library(neuralGAM)
#' dat <- sim_neuralGAM_data()
#' train <- dat$train
#' test  <- dat$test
#'
#' ngam <- neuralGAM(
#'   y ~ s(x1) + x2 + s(x3),
#'   data = train,
#'   num_units = 1024,
#'   family = "gaussian",
#'   activation = "relu",
#'   learning_rate = 0.001,
#'   bf_threshold = 0.001,
#'   max_iter_backfitting = 10,
#'   max_iter_ls = 10,
#'   seed = seed
#' )
#' print(ngam)
#' }
print.neuralGAM <- function(x, ...) {
  if (inherits(x, "neuralGAM")) {
    # Print the class name
    # Print the object's contents
    ngam <- x

    build_pi <- tryCatch(isTRUE(ngam$build_pi), error = function(e) FALSE)
    alpha    <- tryCatch(ngam$alpha, error = function(e) NA_real_)

    cat("Class: neuralGAM \n")
    cat("Family             : ", ngam$family, "\n", sep = "")
    cat("Formula            : ", deparse(ngam$formula$formula), "\n", sep = "")
    cat("Observations       : ", NROW(ngam$y), "\n", sep = "")
    cat("Intercept (eta0)   : ", format(ngam$eta0, digits = 6), "\n", sep = "")
    cat(sprintf("Deviance explained : %.2f%%\n", attr(.deviance_explained.neuralGAM(ngam), "percent")))
    cat("Train MSE          : ", format(ngam$mse, digits = 6), "\n", sep = "")
    if (isTRUE(build_pi)) {
      cat("Pred. / Conf. Int. : ENABLED (alpha = ", alpha, ", method = ", ngam$uncertainty_method, ")\n", sep = "")
    } else {
      cat("Pred. / Conf. Int. : disabled\n")
    }

    invisible(x)
  } else{
    stop("Argument x must be a neuralGAM object.")
  }
}

