#' @title Predictions from a fitted \code{neuralGAM} object
#'
#' @description
#' Generates predictions from a fitted \code{neuralGAM} model for new or training data.
#' Supports both standard point predictions and prediction intervals if the model was trained
#' with \code{BUILD_PI = TRUE}.
#'
#' @param object A fitted \code{neuralGAM} object.
#' @param newdata A data frame or list containing covariate values at which predictions
#' are required. If not provided, predictions are returned for the training data.
#' @param type Type of prediction:
#'   \describe{
#'     \item{"link"}{Returns the linear predictor (default).}
#'     \item{"terms"}{Returns each model component contribution as separate columns in a \code{data.frame}.}
#'     \item{"response"}{Returns predictions on the response scale.
#'       If the model was trained with \code{BUILD_PI = TRUE}, this will return a \code{data.frame} with columns:
#'         \itemize{
#'           \item \code{y_L}: Lower bound of prediction interval.
#'           \item \code{y_U}: Upper bound of prediction interval.
#'           \item \code{y_hat}: Mean prediction (point estimate).
#'         }
#'       Otherwise, a numeric vector of point predictions is returned.
#'     }
#'   }
#' @param terms If \code{type = "terms"}, a character vector of term names to return.
#'   If \code{NULL} (default), all terms are returned.
#' @param verbose Verbosity mode (0 = silent, 1 = print messages). Defaults to 1.
#' @param ... Additional arguments passed to underlying prediction methods.
#'
#' @return
#' - For \code{type = "link"}: numeric vector of linear predictor values.
#' - For \code{type = "terms"}: \code{data.frame} of term contributions.
#' - For \code{type = "response"}:
#'   \describe{
#'     \item{If \code{BUILD_PI = FALSE}}{Numeric vector of predicted response values.}
#'     \item{If \code{BUILD_PI = TRUE}}{\code{data.frame} with columns \code{y_L}, \code{y_U}, \code{y_hat}.}
#'   }
#'
#' @details
#' When prediction intervals are available (\code{BUILD_PI = TRUE}), the lower and upper bounds
#' are returned along with the mean prediction. For plotting, these can be passed to
#' \code{autoplot.neuralGAM()}, which will automatically add ribbons for intervals.
#' @importFrom stats predict
#' @export
#' @examples \dontrun{
#'
#' n <- 24500
#'
#' seed <- 42
#' set.seed(seed)
#'
#' x1 <- runif(n, -2.5, 2.5)
#' x2 <- runif(n, -2.5, 2.5)
#' x3 <- runif(n, -2.5, 2.5)
#'
#' f1 <-x1**2
#' f2 <- 2*x2
#' f3 <- sin(x3)
#' f1 <- f1 - mean(f1)
#' f2 <- f2 - mean(f2)
#' f3 <- f3 - mean(f3)
#'
#' eta0 <- 2 + f1 + f2 + f3
#' epsilon <- rnorm(n, 0.25)
#' y <- eta0 + epsilon
#' train <- data.frame(x1, x2, x3, y)
#'
#' library(neuralGAM)
#' ngam <- neuralGAM(y ~ s(x1) + x2 + s(x3), data = train,
#'                  num_units = 1024, family = "gaussian",
#'                  activation = "relu",
#'                  learning_rate = 0.001, bf_threshold = 0.001,
#'                  max_iter_backfitting = 10, max_iter_ls = 10,
#'                  seed = seed
#'                  )
#' n <- 5000
#' x1 <- runif(n, -2.5, 2.5)
#' x2 <- runif(n, -2.5, 2.5)
#' x3 <- runif(n, -2.5, 2.5)
#' test <- data.frame(x1, x2, x3)
#'
#' # Obtain linear predictor
#' eta <- predict(ngam, test, type = "link")
#'
#' # Obtain predicted response
#' yhat <- predict(ngam, test, type = "response")
#'
#' # Obtain each component of the linear predictor
#' terms <- predict(ngam, test, type = "terms")
#'
#' # Obtain only certain terms:
#' terms <- predict(ngam, test, type = "terms", terms = c("x1", "x2"))
#' }

predict.neuralGAM <- function(object,
                              newdata = NULL,
                              type = "link",
                              terms = NULL,
                              verbose = 1,
                              ...) {

  if (!inherits(object, "neuralGAM")) stop("The object must be a fitted neuralGAM.")
  ngam <- object

  if (missing(newdata)) {
    x <- ngam$x
  } else {
    x <- newdata
  }

  valid_types <- c("link", "terms", "response")
  if (!type %in% valid_types) stop("Invalid type. Choose from link, terms, response.")

  f <- data.frame(matrix(0, nrow = nrow(x), ncol = ncol(x)))
  colnames(f) <- colnames(x)

  pi_components <- list()

  for (i in seq_along(colnames(x))) {
    term <- colnames(x)[i]
    term_pred <- get_model_predictions(ngam, x, term, verbose)

    if (is.data.frame(term_pred) && all(c("y_L","y_U","y_hat") %in% colnames(term_pred))) {
      # Store PI info separately
      pi_components[[term]] <- term_pred
      f[[term]] <- term_pred$y_hat
    } else {
      f[[term]] <- term_pred
    }
  }

  if (type == "terms") {
    if (!is.null(terms)) f <- f[, terms, drop = FALSE]
    return(f)
  }

  eta <- rowSums(f) + ngam$eta0
  if (type == "link"){
    return(eta)
  }

  if (type == "response") {
    y <- link(family = ngam$family, eta)

    # Attach PI if available
    if (length(pi_components) > 0) {
      y_L <- rowSums(sapply(pi_components, function(pc) pc$y_L)) + ngam$eta0
      y_U <- rowSums(sapply(pi_components, function(pc) pc$y_U)) + ngam$eta0
      return(data.frame(y_L = y_L, y_U = y_U, y_hat = y))
    }

    return(y)
  }
}

get_model_predictions <- function(ngam, x, term, verbose) {
  # Linear term
  if (term %in% ngam$formula$p_terms) {
    model <- ngam$model$linear
    lm_data <- data.frame(x[ngam$formula$p_terms])
    colnames(lm_data) <- ngam$formula$p_terms

    return(stats::predict(model, newdata = lm_data,
                          type = "terms", terms = term))
  }

  # Non-Parametric term
  if (term %in% ngam$formula$np_terms) {
    model <- ngam$model[[term]]
    preds <- model$predict(x[[term]], verbose = verbose)

    # Handle PI vs no PI automatically
    if (is.matrix(preds) && ncol(preds) == 3) {
      # Return as data.frame with columns y_L, y_U, y_hat
      preds_df <- data.frame(
        y_L = preds[, 1],
        y_U = preds[, 2],
        y_hat = preds[, 3]
      )
      return(preds_df)
    } else {
      # Return mean prediction only
      return(as.numeric(preds))
    }
  }
}
