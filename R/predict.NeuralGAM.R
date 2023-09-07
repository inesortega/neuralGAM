#' Produces predictions from a fitted \code{neuralGAM} object
#' @description Takes a fitted \code{neuralGAM} object produced by
#' \code{neuralGAM()} and produces predictions given a new set of values for the model covariates.
#' @param object a fitted `neuralGAM` object
#' @param newdata A data frame or list containing the values of covariates at which
#' predictions are required. If not provided, the function returns the predictions
#' for the original training data.
#' @param type when \code{type="link"} (default), the linear
#' predictor is returned. When \code{type="terms"} each component of the linear
#' predictor is returned separately on each column of a \code{data.frame}. When
#' \code{type="response"} predictions on the scale of the response are returned.
#' @param terms If \code{type="terms"}, then only results for the terms named
#' in this list will be returned. If \code{NULL} then no terms are excluded (default).
#' @param verbose Verbosity mode (0 = silent, 1 = print messages). Defaults to 1.
#' @param \ldots Other options.
#' @return Predicted values according to \code{type} parameter.
#' @importFrom stats predict
#' @export
#' @examples \donttest{
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

predict.neuralGAM <-
  function(object,
           newdata = NULL,
           type = "link",
           terms = NULL,
           verbose = 1,
           ...) {
    # Check if object is of class "neuralGAM"
    if (!inherits(object, "neuralGAM")) {
      stop("The object argument must be a fitted neuralGAM object.")
    }

    # Check if object argument is missing or NULL
    if (missing(object) || is.null(object)) {
      stop("Please provide a fitted neuralGAM object as the object argument.")
    }

    ngam <- object

    # check that all parameters are OK
    if (missing(newdata)) {
      x <- ngam$x
    }
    else{
      x <- newdata
    }

    # Check if newdata columns match ngam$model columns
    if (type != "terms" &&
        !all(colnames(ngam$x) %in% colnames(x))) {
      stop("The newdata argument does not have the same columns as the fitted ngam model.")
    }

    # Check if type argument is valid
    valid_types <- c("link", "terms", "response")
    if (!type %in% valid_types) {
      stop("The value of the type argument is invalid. Valid options are {link, terms, response}.")
    }

    if (type == "terms" &&
        !is.null(terms) && !all(terms %in% colnames(x))) {
      stop(paste(
        "Invalid terms. Valid options are: ",
        paste(colnames(x), collapse = ",")
      ))
    }

    f <- data.frame(matrix(0, nrow = nrow(x), ncol = ncol(x)))
    colnames(f) <- colnames(x)

    for (i in 1:ncol(x)) {
      term <- colnames(x)[[i]]
      if (type == "terms" && !is.null(terms)) {
        # compute only certain terms
        if (term %in% terms) {
          f[[term]] <- get_model_predictions(ngam, x, term, verbose)
        }
        else{
          next
        }
      }
      else{
        f[[term]] <- get_model_predictions(ngam, x, term, verbose)
      }
    }

    if (type == "terms") {
      if (!is.null(terms)) {
        f <- f[, terms]
        colnames(f) <- terms
      }
      return(f)
    }

    eta <- rowSums(f) + ngam$eta0
    if (type == "link") {
      # Return the linear predictor
      return(eta)
    }

    if (type == "response") {
      y <- link(family = ngam$family, eta)
      return(y)
    }
  }

get_model_predictions <- function(ngam, x, term, verbose) {
  # Linear term
  if (term %in% ngam$formula$p_terms) {
    model <- ngam$model$linear
    lm_data <- data.frame(x[ngam$formula$p_terms])
    colnames(lm_data) <- ngam$formula$p_terms

    return(stats::predict(model,newdata = lm_data,
                          type = "terms",terms = term))
  }
  # Non-Parametric term
  if (term %in% ngam$formula$np_terms) {
    model <- ngam$model[[term]]
    return(model$predict(x[[term]], verbose = verbose))
  }
}
