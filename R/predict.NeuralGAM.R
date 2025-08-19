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
#'           \item \code{lwr}: Lower bound of prediction interval.
#'           \item \code{upr}: Upper bound of prediction interval.
#'           \item \code{fit}: Mean prediction (point estimate).
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
#'     \item{If \code{BUILD_PI = TRUE}}{\code{data.frame} with columns \code{lwr}, \code{upr}, \code{fit}.}
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

    if (is.data.frame(term_pred) && all(c("lwr","upr","fit") %in% colnames(term_pred))) {
      # Store PI info separately
      pi_components[[term]] <- term_pred
      f[[term]] <- term_pred$fit
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
      lwr <- rowSums(sapply(pi_components, function(pc) pc$lwr)) + ngam$eta0
      upr <- rowSums(sapply(pi_components, function(pc) pc$upr)) + ngam$eta0
      return(data.frame(lwr = lwr, upr = upr, fit = y))
    }

    return(y)
  }
}

get_model_predictions <- function(ngam, x, term, verbose) {
  ## -------------------------
  ## Parametric (linear) term
  ## -------------------------
  if (term %in% ngam$formula$p_terms) {
    model <- ngam$model$linear
    lm_data <- data.frame(x[ngam$formula$p_terms])
    colnames(lm_data) <- ngam$formula$p_terms

    if (isTRUE(ngam$build_pi)) {
      preds <- stats::predict(
        model,
        newdata = lm_data,
        type = "terms",
        terms = term,
        interval = "prediction",
        level = ngam$alpha
      )
      preds_df <- data.frame(
        lwr = preds$lwr,
        upr = preds$upr,
        fit = preds$fit,
        var_epistemic = NA_real_,
        var_aleatoric = NA_real_,
        var_total = NA_real_
      )
      colnames(preds_df) <- c("lwr", "upr", "fit")
      return(preds_df)
    } else {
      return(stats::predict(model, newdata = lm_data, type = "terms", terms = term))
    }
  }

  ## -------------------------
  ## Non-parametric (NN) term
  ## -------------------------
  if (term %in% ngam$formula$np_terms) {
    model <- ngam$model[[term]]
    X <- x[[term]]
    if (is.null(dim(X))) X <- matrix(X, ncol = 1L)

    # No PI: return point prediction on link scale
    if (!isTRUE(ngam$build_pi)) {
      preds <- model$predict(X, verbose = verbose)
      return(as.numeric(preds))
    }

    # Determine PI method
    pm <- tryCatch(model$pi_method, error = function(e) NULL)
    if (is.null(pm) || pm %in% c("", "none", NA)) {
      probe <- try(suppressWarnings(model$predict(X[1, , drop = FALSE], verbose = 0)), silent = TRUE)
      outdim <- if (inherits(probe, "try-error")) 1L else ncol(as.matrix(probe))
      pm <- if (outdim == 3L) "aleatoric" else "epistemic"
    }

    alpha <- ngam$alpha
    lower_q <- alpha / 2
    upper_q <- 1 - alpha / 2

    if (identical(pm, "aleatoric")) {
      # Expect 3-head output: [lwr, upr, fit] (link scale)
      preds <- as.matrix(model$predict(X, verbose = verbose))
      if (ncol(preds) >= 3L) {
        lwr <- preds[, 1]; upr <- preds[, 2]; mu <- preds[, 3]
        z_val <- stats::qnorm(1 - alpha / 2)
        width <- pmax(upr - lwr, 0)
        sd_ale <- if (z_val > 0) width / (2 * z_val) else rep(NA_real_, length(width))
        var_ale <- sd_ale^2
        return(data.frame(
          lwr = lwr,
          upr = upr,
          fit = mu,
          var_epistemic = NA_real_,
          var_aleatoric = var_ale,
          var_total = var_ale
        ))
      } else {
        # Fallback: no PI available from head
        return(as.numeric(preds[, 1]))
      }
    }

    if (identical(pm, "epistemic")) {
      # MC Dropout around a single-head mean
      passes <- tryCatch(as.integer(model$forward_passes), error = function(e) NA_integer_)
      if (is.na(passes) || passes <= 0L) passes <- 30L
      ya <- .mc_dropout_forward(model, X, passes = passes, output_dim = 1L)  # [passes, n, 1]
      ymat <- ya[, , 1]
      lwr <- matrixStats::colQuantiles(ymat, probs = lower_q)
      upr <- matrixStats::colQuantiles(ymat, probs = upper_q)
      mu  <- Matrix::colMeans(ymat)
      v   <- matrixStats::colVars(ymat)
      return(data.frame(
        lwr = as.numeric(lwr),
        upr = as.numeric(upr),
        fit = as.numeric(mu),
        var_epistemic = as.numeric(v),
        var_aleatoric = NA_real_,
        var_total = as.numeric(v)
      ))
    }

    if (identical(pm, "both")) {
      # MC Dropout around 3-head model: combine aleatoric + epistemic
      passes <- tryCatch(as.integer(model$forward_passes), error = function(e) NA_integer_)
      if (is.na(passes) || passes <= 0L) passes <- 30L
      ya <- .mc_dropout_forward(model, X, passes = passes, output_dim = 3L)  # [passes, n, 3]
      lwr_mat  <- ya[, , 1]
      upr_mat  <- ya[, , 2]
      mean_mat <- ya[, , 3]
      # This returns fit, lwr, upr, var_epistemic, var_aleatoric, var_total
      return(.combine_uncertainties(
        lwr_mat = lwr_mat,
        upr_mat = upr_mat,
        mean_mat = mean_mat,
        alpha = alpha
      ))
    }

    # Unknown method: best-effort fallback
    preds <- as.matrix(model$predict(X, verbose = verbose))
    if (ncol(preds) >= 3L) {
      lwr <- preds[, 1]; upr <- preds[, 2]; mu <- preds[, 3]
      z_val <- stats::qnorm(1 - alpha / 2)
      width <- pmax(upr - lwr, 0)
      sd_ale <- if (z_val > 0) width / (2 * z_val) else rep(NA_real_, length(width))
      var_ale <- sd_ale^2
      return(data.frame(
        lwr = lwr,
        upr = upr,
        fit = mu,
        var_epistemic = NA_real_,
        var_aleatoric = var_ale,
        var_total = var_ale
      ))
    } else {
      return(as.numeric(preds[, 1]))
    }
  }

  stop("Term '", term, "' not found in the model formula.")
}
