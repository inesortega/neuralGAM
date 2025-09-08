#' Produces predictions from a fitted \code{neuralGAM} object
#'
#' @description
#' Generate predictions from a fitted \code{neuralGAM} model. Supported types:
#'
#' \itemize{
#'   \item \code{type = "link"} (default): linear predictor on the link scale.
#'   \item \code{type = "response"}: predictions on the response scale.
#'   \item \code{type = "terms"}: per-term contributions to the linear predictor (no intercept).
#' }
#'
#' \strong{Standard errors and intervals}
#'
#' \itemize{
#'   \item If \code{se.fit = TRUE}, standard errors (SE) of the \emph{fitted mean} are returned
#'         (epistemic only; mgcv-style via MC Dropout).
#'   \item For \code{type = "response"}, SEs are mapped by the delta method:
#'         \eqn{se_\mu = |d\mu/d\eta| \cdot se_\eta}.
#'   \item Confidence intervals (CI) always reflect epistemic uncertainty about the fitted mean
#'         (as in mgcv).
#'   \item Prediction intervals (PI) are available only on the \emph{response scale}, and only if
#'         the model was trained with \code{uncertainty_method \%in\% c("aleatoric", "both")}. On the per-term scale
#'         quantile bands are available for diagnosis purposes only.
#' }
#'
#' \strong{Important implementation details}
#'
#' \itemize{
#'   \item \emph{Epistemic SEs (CIs)} are obtained via Monte Carlo Dropout: multiple stochastic
#'         forward passes provide an across-pass variance. For full-model CIs, uncertainty is
#'         aggregated jointly by summing per-pass smooth contributions and taking the
#'         across-pass variance (implicitly accounting for cross-term covariance). For
#'         \code{type="terms"}, SEs are returned per smooth.
#'   \item \emph{Prediction intervals (PIs)} reflect aleatoric uncertainty (and, for
#'         \code{uncertainty_method="both"}, combined effects). PIs are provided \emph{only} on the
#'         response scale.
#'   \item For \code{type = "terms"}, only \emph{confidence intervals for the smooth effect}
#'         are meaningful (from SEs). \strong{Prediction intervals are not defined for terms}
#'         because partial effects are fitted functions, not noisy observations.
#'   \item For \code{type = "link"}, only confidence intervals are provided;
#'         prediction intervals are not defined on the link scale.
#' }
#'
#' @param object A fitted \code{neuralGAM} object.
#' @param newdata Optional \code{data.frame}/list of covariates at which to predict.
#'   If omitted, the training data cached in the object are used.
#' @param type One of \code{c("link","response","terms")}. Default \code{"link"}.
#' @param terms If \code{type = "terms"}, character vector of term names to include.
#'   If \code{NULL}, all terms are returned. Intercept is not included (as in \pkg{mgcv}).
#' @param se.fit Logical; if \code{TRUE}, return SEs of the fitted mean (mgcv-style). Default \code{FALSE}.
#'   For \code{type="terms"}, returns a matrix of per-term SEs (epistemic) when available.
#' @param interval One of \code{c("none","confidence","prediction","both")}. Default \code{"none"}.
#'   Ignored when \code{type = "terms"} (prediction intervals are not defined for terms).
#' @param level Coverage level for intervals (e.g., \code{0.95}). Default \code{0.95}.
#' @param forward_passes Integer; number of MC-dropout forward passes when computing
#'   epistemic uncertainty (used when \code{uncertainty_method \%in\% c("epistemic","both")}).
#' @param diagnostic_bands Logical (default FALSE). When \code{type="terms"}, return
#'   per-term \emph{aleatoric residual quantile bands} (from quantile heads) for
#'   diagnostic plotting. These are bands for partial residuals \eqn{R_j} and should
#'   \emph{not} be interpreted as prediction intervals for \eqn{Y}.
#' @param verbose Integer (0/1). Default \code{1}.
#' @param \ldots Other options (passed on to internal predictors).
#'
#' @return
#' \itemize{
#'   \item If \code{interval == "none"}:
#'     \itemize{
#'       \item \code{se.fit = FALSE}: a numeric vector (link/response) or a numeric matrix (terms).
#'       \item \code{se.fit = TRUE}: a list with \code{$fit} and \code{$se.fit} for link/response;
#'             for \code{type="terms"}, \code{list(fit = terms_matrix, se.fit = se_terms_matrix)}.
#'     }
#'   \item If \code{interval != "none"}:
#'     \itemize{
#'       \item \code{type="link"}: a data frame with CI columns \code{lwr}, \code{upr}
#'             (PIs are not provided on the link scale).
#'       \item \code{type="response"}: a data frame with CI, PI, or both (when available).
#'       \item \code{type="terms"}: same as \code{interval = "none"} (intervals not returned).
#'     }
#' }
#'
#' @importFrom stats predict qnorm
#' @export
#' @author Ines Ortega-Fernandez, Marta Sestelo
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 2000
#' x1 <- runif(n, -2.5, 2.5)
#' x2 <- runif(n, -2.5, 2.5)
#' x3 <- runif(n, -2.5, 2.5)
#' f1 <- x1^2; f2 <- 2 * x2; f3 <- sin(x3)
#' y  <- 2 + f1 + f2 + f3 + rnorm(n, 0.25)
#' train <- data.frame(x1, x2, x3, y)
#'
#' library(neuralGAM)
#'
#' # 1) Model without PIs
#' ngam0 <- neuralGAM(
#'   y ~ s(x1) + x2 + s(x3),
#'   data = train, family = "gaussian", num_units = 128
#' )
#' eta <- predict(ngam0, type = "link")
#' mu  <- predict(ngam0, type = "response")
#' trm <- predict(ngam0, type = "terms")
#' pr_link <- predict(ngam0, type = "link", se.fit = TRUE)
#' pr_resp <- predict(ngam0, type = "response", se.fit = TRUE)
#' pr_terms <- predict(ngam0, type = "terms", se.fit = TRUE)
#'
#' newx <- data.frame(x1 = seq(-2.5, 2.5, length.out = 200), x2 = 0, x3 = 0)
#' mu_new <- predict(ngam0, newdata = newx, type = "response")
#'
#' # 2) Model with PIs (aleatoric)
#' ngam_ale <- neuralGAM(
#'   y ~ s(x1) + x2 + s(x3),
#'   data = train, family = "gaussian", num_units = 128,
#'   uncertainty_method = "aleatoric", alpha = 0.05
#' )
#' ci_df  <- predict(ngam_ale, type = "response", interval = "confidence", level = 0.95)
#' pi_df  <- predict(ngam_ale, type = "response", interval = "prediction", level = 0.95)
#' both_df <- predict(ngam_ale, type = "response", interval = "both", level = 0.95)
#' trm_ci <- predict(ngam_ale, type = "terms", se.fit = TRUE)
#' trm_x1x2 <- predict(ngam_ale, type = "terms", terms = c("x1", "x2"))
#'
#' newx2 <- data.frame(x1 = seq(-2.5, 2.5, length.out = 300), x2 = 0.5, x3 = 0)
#' both_new <- predict(ngam_ale, newdata = newx2, type = "response", interval = "both", level = 0.95)
#'
#' # 3) Model with PIs (both: aleatoric + epistemic)
#' ngam_both <- neuralGAM(
#'   y ~ s(x1) + x2 + s(x3),
#'   data = train, family = "gaussian", num_units = 128,
#'   uncertainty_method = "both", alpha = 0.05, forward_passes = 50)
#' res_both   <- predict(ngam_both, type = "response", interval = "both")
#' link_ci    <- predict(ngam_both, type = "link", interval = "confidence")
#' }
predict.neuralGAM <- function(object,
                              newdata = NULL,
                              type = c("link","response","terms"),
                              terms = NULL,
                              se.fit = FALSE,
                              interval = c("none","confidence","prediction","both"),
                              level = 0.95, forward_passes = 30,
                              diagnostic_bands = FALSE,
                              verbose = 1, ...) {
  stopifnot(inherits(object, "neuralGAM"))
  type     <- match.arg(type)
  interval <- match.arg(interval)
  ngam     <- object

  # ---- data ----
  if (is.null(newdata)){
    x <- ngam$x
    use_cache <- TRUE
  }
  else{
    x <- as.data.frame(newdata)
    use_cache <- FALSE
  }
  needed <- colnames(ngam$x)
  if (!all(needed %in% colnames(x))) {
    miss <- setdiff(needed, colnames(x))
    stop("newdata is missing required columns: ", paste(miss, collapse = ", "))
  }
  x <- x[, needed, drop = FALSE]

  p_terms  <- ngam$formula$p_terms %||% character(0L)
  np_terms <- ngam$formula$np_terms %||% character(0L)
  all_terms <- c(p_terms, np_terms)

  if (!is.null(terms)) {
    if (!all(terms %in% all_terms)) {
      bad <- setdiff(terms, all_terms)
      stop("Unknown term(s) in `terms`: ", paste(bad, collapse = ", "))
    }
    sel_terms <- terms
  } else {
    sel_terms <- all_terms
  }

  # ---- per-term link-scale ----

  # containers (n x p)
  n <- nrow(x); p <- length(all_terms)
  term_fit <- matrix(0, n, p, dimnames = list(NULL, all_terms))
  var_epi  <- matrix(NA_real_, n, p, dimnames = list(NULL, all_terms))
  var_ale  <- matrix(NA_real_, n, p, dimnames = list(NULL, all_terms))

  lwr_abs  <- matrix(NA_real_, n, p, dimnames = list(NULL, all_terms))
  upr_abs  <- matrix(NA_real_, n, p, dimnames = list(NULL, all_terms))

  if (use_cache) {
    term_fit[,] <- as.matrix(ngam$partial[, all_terms, drop = FALSE])

    if (!is.null(ngam$var_epistemic)) {
      var_epi[,] <- as.matrix(ngam$var_epistemic[, all_terms, drop = FALSE])
    }
    if (!is.null(ngam$var_aleatoric)) {
      var_ale[,] <- as.matrix(ngam$var_aleatoric[, all_terms, drop = FALSE])
    }
    if (isTRUE(ngam$build_pi)) {
      lwr_abs[,] <- as.matrix(ngam$lwr[, all_terms, drop = FALSE])
      upr_abs[,] <- as.matrix(ngam$upr[, all_terms, drop = FALSE])
    }

  } else {
    # re-predict each term with the fitted Keras models
    if (length(p_terms)) {
      lm_data <- x[, p_terms, drop = FALSE]
      colnames(lm_data) <- p_terms
      linmod <- ngam$model$linear
      for (tm in p_terms) {
        pr <- stats::predict(linmod, newdata = lm_data, type = "terms", terms = tm, se.fit = TRUE)
        term_fit[, tm] <- as.numeric(pr$fit)
        var_epi[, tm]  <- (as.numeric(pr$se.fit))^2
        # Do NOT fill lwr_mat/upr_mat for parametric terms (leave NA)
      }
    }
    # nonparametric terms via NN
    if (length(np_terms)) {
      for (tm in np_terms) {
        need_se <- isTRUE(se.fit) ||
          interval %in% c("confidence","both") ||
          (type == "link" && interval %in% c("prediction","both"))
        need_pi <- isTRUE(ngam$build_pi)

        pt <- .ngam_predict_term(
          ngam, x[[tm]], term_name = tm,
          want_pi = need_pi,     # deterministic lwr/upr for aleatoric models
          want_se = need_se,     # MC for epistemic variance
          level = level, forward_passes = forward_passes,
          verbose = verbose
        )
        term_fit[, tm] <- pt$fit
        var_epi[, tm]  <- pt$var_epistemic
        if (need_pi) {
          var_ale[, tm] <- pt$var_aleatoric
          lwr_abs[, tm] <- pt$lwr_abs
          upr_abs[, tm] <- pt$upr_abs
        }
      }
    }
  }

  # --- Fill missing epistemic variance if CI/SE requested (works for cache & newdata)
  need_se_req <- isTRUE(se.fit) ||
    interval %in% c("confidence","both") ||
    (type == "link" && interval %in% c("prediction","both"))

  if (need_se_req && any(!is.finite(var_epi))) {

    # 1) Parametric terms: mgcv-style SEs from linear submodel
    if (length(p_terms)) {
      linmod <- ngam$model$linear
      if (!is.null(linmod)) {
        lm_data <- x[, p_terms, drop = FALSE]
        colnames(lm_data) <- p_terms
        for (tm in p_terms) {
          miss <- !is.finite(var_epi[, tm])
          if (any(miss)) {
            pr <- stats::predict(linmod, newdata = lm_data,
                                 type = "terms", terms = tm, se.fit = TRUE)
            ve <- (as.numeric(pr$se.fit))^2
            var_epi[miss, tm] <- ve[miss]
          }
        }
      }
    }

    # 2) Nonparametric terms: MC-dropout on mean head for epistemic variance
    if (length(np_terms)) {
      for (tm in np_terms) {
        miss <- !is.finite(var_epi[, tm])
        if (any(miss)) {
          pt <- .ngam_predict_term(
            ngam, x[[tm]], term_name = tm,
            want_pi = FALSE,        # only epistemic SEs
            want_se = TRUE,         # MC-dropout
            level = level,
            forward_passes = forward_passes,
            verbose = verbose
          )
          var_epi[miss, tm] <- pt$var_epistemic[miss]
        }
      }
    }
  }


  # ---- assemble on link scale ----
  eta0 <- ngam$eta0 %||% 0
  eta  <- eta0 + rowSums(term_fit, na.rm = FALSE)

  # epistemic SE on link

  # prefer joint MC for epistemic SE when using dropout-based uncertainty -> makes the CI more faithful when smooths are correlated
  # under the data distribution. Recompute MC if needed.
  use_joint_mc <- need_se_req &&
    length(ngam$formula$np_terms %||% character(0L)) > 0L &&
    any(!is.finite(rowSums(var_epi, na.rm = TRUE)))

  if (isTRUE(use_joint_mc)) {
    se_eta <- .joint_se_eta_mcdropout(
      ngam, x,
      forward_passes = forward_passes,   # pass through user arg
      verbose = 0
    )
  } else {
    # fallback: sum of per-term variances (no cross-term covariance)
    row_var_epi <- .row_sum_var(var_epi)
    se_eta      <- sqrt(pmax(row_var_epi, 0))
  }

  # ---- type="terms" ----
  if (type == "terms") {
    fit_terms <- term_fit[, sel_terms, drop = FALSE]

    # default (legacy) behavior: no bands requested
    if (!diagnostic_bands && (!se.fit || interval != "none")) {
      return(fit_terms)
    }

    # when SE requested, compute epistemic SEs
    se_terms <- if (isTRUE(se.fit)) sqrt(pmax(var_epi[, sel_terms, drop = FALSE], 0)) else NULL

    # expose aleatoric residual bands if asked (diagnostic only)
    lwr_ale_terms <- upr_ale_terms <- NULL
    if (isTRUE(diagnostic_bands)) {
      # These are quantiles for partial residuals R_j, not PIs for Y
      # if diagnostic_bands are requested, caller gets fit/se.fit/lwr_ale/upr_ale matrices
      lwr_ale_terms <- lwr_abs[, sel_terms, drop = FALSE] - term_fit[, sel_terms, drop = FALSE]
      upr_ale_terms <- upr_abs[, sel_terms, drop = FALSE] - term_fit[, sel_terms, drop = FALSE]
    }

    # Return a list so autoplot (and users) can access what's available
    out <- list(fit = fit_terms)
    if (!is.null(se_terms))        out$se.fit   <- se_terms
    if (!is.null(lwr_ale_terms))   out$lwr_ale  <- lwr_ale_terms
    if (!is.null(upr_ale_terms))   out$upr_ale  <- upr_ale_terms
    return(out)
  }
  # ---- type="link" ----
  if (type == "link") {
    if (interval == "none") {
      if (!se.fit) return(eta)
      return(list(fit = eta, se.fit = se_eta))
    } else {
      # CI on link
      z <- stats::qnorm(1 - (1 - level)/2)
      lwr_ci <- eta - z * se_eta
      upr_ci <- eta + z * se_eta

      if (!any(is.finite(lwr_ci)) || !any(is.finite(upr_ci))) {
        warning("Confidence intervals not available on link scale (missing epistemic variance).")
      }
      if (interval == "confidence") {
        return(data.frame(fit = eta, lwr = lwr_ci, upr = upr_ci))
      } else {
        warning("Prediction intervals are defined on the response scale; returning CI on link.")
        return(data.frame(fit = eta, lwr = lwr_ci, upr = upr_ci))
      }
    }
  }

  # ---- type="response" ----
  mu <- inv_link(ngam$family, eta)

  if (interval == "none") {
    if (!se.fit) return(mu)
    gprime <- abs(mu_eta(ngam$family, eta))
    se_mu  <- gprime * se_eta
    return(list(fit = mu, se.fit = se_mu))
  }

  # CI for mean on response
  z <- stats::qnorm(1 - (1 - level)/2)
  gprime <- abs(mu_eta(ngam$family, eta))
  se_mu  <- gprime * se_eta
  lwr_ci <- mu - z * se_mu
  upr_ci <- mu + z * se_mu
  ci_available <- all(!is.na(se_mu))

  # PI for new observation on response
  pi_available <- FALSE
  lwr_pi <- upr_pi <- rep(NA_real_, length(mu))

  pi_available <- FALSE
  lwr_pi <- upr_pi <- rep(NA_real_, length(mu))

  if (isTRUE(ngam$build_pi)) {
    if (identical(tolower(ngam$uncertainty_method), "both")) {
      # Joint combination across terms and passes
      joint <- .joint_pi_both_variance(ngam, x, level = level, forward_passes = forward_passes)
      if (!is.null(joint)) {
        lwr_pi <- inv_link(ngam$family, joint$lwr)
        upr_pi <- inv_link(ngam$family, joint$upr)
        pi_available <- any(is.finite(lwr_pi) | is.finite(upr_pi))
      }
    } else {
      # Aleatoric only: use absolute per-term bounds (still an approximation)
      if (any(is.finite(lwr_abs)) && any(is.finite(upr_abs))) {
        eta_lwr <- eta0 + rowSums(lwr_abs, na.rm = FALSE)
        eta_upr <- eta0 + rowSums(upr_abs, na.rm = FALSE)
        lwr_pi  <- inv_link(ngam$family, eta_lwr)
        upr_pi  <- inv_link(ngam$family, eta_upr)
        pi_available <- any(is.finite(lwr_pi) | is.finite(upr_pi))
      } else if (any(is.finite(var_ale))) {
        # Fallback: Normal approximation using summed variances
        row_var_epi   <- se_eta^2
        row_var_ale   <- .row_sum_var(var_ale)
        row_var_total <- row_var_epi + row_var_ale
        sd_tot <- sqrt(pmax(row_var_total, 0))
        z      <- stats::qnorm(1 - (1 - level)/2)
        eta_lwr <- eta - z * sd_tot
        eta_upr <- eta + z * sd_tot
        lwr_pi  <- inv_link(ngam$family, eta_lwr)
        upr_pi  <- inv_link(ngam$family, eta_upr)
        pi_available <- any(is.finite(lwr_pi) | is.finite(upr_pi))
      }
    }
  }

  if (interval == "confidence") {
    if (!ci_available) warning("Confidence intervals not available (missing epistemic variance).")
    return(data.frame(fit = mu, lwr = lwr_ci, upr = upr_ci))
  } else if (interval == "prediction") {
    if (!pi_available) warning("Prediction intervals not available (missing aleatoric variance).")
    return(data.frame(fit = mu, lwr = lwr_pi, upr = upr_pi))
  } else { # both
    if (!ci_available) warning("CI not available (missing epistemic variance).")
    if (!pi_available) warning("PI not available.")
    return(data.frame(fit = mu,
                      lwr_ci = lwr_ci, upr_ci = upr_ci,
                      lwr_pi = lwr_pi, upr_pi = upr_pi))
  }
}

.ngam_predict_term <- function(ngam, xvec, term_name,
                               want_pi = FALSE,
                               want_se = FALSE,
                               verbose = 1, level = 0.95,
                               forward_passes = 30) {

  mdl <- ngam$model[[term_name]]
  X <- xvec; if (is.null(dim(X))) X <- matrix(X, ncol = 1L)

  alpha <- 1 - level
  pm <- ngam$uncertainty_method

  y_det <- try(as.matrix(mdl$predict(X, verbose = 0)), silent = TRUE)
  if (inherits(y_det, "try-error") || is.null(dim(y_det))) {
    fit <- as.numeric(mdl$predict(X, verbose = 0))
    return(list(fit = fit,
                var_epistemic = rep(NA_real_, length(fit)),
                var_aleatoric = rep(NA_real_, length(fit)),
                # absolute bounds are NA in this fallback
                lwr_abs = rep(NA_real_, length(fit)),
                upr_abs = rep(NA_real_, length(fit))))
  }

  nout <- ncol(y_det)
  mean_col <- if (nout >= 3L) 3L else 1L
  mu_det <- as.numeric(y_det[, mean_col])

  preds <- .compute_uncertainty(mdl, X, pm, alpha, forward_passes)

  # Absolute bounds (on the term scale)
  lwr_abs <- upr_abs <- rep(NA_real_, length(mu_det))
  if (want_pi) {
    # When aleatoric/both enabled, .compute_uncertainty returns absolute lwr/upr
    lwr_abs <- as.numeric(preds$lwr)
    upr_abs <- as.numeric(preds$upr)
  }

  list(
    fit = mu_det,
    var_epistemic = as.numeric(preds$var_epistemic),
    var_aleatoric = as.numeric(preds$var_aleatoric),
    lwr_abs = lwr_abs,   # absolute term quantiles
    upr_abs = upr_abs
  )
}

# row-sum variance with NA propagation
.row_sum_var <- function(var_mat) {
  na_any <- apply(var_mat, 1L, function(z) any(is.na(z)))
  s <- rowSums(var_mat, na.rm = TRUE)
  s[na_any] <- NA_real_
  s
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
