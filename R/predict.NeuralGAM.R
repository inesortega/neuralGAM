#' Produces predictions from a fitted \code{neuralGAM} object
#'
#' @description
#' Generate predictions from a fitted \code{neuralGAM} model. Supported types:
#' \itemize{
#'   \item \code{type = "link"} (default): linear predictor on the link scale.
#'   \item \code{type = "response"}: predictions on the response scale.
#'   \item \code{type = "terms"}: per-term contributions to the linear predictor (no intercept).
#' }
#'
#' \strong{Uncertainty estimation via MC Dropout (epistemic only)}
#' \itemize{
#'   \item If \code{se.fit = TRUE}, standard errors (SE) of the \emph{fitted mean} are returned
#'         (mgcv-style via Monte Carlo Dropout).
#'   \item For \code{type = "response"}, SEs are mapped to the response scale by the delta method:
#'         \eqn{se_\mu = |d\mu/d\eta| \cdot se_\eta}.
#'   \item \code{interval = "confidence"} returns CI bands derived from SEs; prediction intervals are not supported.
#'   \item For \code{type = "terms"}, \code{interval="confidence"} returns per-term CI matrices (and \code{se.fit} when requested).
#' }
#'
#' \strong{Details}
#'
#' \itemize{
#'   \item Epistemic SEs (CIs) are obtained via Monte Carlo Dropout. When \code{type != "terms"}
#'         and SEs/CIs are requested in the presence of smooth terms, uncertainty is aggregated
#'         \emph{jointly} to capture cross-term covariance in a single MC pass set. Otherwise,
#'         per-term variances are used (parametric variances are obtained from \code{stats::predict(..., se.fit=TRUE)}).
#'   \item For \code{type="terms"}, epistemic SEs and CI matrices are returned when requested.
#'   \item PIs are not defined on the link scale and are not supported.
#' }
#'
#' @param object A fitted \code{neuralGAM} object.
#' @param newdata Optional \code{data.frame}/list of covariates at which to predict.
#'   If omitted, the training data cached in the object are used.
#' @param type One of \code{c("link","response","terms")}. Default \code{"link"}.
#' @param terms If \code{type = "terms"}, character vector of term names to include.
#'   If \code{NULL}, all terms are returned. Intercept is not included (as in \pkg{mgcv}).
#' @param se.fit Logical; if \code{TRUE}, return SEs of the fitted mean (epistemic). Default \code{FALSE}.
#'   For \code{type="terms"}, returns a matrix of per-term SEs when available.
#' @param interval One of \code{c("none","confidence")} (default \code{"none"}). For \code{type="terms"},
#'   setting \code{interval="confidence"} returns per-term CI matrices.
#' @param level Coverage level for confidence intervals (e.g., \code{0.95}). Default \code{0.95}.
#' @param forward_passes Integer; number of MC-dropout forward passes when computing
#'   epistemic uncertainty.
#' @param verbose Integer (0/1). Default \code{1}.
#' @param \ldots Other options (passed on to internal predictors).
#'
#' @return
#' \itemize{
#'   \item \code{type="terms"}:
#'     \itemize{
#'       \item \code{interval="none"}: matrix of per-term contributions; if \code{se.fit=TRUE}, a list with \code{$fit}, \code{$se.fit}.
#'       \item \code{interval="confidence"}: a list with matrices \code{$fit}, \code{$se.fit}, \code{$lwr}, \code{$upr}.
#'     }
#'   \item \code{type="link"} or \code{type="response"}:
#'     \itemize{
#'       \item \code{interval="none"}: vector (or list with \code{$fit}, \code{$se.fit} if \code{se.fit=TRUE}).
#'       \item \code{interval="confidence"}: data.frame with \code{fit}, \code{lwr}, \code{upr}.
#'     }
#' }
#' @examples
#' \dontrun{
#'
#' library(neuralGAM)
#' dat <- sim_neuralGAM_data()
#' train <- dat$train
#' test  <- dat$test
#'
#' ngam0 <- neuralGAM(
#'   y ~ s(x1) + x2 + s(x3),
#'   data = train, family = "gaussian",
#'   num_units = 128, uncertainty_method = "epistemic"
#' )
#' eta      <- predict(ngam0, type = "link")
#' mu       <- predict(ngam0, type = "response")
#' trm      <- predict(ngam0, type = "terms")
#' link_ci  <- predict(ngam0, type = "link", interval = "confidence", level = 0.95)
#' resp_ci  <- predict(ngam0, type = "response", interval = "confidence", level = 0.95)
#' trm_se   <- predict(ngam0, type = "terms", se.fit = TRUE)
#' }
#' @importFrom stats predict qnorm setNames
#' @export
#' @author Ines Ortega-Fernandez, Marta Sestelo
predict.neuralGAM <- function(object,
                              newdata = NULL,
                              type = c("link","response","terms"),
                              terms = NULL,
                              se.fit = FALSE,
                              interval = c("none","confidence"),
                              level = 0.95,
                              forward_passes = 150,
                              verbose = 1, ...) {
  stopifnot(inherits(object, "neuralGAM"))
  type     <- match.arg(type)
  interval <- match.arg(interval)
  ngam     <- object

  # ---- data ----
  if (is.null(newdata)){
    x <- ngam$x; use_cache <- TRUE
  } else {
    x <- as.data.frame(newdata); use_cache <- FALSE
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
  sel_terms <- if (is.null(terms)) all_terms else {
    if (!all(terms %in% all_terms)) stop("Unknown term(s) in `terms`: ", paste(setdiff(terms, all_terms), collapse = ", "))
    terms
  }

  # ---- per-term link-scale ----
  n <- nrow(x); p <- length(all_terms)
  term_fit <- matrix(0, n, p, dimnames = list(NULL, all_terms))
  var_epi  <- matrix(NA_real_, n, p, dimnames = list(NULL, all_terms))

  need_se_req <- isTRUE(se.fit) || interval == "confidence"
  has_np <- length(np_terms) > 0L

  if (use_cache) {
    term_fit[,] <- as.matrix(ngam$partial[, all_terms, drop = FALSE])
    if (!is.null(ngam$var_epistemic)) var_epi[,] <- as.matrix(ngam$var_epistemic[, all_terms, drop = FALSE])
  } else {
    # parametric terms (batch)
    if (length(p_terms)) {
      lm_data <- x[, p_terms, drop = FALSE]; colnames(lm_data) <- p_terms
      linmod <- ngam$model$linear
      if (!is.null(linmod)) {
        pr_terms <- stats::predict(linmod, newdata = lm_data, type = "terms", se.fit = need_se_req)
        if(need_se_req && !is.null(pr_terms$se.fit)){
          term_fit[, p_terms] <- unname(as.matrix(pr_terms$fit))
          var_epi[, p_terms] <- unname(as.matrix(pr_terms$se.fit))^2
        }
        else{
          term_fit[, p_terms] <- pr_terms
        }
      }
    }
    # nonparametric terms (epistemic only)
    if (length(np_terms)) {
      for (tm in np_terms) {
        need_se <- isTRUE(se.fit) || interval == "confidence"
        pt <- .ngam_predict_term_epistemic(
          ngam, x[[tm]], term_name = tm,
          want_se = need_se, level = level,
          forward_passes = forward_passes, verbose = verbose
        )
        center <- (ngam$term_center %||% setNames(rep(0, length(all_terms)), all_terms))[tm]
        term_fit[, tm] <- pt$fit - center
        var_epi[, tm]  <- pt$var_epistemic
      }
    }
  }

  # assemble link
  eta0 <- ngam$eta0 %||% 0
  eta  <- eta0 + rowSums(term_fit, na.rm = FALSE)

  # epistemic SE on link
  se_eta <- NULL
  if (need_se_req) {
    # per-term aggregation: LM full-fit variance + sum of smooth variances
    var_np_sum <- if (has_np) .row_sum_var(var_epi[, np_terms, drop=FALSE]) else rep(0, n)
    var_param_full <- rep(0, n)
    if (length(p_terms)) {
      linmod <- ngam$model$linear
      if (!is.null(linmod)) {
        lm_data <- x[, p_terms, drop = FALSE]; colnames(lm_data) <- p_terms
        pr_lin_all <- stats::predict(linmod, newdata = lm_data, se.fit = TRUE)
        var_param_full <- (as.numeric(pr_lin_all$se.fit))^2
      }
    }
    row_var <- var_param_full + var_np_sum
    se_eta  <- sqrt(pmax(row_var, 0))
  }

  # ---- type="terms" (now supports CI matrices) ----
  if (type == "terms") {
    fit_terms <- term_fit[, sel_terms, drop = FALSE]
    if (!need_se_req) return(fit_terms)

    se_terms <- sqrt(pmax(var_epi[, sel_terms, drop = FALSE], 0))
    if (interval == "none" && isTRUE(se.fit)) {
      return(list(fit = fit_terms, se.fit = se_terms))
    }
    # interval == "confidence": return CI matrices (and se.fit for convenience)
    z <- stats::qnorm(1 - (1 - level)/2)
    lwr <- fit_terms - z * se_terms
    upr <- fit_terms + z * se_terms
    return(list(fit = fit_terms, se.fit = se_terms, lwr = lwr, upr = upr))
  }

  # ---- type="link" ----
  if (type == "link") {
    if (interval == "none") {
      if (!se.fit) return(eta)
      return(list(fit = eta, se.fit = se_eta))
    } else {
      z <- stats::qnorm(1 - (1 - level)/2)
      lwr_ci <- eta - z * se_eta; upr_ci <- eta + z * se_eta
      if (!any(is.finite(lwr_ci)) || !any(is.finite(upr_ci))) {
        warning("Confidence intervals not available on link scale (missing epistemic variance).")
      }
      return(data.frame(fit = eta, lwr = lwr_ci, upr = upr_ci))
    }
  }

  # ---- type="response" ----
  mu <- inv_link(ngam$family, eta)
  if (interval == "none") {
    if (!se.fit) return(mu)
    gprime <- abs(mu_eta(ngam$family, eta))
    se_mu  <- gprime * se_eta
    return(list(fit = mu, se.fit = se_mu))
  } else {
    z <- stats::qnorm(1 - (1 - level)/2)
    gprime <- abs(mu_eta(ngam$family, eta)); se_mu <- gprime * se_eta
    lwr_ci <- mu - z * se_mu; upr_ci <- mu + z * se_mu
    if (any(is.na(se_mu))) warning("Confidence intervals not available (missing epistemic variance).")
    return(data.frame(fit = mu, lwr = lwr_ci, upr = upr_ci))
  }
}

.ngam_predict_term_epistemic <- function(ngam, xvec, term_name,
                                         want_se = FALSE,
                                         verbose = 1, level = 0.95,
                                         forward_passes = 30) {
  mdl <- ngam$model[[term_name]]
  X <- xvec; if (is.null(dim(X))) X <- matrix(X, ncol = 1L)

  y_det <- try(as.matrix(mdl$predict(X, verbose = 0)), silent = TRUE)
  if (inherits(y_det, "try-error") || is.null(dim(y_det))) {
    fit <- as.numeric(mdl$predict(X, verbose = 0))
    return(list(fit = fit, var_epistemic = rep(NA_real_, length(fit))))
  }

  nout <- ncol(y_det)
  mean_col <- if (nout >= 3L) 3L else 1L
  mu_det <- as.numeric(y_det[, mean_col])
  # Do not center here. Centering is applied upstream using training-time term_center.

  var_ep <- rep(NA_real_, length(mu_det))
  if (isTRUE(want_se)) {
    preds <- .compute_uncertainty(mdl, X, uncertainty_method = "epistemic",
                                  alpha = 1 - level, forward_passes = forward_passes)
    var_ep <- as.numeric(preds$var_epistemic)
  }
  list(fit = mu_det, var_epistemic = var_ep)
}

.row_sum_var <- function(var_mat) {
  na_any <- apply(var_mat, 1L, function(z) any(is.na(z)))
  s <- rowSums(var_mat, na.rm = TRUE)
  s[na_any] <- NA_real_
  s
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

#' Internal helper: joint epistemic SE on link scale
#'
#' @description
#' Computes joint epistemic standard errors on the link scale by aggregating
#' across all smooth terms via MC Dropout, capturing cross-term covariance.
#' Parametric model uncertainty (from the linear submodel) is added assuming
#' independence from NN-based epistemic uncertainty.
#'
#' @param ngam Fitted \code{neuralGAM} object.
#' @param x New data frame of covariates.
#' @param forward_passes Number of MC Dropout passes (default 300).
#' @param verbose Verbosity (0/1).
#'
#' @return A numeric vector of length \code{nrow(x)} giving epistemic SEs on the link scale.
#'
#' @details
#' Steps:
#' \enumerate{
#'   \item Parametric part: mean + variance from linear model.
#'   \item Nonparametric part: pass-level sums across all smooths.
#'   \item Joint across-pass variance captures covariance between smooths.
#'   \item Combined with parametric variance (assumed independent).
#' }
#' @keywords internal
.joint_se_eta_mcdropout <- function(ngam, x,
                                    forward_passes = 300,   # 300–1000 recommended for smooth bands
                                    verbose = 0) {
  p_terms  <- ngam$formula$p_terms %||% character(0L)
  np_terms <- ngam$formula$np_terms %||% character(0L)

  n <- nrow(x)
  # 1) Parametric component: mean & SE (includes intercept)
  eta_param_fit <- rep(0, n)
  var_param     <- rep(0, n)
  if (length(p_terms)) {
    lm_data <- x[, p_terms, drop = FALSE]
    colnames(lm_data) <- p_terms
    linmod <- ngam$model$linear
    if (!is.null(linmod)) {
      pr_lin <- stats::predict(linmod, newdata = lm_data, se.fit = TRUE)
      eta_param_fit <- as.numeric(pr_lin$fit)               # includes intercept
      var_param     <- (as.numeric(pr_lin$se.fit))^2
    } else {
      eta_param_fit <- rep(ngam$eta0 %||% 0, n)
    }
  } else {
    # If no parametric part, intercept lives in eta0; keep var_param = 0
    eta_param_fit <- rep(ngam$eta0 %||% 0, n)
  }

  # 2) Nonparametric component: joint MC Dropout over all smooths
  if (!length(np_terms)) {
    # No smooths: SE = sqrt(var_param)
    return(sqrt(pmax(var_param, 0)))
  }

  passes <- max(2L, as.integer(forward_passes))
  eta_np_pass <- matrix(0.0, nrow = passes, ncol = n)  # each row = one pass

  for (tm in np_terms) {
    mdl <- ngam$model[[tm]]
    Xtm <- x[[tm]]; if (is.null(dim(Xtm))) Xtm <- matrix(Xtm, ncol = 1L)

    probe <- try(as.matrix(mdl$predict(Xtm, verbose = 0)), silent = TRUE)
    if (inherits(probe, "try-error") || is.null(dim(probe))) {
      # fallback: deterministic mean only, no extra variance
      mu_det <- as.numeric(mdl$predict(Xtm, verbose = 0))
      eta_np_pass <- sweep(eta_np_pass, 2L, mu_det, `+`)
      next
    }
    nout <- ncol(probe)
    mean_col <- if (nout >= 3L) 3L else 1L

    # MC Dropout forward: returns [passes, n, nout] or [passes, n] if nout==1 --> this is prepared for future integration of aleatoric uncertainty via Quantile Regression.
    y_arr <- .mc_dropout_forward(mdl, Xtm, passes = passes, output_dim = nout)
    y_mat <- if (length(dim(y_arr)) == 2L) {
      y_arr                               # [passes, n] (single-head)
    } else {
      y_arr[, , mean_col, drop = TRUE]    # [passes, n] (mean head)
    }

    # Accumulate this term's mean head across passes
    eta_np_pass <- eta_np_pass + y_mat
  }

  # 3) Full linear predictor per pass (parametric mean + sum of smooths per pass)
  #    Note: parametric part is deterministic across passes here; its *uncertainty*
  #    is added via var_param, assuming independence.
  eta_full_pass <- sweep(eta_np_pass, 2L, eta_param_fit, `+`)  # [passes, n]

  # 4) Joint epistemic variance from passes (captures cross-term covariance across smooths)
  var_ep_joint <- apply(eta_full_pass, 2L, stats::var)  # length n

  # 5) Combine with parametric variance (independence assumption)
  var_eta <- var_ep_joint + var_param
  sqrt(pmax(var_eta, 0))
}
