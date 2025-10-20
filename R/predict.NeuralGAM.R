#' Predict from a fitted \code{neuralGAM} (quantile-based epistemic uncertainty)
#'
#' @description
#' Generate predictions from a fitted \code{neuralGAM} model. Supported types:
#' \itemize{
#'   \item \code{type = "link"} (default): linear predictor on the link scale.
#'   \item \code{type = "response"}: predictions on the response scale.
#'   \item \code{type = "terms"}: per-term contributions to the linear predictor (no intercept).
#' }
#'
#' \strong{Uncertainty estimation (epistemic only)}\cr
#' Epistemic uncertainty is computed via \emph{joint draws}:
#' \itemize{
#'   \item \strong{Nonparametric (smooth) part}: Monte Carlo Dropout across all smooth networks
#'         in joint passes to capture cross-term covariance.
#'   \item \strong{Parametric (linear) part}: Gaussian coefficient draws
#'         \eqn{\beta^{(b)} \sim \mathcal{N}(\hat\beta,\widehat{\mathrm{Var}}(\hat\beta))} evaluated with the
#'         linear submodel's design matrix.
#' }
#' For \code{type != "terms"}, the link-scale draws \eqn{\{\eta^{(b)}\}} are formed by summing the parametric
#' and smooth contributions (plus intercept). For \code{type = "response"}, the inverse link is applied to
#' each draw \eqn{\mu^{(b)} = h^{-1}(\eta^{(b)})} and uncertainty is summarized \emph{on the response scale}
#' directly from these draws. \strong{Confidence intervals are empirical quantiles across draws}
#' (no Normal or delta-method approximations).
#'
#' \strong{Returned standard errors}\cr
#' If \code{se.fit = TRUE}, standard errors are the \emph{sample standard deviation of the draws}
#' (on the requested scale): \eqn{\widehat{SE} = \mathrm{sd}\{\eta^{(b)}\}} for \code{type="link"} and
#' \eqn{\widehat{SE} = \mathrm{sd}\{\mu^{(b)}\}} for \code{type="response"}.
#'
#' \strong{Confidence intervals}\cr
#' If \code{interval = "confidence"}, the lower/upper bounds are empirical
#' quantiles at levels \eqn{\alpha/2} and \eqn{1-\alpha/2} across the relevant draw set:
#' \itemize{
#'   \item \code{type="link"}: quantiles of \eqn{\{\eta^{(b)}\}}.
#'   \item \code{type="response"}: quantiles of \eqn{\{\mu^{(b)}\}} with \eqn{\mu^{(b)} = h^{-1}(\eta^{(b)})}.
#'   \item \code{type="terms"}: per-term quantiles; smooth terms use per-term dropout draws,
#'         parametric terms use contributions from the coefficient draws.
#' }
#'
#' \strong{Notes}\cr
#' \itemize{
#'   \item Prediction intervals (aleatoric + epistemic) are not returned by this method.
#'   \item Set a random seed for reproducibility of the Monte Carlo draws.
#'   \item The number of passes \code{forward_passes} controls tail stability of the empirical quantiles.
#' }
#'
#' @param object A fitted \code{neuralGAM} object.
#' @param newdata Optional \code{data.frame} or \code{list} of covariates at which to predict.
#'   If omitted, the training data cached in the object are used.
#' @param type One of \code{c("link","response","terms")}. Default \code{"link"}.
#' @param terms If \code{type = "terms"}, character vector of term names to include.
#'   If \code{NULL}, all terms are returned. Intercept is not included (as in \pkg{mgcv}).
#' @param se.fit Logical; if \code{TRUE}, return \emph{SD of draws} as standard errors.
#'   For \code{type="terms"}, returns a matrix of per-term SDs when available. Default \code{FALSE}.
#' @param interval One of \code{c("none","confidence")} (default \code{"none"}).
#'   For \code{type="terms"}, \code{interval="confidence"} returns per-term CI matrices.
#' @param level Coverage level for confidence intervals (e.g., \code{0.95}). Default \code{0.95}.
#' @param forward_passes Integer; number of joint Monte Carlo passes (dropout + parametric draws)
#'   used to compute uncertainty. Larger values stabilize tail quantiles. Default \code{150}.
#' @param verbose Integer (0/1). Default \code{1}.
#' @param \ldots Other options (passed to internal predictors).
#'
#' @return
#' \itemize{
#'   \item \code{type="terms"}:
#'     \itemize{
#'       \item \code{interval="none"}: matrix of per-term contributions;
#'             if \code{se.fit=TRUE}, a list with \code{$fit}, \code{$se.fit}.
#'       \item \code{interval="confidence"}: a list with matrices \code{$fit}, \code{$se.fit}, \code{$lwr}, \code{$upr}
#'             (per-term empirical quantile bands).
#'     }
#'   \item \code{type="link"} or \code{type="response"}:
#'     \itemize{
#'       \item \code{interval="none"}: vector (or list with \code{$fit}, \code{$se.fit} if \code{se.fit=TRUE}),
#'             where \code{se.fit} is the SD of draws on the requested scale.
#'       \item \code{interval="confidence"}: \code{data.frame} with \code{fit}, \code{se.fit}, \code{lwr}, \code{upr},
#'             where \code{lwr}/\code{upr} are empirical quantiles across draws.
#'     }
#' }
#'
#' @examples \dontrun{
#' set.seed(1)
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
#'
#' # Link-scale empirical-quantile CIs (joint dropout + beta draws)
#' link_ci <- predict(ngam0, type = "link", interval = "confidence",
#'                    level = 0.95, forward_passes = 300)
#'
#' # Response-scale empirical-quantile CIs (transform draws then take quantiles)
#' resp_ci <- predict(ngam0, type = "response", interval = "confidence",
#'                    level = 0.95, forward_passes = 300)
#'
#' # Per-term bands: smooth terms via per-term dropout, parametric via beta draws
#' trm_ci <- predict(ngam0, type = "terms", se.fit = TRUE, interval = "confidence",
#'                   level = 0.95, forward_passes = 300)
#' }
#'
#' @importFrom stats predict setNames
#' @method predict neuralGAM
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

  # -------------------------------
  # Prepare data
  # -------------------------------
  if (is.null(newdata)) {
    x <- ngam$x; use_cache <- TRUE
  } else {
    x <- as.data.frame(newdata); use_cache <- FALSE
  }
  needed <- colnames(ngam$x)
  if (!all(needed %in% colnames(x))) {
    stop("newdata is missing required columns: ",
         paste(setdiff(needed, colnames(x)), collapse = ", "))
  }
  x <- x[, needed, drop = FALSE]

  p_terms  <- ngam$formula$p_terms %||% character(0L)
  np_terms <- ngam$formula$np_terms %||% character(0L)
  all_terms <- c(p_terms, np_terms)
  sel_terms <- if (is.null(terms)) all_terms else {
    if (!all(terms %in% all_terms))
      stop("Unknown term(s) in `terms`: ",
           paste(setdiff(terms, all_terms), collapse = ", "))
    terms
  }

  n <- nrow(x)
  need_unc <- isTRUE(se.fit) || interval == "confidence"

  # -------------------------------
  # 2️ Compute per-term contributions
  # -------------------------------
  term_fit <- matrix(0, n, length(all_terms), dimnames = list(NULL, all_terms))
  var_epi  <- matrix(NA_real_, n, length(all_terms), dimnames = list(NULL, all_terms))

  if (use_cache) {
    term_fit[,] <- as.matrix(ngam$partial[, all_terms, drop = FALSE])
    if (!is.null(ngam$var_epistemic))
      var_epi[,] <- as.matrix(ngam$var_epistemic[, all_terms, drop = FALSE])
  } else {
    # ---- parametric terms ----
    if (length(p_terms)) {
      linmod <- ngam$model$linear
      if (!is.null(linmod)) {
        pr <- stats::predict(linmod, newdata = x[, p_terms, drop = FALSE],
                             type = "terms", se.fit = FALSE)
        term_fit[, p_terms] <- unname(as.matrix(pr))
      }
    }

    # ---- smooth (nonparametric) terms ----
    if (length(np_terms)) {
      for (tm in np_terms) {
        pt <- .ngam_predict_term_epistemic(
          ngam, x[[tm]], term_name = tm,
          want_se = need_unc, level = level,
          forward_passes = forward_passes, verbose = verbose
        )
        center_j <- (ngam$term_center %||%
                       setNames(rep(0, length(all_terms)), all_terms))[tm]
        term_fit[, tm] <- pt$fit - center_j
        var_epi[, tm]  <- pt$var_epistemic
      }
    }
  }

  # get eta in the usual way as a fallback
  eta0 <- ngam$eta0
  eta  <- eta0 + rowSums(term_fit, na.rm = FALSE)

  # -------------------------------
  # Type = "terms" : per-term bands
  # -------------------------------
  if (type == "terms") {
    fit_terms <- term_fit[, sel_terms, drop = FALSE]
    if (!need_unc) return(fit_terms)

    B <- max(2L, as.integer(forward_passes))
    term_sd  <- term_lwr <- term_upr <- matrix(NA_real_, nrow = n, ncol = length(sel_terms),
                                               dimnames = list(NULL, sel_terms))

    # Get parametric draws once
    par_out <- .parametric_draws(ngam, x, forward_passes = forward_passes)

    for (tm in sel_terms) {
      if (tm %in% np_terms) {
        # MC Dropout draws for smooth term
        mdl <- ngam$model[[tm]]
        Xtm <- if (is.null(dim(x[[tm]]))) matrix(x[[tm]], ncol = 1L) else x[[tm]]
        probe <- try(as.matrix(mdl$predict(Xtm, verbose = 0)), silent = TRUE)

        if (inherits(probe, "try-error") || is.null(dim(probe))) {
          draw_mat <- matrix(NA_real_, nrow = B, ncol = n)
        } else {
          nout <- ncol(probe); mean_col <- if (nout >= 3L) 3L else 1L
          y_arr <- .mc_dropout_forward(mdl, Xtm, passes = B, output_dim = nout)
          draw_mat <- if (length(dim(y_arr)) == 2L) y_arr else y_arr[, , mean_col, drop = TRUE]
        }
      } else {
        # Parametric term draws from precomputed .parametric_draws()
        draw_mat <- par_out$term_draws[[tm]]
        if (is.null(draw_mat))
          draw_mat <- matrix(NA_real_, nrow = B, ncol = n)
      }

      bias_vec <- fit_terms[, tm] - colMeans(draw_mat, na.rm = TRUE)
      draw_mat <- sweep(draw_mat, 2L, bias_vec, `+`)

      term_sd[, tm]  <- apply(draw_mat, 2L, stats::sd, na.rm = TRUE)
      term_lwr[, tm] <- as.numeric(matrixStats::colQuantiles(draw_mat, probs = (1 - level)/2, na.rm = TRUE))
      term_upr[, tm] <- as.numeric(matrixStats::colQuantiles(draw_mat, probs = 1 - (1 - level)/2, na.rm = TRUE))
    }

    if (interval == "none" && isTRUE(se.fit))
      return(list(fit = fit_terms, se.fit = term_sd))

    return(list(fit = fit_terms, se.fit = term_sd, lwr = term_lwr, upr = term_upr))
  }

  # -------------------------------
  # Compute joint link-scale draws if needed (from parametric + NP)
  # -------------------------------
  if (type %in% c("link","response") && need_unc) {
    eta_draws <- .joint_draws_eta(ngam, x, eta, forward_passes = forward_passes, verbose = verbose)
    # sanitize
    eta_draws[!is.finite(eta_draws)] <- NA_real_
    se_link  <- apply(eta_draws, 2L, stats::sd, na.rm = TRUE)
    lwr_link <- matrixStats::colQuantiles(eta_draws, probs = (1 - level)/2, na.rm = TRUE)
    upr_link <- matrixStats::colQuantiles(eta_draws, probs = 1 - (1 - level)/2, na.rm = TRUE)
  }

  # -------------------------------
  # Type = "link" : link-scale CIs
  # -------------------------------
  if (type == "link") {
    if (interval == "none") {
      if (!isTRUE(se.fit)) return(eta)
      return(list(fit = eta, se.fit = se_link))
    }
    return(data.frame(fit = eta, se.fit = se_link,
                      lwr = as.numeric(lwr_link),
                      upr = as.numeric(upr_link)))
  }

  # -------------------------------
  # Type = "response" : transform each draw
  # -------------------------------
  if (type == "response") {
    mu <- inv_link(ngam$family, eta)
    if (!need_unc) return(mu)

    mu_draws <- inv_link(ngam$family, eta_draws)
    se_mu  <- apply(mu_draws, 2L, stats::sd, na.rm = TRUE)
    lwr_mu <- matrixStats::colQuantiles(mu_draws, probs = (1 - level)/2, na.rm = TRUE)
    upr_mu <- matrixStats::colQuantiles(mu_draws, probs = 1 - (1 - level)/2, na.rm = TRUE)

    if (interval == "none" && isTRUE(se.fit))
      return(list(fit = mu, se.fit = se_mu))

    return(data.frame(fit = mu, se.fit = se_mu,
                      lwr = as.numeric(lwr_mu),
                      upr = as.numeric(upr_mu)))
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

#' Joint MC-Dropout draws of the additive predictor on the link scale
#'
#' @description
#' Generate \strong{joint} draws of the additive predictor \eqn{\eta(\boldsymbol{x})}
#' on the link scale by combining:
#' \itemize{
#'   \item \emph{Parametric} uncertainty via Gaussian draws of the linear
#'         coefficients \eqn{\beta^{(b)} \sim \mathcal{N}(\hat\beta, \widehat{\mathrm{Var}}(\hat\beta))},
#'         evaluated with the linear model matrix; and
#'   \item \emph{Nonparametric} (smooth) epistemic uncertainty via Monte Carlo
#'         Dropout draws, summing all smooth terms per pass.
#' }
#' The function returns a \eqn{B \times n} matrix of link-scale draws, where
#' \eqn{B} is the number of MC passes and \eqn{n} is the number of rows in \code{x}.
#'
#' @param ngam A fitted \code{neuralGAM} object containing:
#'   \itemize{
#'     \item \code{model$linear}: the fitted linear (parametric) submodel, or \code{NULL};
#'     \item \code{model[[tm]]}: Keras models for each smooth term \code{tm};
#'     \item \code{formula$p_terms}, \code{formula$np_terms}: character vectors with
#'           parametric and nonparametric term names;
#'     \item \code{eta0}: (numeric) intercept on the link scale.
#'   }
#' @param x A \code{data.frame} or \code{matrix} of covariates with the columns
#'   expected by \code{ngam}. The number of rows defines \eqn{n}.
#' @param eta The deterministic eta to be used for bias correction
#' @param forward_passes Integer \eqn{B \ge 2}; number of Monte Carlo passes (default \code{300L}).
#' @param verbose Integer flag (0/1); controls optional messaging (unused at present).
#'
#' @return A numeric matrix of dimension \code{B x n} containing draws of
#'   \eqn{\eta(\boldsymbol{x})} on the link scale (each row is one pass, each
#'   column an observation).
#'
#' @details
#' \strong{Parametric part:} draws are generated with \code{MASS::mvrnorm} from
#' \code{coef(linmod)} and \code{vcov(linmod)}; predictions are computed using the
#' linear submodel’s \emph{exact} model matrix (via \code{stats::model.frame} and
#' \code{stats::model.matrix}), excluding the intercept. If no parametric terms exist, this part contributes zeros.
#'
#' \strong{Nonparametric part:} for each smooth term, the function calls
#' \code{.mc_dropout_forward()} and sums the chosen “mean” head across terms per pass.
#' If a smooth model cannot produce draws (e.g., errors), a deterministic prediction
#' is used (no extra variance contribution).
#'
#' The returned draws include the intercept \code{eta0} and are bias corrected.
#'
#' @seealso \code{\link{.parametric_draws}}, \code{\link{.mc_dropout_forward}}
#'
#' @keywords internal
#' @importFrom MASS mvrnorm
#' @importFrom stats coef vcov model.frame model.matrix formula
.joint_draws_eta <- function(ngam, x, eta_det, forward_passes = 300L, verbose = 0L, coef_draw = c("t", "normal")) {
  stopifnot(is.data.frame(x) || is.matrix(x))
  p_terms  <- ngam$formula$p_terms %||% character(0L)
  np_terms <- ngam$formula$np_terms %||% character(0L)
  B <- max(2L, as.integer(forward_passes))
  n <- nrow(x)
  coef_draw = match.arg(coef_draw)

  ## --- Parametric draws (NO intercept here) ---
  eta_param_draws <- matrix(0.0, nrow = B, ncol = n)

  if (length(p_terms)) {
    linmod <- ngam$model$linear
    if (!is.null(linmod)) {
      new_df <- as.data.frame(x)
      Xmm <- .mm_from_lm(linmod, new_df)   # may include (Intercept)
      beta_hat <- stats::coef(linmod)
      nu <- stats::df.residual(linmod)
      Vb <- try(stats::vcov(linmod), silent = TRUE)
      if (inherits(Vb, "try-error") || anyNA(Vb)) Vb <- diag(length(beta_hat))

      # ---- Drop intercept everywhere ----
      keep <- !is.na(beta_hat) & names(beta_hat) != "(Intercept)"
      if (any(keep)) {
        beta_hat <- beta_hat[keep]
        Vb <- Vb[keep, keep, drop = FALSE]
        Xmm <- Xmm[, names(beta_hat), drop = FALSE]
        if(coef_draw == "t"){
          Beta_centered <- mvtnorm::rmvt(n = B, sigma = Vb, df = nu, type = "shifted")
          Beta <- sweep(Beta_centered, 2L, beta_hat, `+`)
        }
        else if(coef_draw == "normal"){
          Beta <- MASS::mvrnorm(n = B, mu = beta_hat, Sigma = Vb)
        }
        if (is.vector(Beta)) Beta <- matrix(Beta, nrow = B)
        eta_param_draws <- unname(Beta %*% t(Xmm))
      } else {
        # no parametric columns after dropping intercept
        eta_param_draws[,] <- 0
      }
    }
  }

  ## --- Nonparametric joint dropout: sum across smooths per pass ---
  eta_np_draws <- matrix(0.0, nrow = B, ncol = n)
  if (length(np_terms)) {
    for (tm in np_terms) {
      mdl <- ngam$model[[tm]]
      Xtm <- x[[tm]]
      if (is.null(dim(Xtm))) Xtm <- matrix(Xtm, ncol = 1L)

      probe <- try(as.matrix(mdl$predict(Xtm, verbose = 0)), silent = TRUE)

      if (inherits(probe, "try-error") || is.null(dim(probe))) {
        # deterministic fallback (no dropout)
        mu_det <- as.numeric(mdl$predict(Xtm, verbose = 0))
        eta_np_draws <- sweep(eta_np_draws, 2L, mu_det, `+`)
      } else {
        nout <- ncol(probe)
        mean_col <- if (nout >= 3L) 3L else 1L
        y_arr <- .mc_dropout_forward(mdl, Xtm, passes = B, output_dim = nout)
        f_mat <- if (length(dim(y_arr)) == 2L) y_arr else y_arr[, , mean_col, drop = TRUE]  # B × n
        eta_np_draws <- eta_np_draws + f_mat
      }
    }
  }
  # eta_np_draws has accumulated f_mat from all np_terms + intercept + p_terms

  ## --- Combine: add ONLY intercept from ngam$eta0 never from linear model ---
  eta_draws <- eta_param_draws + eta_np_draws + (ngam$eta0 %||% 0)

  ## --- Bias correction to match deterministic eta (which already uses eta0) ---
  bias_vec <- as.numeric(eta_det) - colMeans(eta_draws, na.rm = TRUE)
  eta_draws <- sweep(eta_draws, 2L, bias_vec, `+`)

  eta_draws
}


.mm_from_lm <- function(linmod, new_df) {
  stopifnot(inherits(linmod, c("lm")))
  tt  <- stats::terms(linmod)
  ttX <- stats::delete.response(tt)

  # recover training model.frame (most reliable source of factor info)
  mf_fit <- try(stats::model.frame(linmod), silent = TRUE)
  if (inherits(mf_fit, "try-error"))
    stop("Could not retrieve training model.frame from linmod.")

  # training levels and contrasts
  xlev_fit <- stats::.getXlevels(tt, mf_fit)
  contr_fit <- attr(mf_fit, "contrasts")
  if (is.null(contr_fit)) contr_fit <- linmod$contrasts

  # coerce characters to factors with training levels
  new_df <- as.data.frame(new_df)
  for (nm in names(xlev_fit)) {
    if (!nm %in% names(new_df)) {
      stop("newdata is missing required column: ", nm)
    }
    if (!is.factor(new_df[[nm]])) {
      # convert character/integer to factor with training levels
      new_df[[nm]] <- factor(new_df[[nm]], levels = xlev_fit[[nm]])
    } else {
      # relevel existing factor to training order (keeps unseen as NA)
      new_df[[nm]] <- factor(as.character(new_df[[nm]]), levels = xlev_fit[[nm]])
    }
  }

  # build model.frame/model.matrix with training levels & contrasts
  mf_new <- stats::model.frame(ttX, data = new_df, na.action = stats::na.pass, xlev = xlev_fit)
  Xmm <- stats::model.matrix(ttX, mf_new, contrasts.arg = contr_fit)

  Xmm
}


#' Parametric draws and per-term parametric contributions
#'
#' @description
#' Generate Gaussian draws of the linear coefficients \eqn{\beta} from the fitted
#' parametric submodel and compute:
#' \itemize{
#'   \item \strong{\code{beta_draws}}: a \eqn{B \times p_\beta} matrix of coefficient draws,
#'   \item \strong{\code{eta_param_draws}}: a \eqn{B \times n} matrix of the parametric
#'         part of the linear predictor \eqn{X\beta^{(b)}}, and
#'   \item \strong{\code{term_draws}}: a named list of \eqn{B \times n} matrices, one per
#'         parametric term, giving its per-observation contribution across draws.
#' }
#' Returns empty placeholders if no parametric terms are present or the linear submodel is \code{NULL}.
#'
#' @param ngam A fitted \code{neuralGAM} object with a linear submodel at \code{$model$linear}.
#' @param x A \code{data.frame}/\code{matrix} with the parametric covariates required by
#'   the linear submodel. The number of rows defines \eqn{n}.
#' @param forward_passes Integer \eqn{B \ge 2}; number of coefficient draws (default \code{300L}).
#'
#' @return A list with components:
#' \itemize{
#'   \item \code{beta_draws}: \eqn{B \times p_\beta} matrix of coefficient draws (or \code{NULL});
#'   \item \code{eta_param_draws}: \eqn{B \times n} matrix of parametric predictor draws (or \code{NULL});
#'   \item \code{term_draws}: named list of \eqn{B \times n} matrices with term-wise contributions.
#' }
#'
#' @details
#' The function reconstructs the linear submodel’s design matrix using
#' \code{stats::model.frame} and \code{stats::model.matrix} to ensure exact alignment with
#' the fitted formula (including factors, contrasts, and interactions). For each named
#' parametric term in \code{ngam$formula$p_terms}, the per-term contribution is computed
#' by selecting the subset of design columns associated with that term and multiplying
#' by the corresponding subset of coefficient draws. If a term cannot be matched to
#' specific design columns (e.g., aliased), it is skipped.
#'
#' @seealso \code{\link{.joint_draws_eta}}
#'
#' @keywords internal
#' @importFrom MASS mvrnorm
#' @importFrom stats coef vcov model.frame model.matrix formula
.parametric_draws <- function(ngam, x, forward_passes = 300L, coef_draw = c("t", "normal")) {
  `%||%` <- function(a, b) if (!is.null(a)) a else b

  p_terms <- ngam$formula$p_terms %||% character(0L)
  coef_draw = match.arg(coef_draw)
  out <- list(beta_draws = NULL, term_draws = list(), eta_param_draws = NULL)
  if (!length(p_terms)) return(out)

  linmod <- ngam$model$linear
  if (is.null(linmod)) return(out)

  # --- Set up ---
  B <- max(2L, as.integer(forward_passes))
  new_df <- as.data.frame(x)[p_terms]

  tt <- stats::terms(linmod)
  ttX <- stats::delete.response(tt)

  # --- Build model.frame and design matrix exactly like training ---
  Xmm <- .mm_from_lm(linmod, new_df)

  beta_hat <- stats::coef(linmod)
  keep_coef <- !is.na(beta_hat) & names(beta_hat) != "(Intercept)"  # <-- drop intercept
  beta_hat <- beta_hat[keep_coef]
  nu <- stats::df.residual(linmod)  # residual df = n - p for OLS

  # Align X with kept coefficients
  if (!all(names(beta_hat) %in% colnames(Xmm))) {
    stop("Column names of model matrix do not match coefficients after aliasing.")
  }
  Xmm <- Xmm[, names(beta_hat), drop = FALSE]

  Vb <- try(stats::vcov(linmod), silent = TRUE)
  if (inherits(Vb, "try-error") || anyNA(Vb)) {
    Vb <- diag(length(stats::coef(linmod)))
  }
  Vb <- Vb[keep_coef, keep_coef, drop = FALSE]

  # Draws (may end up empty if only intercept existed)
  if (length(beta_hat)) {
    if(coef_draw == "t"){
      Beta_centered <- mvtnorm::rmvt(n = B, sigma = Vb, df = nu, type = "shifted")
      Beta <- sweep(Beta_centered, 2L, beta_hat, `+`)
    }
    else if(coef_draw == "normal"){
      Beta <- MASS::mvrnorm(n = B, mu = beta_hat, Sigma = Vb)
    }
    if (is.vector(Beta)) Beta <- matrix(Beta, nrow = B)
    eta_param_draws <- unname(Beta %*% t(Xmm))
  } else {
    Beta <- matrix(numeric(0), nrow = B, ncol = 0)
    eta_param_draws <- matrix(0.0, nrow = B, ncol = n)
  }

  # --- Full parametric linear predictor draws: B x n ---
  eta_param_draws <- Beta %*% t(Xmm)
  eta_param_draws <- unname(eta_param_draws)

  for (tm in p_terms) {
    tm_norm <- gsub("`", "", tm, fixed = TRUE)
    cn_norm <- gsub("`", "", colnames(Xmm), fixed = TRUE)

    # --- Find all design columns that correspond to this term ---
    # Matches if the column starts with "tm" followed by nothing or ":" (for interactions)
    # or equals "tm" exactly (numeric or single factor).
    idx <- grep(paste0("^", tm_norm, "$|^", tm_norm, ":"), cn_norm)

    # If nothing found, try a relaxed partial match (e.g., "poly(x, 2)")
    if (!length(idx)) {
      idx <- grep(paste0("^", tm_norm), cn_norm)
    }

    if (!length(idx)) next  # skip if still not found

    # --- Compute contribution for this term ---
    Xi <- Xmm[, idx, drop = FALSE]
    Betai <- Beta[, idx, drop = FALSE]
    term_draws <- Betai %*% t(Xi)

    out$term_draws[[tm]] <- unname(term_draws)
  }

  out$beta_draws <- unname(Beta)
  out$eta_param_draws <- eta_param_draws
  out
}
