#' Internal helper: compute uncertainty decomposition (epistemic / aleatoric / both)
#'
#' @description
#' Given a fitted Keras submodel and covariate input \code{x}, compute uncertainty
#' estimates according to the \code{pi_method}.
#'
#' - \code{"epistemic"}: estimates only epistemic variance (via MC Dropout passes).
#' - \code{"aleatoric"}: uses deterministic quantile heads to estimate aleatoric variance.
#' - \code{"both"}: combines aleatoric and epistemic using variance decomposition.
#' - Otherwise: returns \code{NA} placeholders.
#'
#' @param model Fitted Keras model for a single smooth term.
#' @param x Input covariate matrix (or vector; will be reshaped as needed).
#' @param pi_method Character; one of \code{"epistemic"}, \code{"aleatoric"}, \code{"both"}, or \code{"none"}.
#' @param alpha Coverage level (e.g. 0.05 for 95% bands).
#' @param forward_passes Integer; number of MC Dropout passes.
#' @param inner_samples Integer; number of inner samples per pass when combining uncertainties.
#'
#' @return A \code{data.frame} with columns:
#'   \itemize{
#'     \item \code{lwr}, \code{upr}: lower/upper bounds of interval estimates.
#'     \item \code{var_epistemic}: epistemic variance.
#'     \item \code{var_aleatoric}: aleatoric variance.
#'     \item \code{var_total}: total variance (epistemic + aleatoric).
#'   }
#'
#' @importFrom stats rnorm qnorm
#' @keywords internal
.compute_uncertainty <- function(model, x, pi_method, alpha, forward_passes, inner_samples) {

  # ---- Uncertainty estimation given a fitted model ----

  preds <- NULL
  mu_det <- model %>% predict(x, verbose = 0)

  lower_q <- alpha / 2
  upper_q <- 1 - alpha / 2

  if (is.null(dim(x))) x <- matrix(x, ncol = 1L)

  if (pi_method == "epistemic") {
    # Compute only epistemic uncertainty by using `forward_passes` with Dropout layer -> output_dim = 1L
    y_array <- .mc_dropout_forward(model, x, forward_passes, output_dim = 1L)
    y_mat   <- y_array[, , 1]

    ci_lower <- matrixStats::colQuantiles(y_mat, probs = lower_q)
    ci_upper <- matrixStats::colQuantiles(y_mat, probs = upper_q)
    y_var    <- matrixStats::colVars(y_mat)

    preds <- data.frame(
      lwr = ci_lower,
      upr = ci_upper,
      var_epistemic = y_var,
      var_aleatoric = NA_real_,
      var_total     = y_var
    )
  } else if (pi_method == "both") {
    # MC-dropout forward + aleatoric uncertainty: 3 outputs per pass -> output_dim = 3L
    y_array <- .mc_dropout_forward(model, x, forward_passes, output_dim = 3L)

    # y_array has dimension: [passes, n_obs, 3], and each obs is [lwr, upr, mean]
    lwr_mat  <- y_array[, , 1]
    upr_mat  <- y_array[, , 2]
    mean_mat <- y_array[, , 3]

    combiner = "variance"

    preds <- if (combiner == "sampling") {
      .combine_uncertainties_sampling(
        lwr_mat  = lwr_mat,
        upr_mat  = upr_mat,
        mean_mat = mean_mat,
        alpha    = alpha,
        inner_samples = inner_samples,
        centerline = mu_det[, 3] # mean is the third dimension
      )
    } else if (combiner == "variance"){
      .combine_uncertainties_variance(
        lwr_mat  = lwr_mat,
        upr_mat  = upr_mat,
        mean_mat = mean_mat,
        alpha    = alpha,
        centerline = mu_det[, 3]
      )
    }
  } else if (pi_method == "aleatoric") {
      lwr <- mu_det[, 1]; upr <- mu_det[, 2]; mu <- mu_det[, 3]
      z_val <- stats::qnorm(1 - alpha / 2)
      width <- pmax(upr - lwr, 0)
      sd_ale <- if (z_val > 0) width / (2 * z_val) else rep(NA_real_, length(width))
      var_ale <- sd_ale^2
      preds <- data.frame(
        lwr = lwr,
        upr = upr,
        var_epistemic = NA_real_,
        var_aleatoric = var_ale,
        var_total = var_ale
      )
    }
    else{
      preds <- data.frame(
        lwr = NA_real_,
        upr = NA_real_,
        var_epistemic = NA_real_,
        var_aleatoric = NA_real_,
        var_total = NA_real_
      )
    }
  preds
}

#' Internal helper: MC Dropout forward sampling
#'
#' @description
#' Run \code{passes} stochastic forward passes with Dropout active at prediction time.
#' Each pass samples a dropout mask and produces predictions, simulating epistemic
#' uncertainty.
#'
#' @param model Fitted Keras model for one smooth term.
#' @param x Input matrix (converted to TensorFlow tensor internally).
#' @param passes Number of stochastic passes (>=2).
#' @param output_dim Expected number of outputs per observation
#'   (e.g., 1 = mean only, 3 = quantile heads (lwr, upr, mean)).
#'
#' @return A numeric array of shape \code{[passes, n_obs, output_dim]}.
#'
#' @keywords internal
.mc_dropout_forward <- function(model, x, passes, output_dim) {
  if (!is.matrix(x)) x <- as.matrix(x)
  n <- nrow(x)
  out <- array(NA_real_, dim = c(passes, n, output_dim))
  x_tf <- tensorflow::tf$convert_to_tensor(x)
  for (b in seq_len(passes)) {
    y <- model(x_tf, training = TRUE) # dropout ON
    if (length(dim(y)) == 1L) y <- tensorflow::tf$expand_dims(y, axis = -1L)
    out[b, , ] <- as.array(y)
  }
  out
}

#' Internal helper: combine epistemic and aleatoric uncertainties via mixture sampling
#'
#' @description
#' Combine uncertainty estimates from multiple MC Dropout passes where each pass
#' produces quantile bounds and a mean. For each observation, samples are drawn from
#' Normal approximations of aleatoric noise across passes, yielding a predictive
#' mixture distribution.
#'
#' @param lwr_mat Matrix \code{[passes, n_obs]} of lower quantile predictions.
#' @param upr_mat Matrix \code{[passes, n_obs]} of upper quantile predictions.
#' @param mean_mat Matrix \code{[passes, n_obs]} of mean predictions.
#' @param alpha Coverage level (default 0.05).
#' @param inner_samples Number of Normal samples per pass/observation.
#' @param centerline Optional vector of deterministic mean predictions (overrides pass-mean).
#'
#' @return A \code{data.frame} with columns:
#'   \itemize{
#'     \item \code{lwr}, \code{upr}: lower/upper predictive interval.
#'     \item \code{var_epistemic}: epistemic variance (across passes).
#'     \item \code{var_aleatoric}: average aleatoric variance.
#'     \item \code{var_total}: sum of epistemic and aleatoric variances.
#'   }
#'
#' @importFrom stats rnorm qnorm
#' @keywords internal
.combine_uncertainties_sampling <- function(lwr_mat, upr_mat, mean_mat,
                                            alpha = 0.05, inner_samples = 50,
                                            centerline = NULL) {
  stopifnot(all(dim(lwr_mat) == dim(upr_mat)),
            all(dim(lwr_mat) == dim(mean_mat)))
  z <- qnorm(1 - alpha/2)
  Tpasses <- nrow(mean_mat); n <- ncol(mean_mat)

  # aleatoric sd per pass/obs
  sd_mat <- pmax((upr_mat - lwr_mat) / (2 * z), 1e-8)

  # sample from the mixture
  # total samples per obs = Tpasses * inner_samples
  # returns [n, T*inner] matrix (but we compute quantiles on the fly to save memory)
  lwr <- numeric(n); upr <- numeric(n); mean_pred <- numeric(n)

  for (i in seq_len(n)) {
    # draw eps for all passes at once
    eps <- matrix(rnorm(Tpasses * inner_samples), nrow = Tpasses)
    y_samps <- mean_mat[, i, drop = TRUE] + sd_mat[, i, drop = TRUE] * eps
    y_samps <- as.vector(y_samps)
    lwr[i]  <- as.numeric(stats::quantile(y_samps, probs = alpha/2, names = FALSE, type = 7))
    upr[i]  <- as.numeric(stats::quantile(y_samps, probs = 1 - alpha/2, names = FALSE, type = 7))
    mean_pred[i] <- if (!is.null(centerline)) centerline[i] else mean(y_samps)
  }

  var_epistemic <- matrixStats::colVars(mean_mat)
  var_aleatoric <- matrixStats::colMeans2(((upr_mat - lwr_mat)/(2*z))^2)

  data.frame(
    lwr = lwr,
    upr = upr,
    var_epistemic = var_epistemic,                 # across passes
    var_aleatoric = var_aleatoric,
    var_total     = var_epistemic + var_aleatoric
  )
}

#' Internal helper: combine epistemic and aleatoric via variance decomposition
#'
#' @description
#' Classical combination of uncertainties without sampling. Assumes the same
#' input shapes as `.combine_uncertainties_sampling`: each argument is a matrix
#' of shape \code{[passes, n_obs]}, where rows index MC-Dropout passes and columns
#' index observations.
#'
#' For each observation (column):
#' - Epistemic variance = variance across passes of the mean head.
#' - Aleatoric variance = average (across passes) of per-pass variance
#'   estimated from quantile width via Normal approximation.
#' - Total variance = epistemic + aleatoric.
#' - Predictive interval = Normal-theory interval around the chosen centerline.
#'
#' @param lwr_mat \code{[passes, n_obs]} lower-quantile predictions per pass.
#' @param upr_mat \code{[passes, n_obs]} upper-quantile predictions per pass.
#' @param mean_mat \code{[passes, n_obs]} mean-head predictions per pass.
#' @param alpha Coverage level (default 0.05).
#' @param centerline Optional numeric vector (length n_obs) of deterministic
#'   mean predictions to use as the PI center. If NULL, uses the across-pass mean.
#'
#' @return data.frame with columns:
#'   - lwr, upr: lower/upper predictive interval (Normal-theory)
#'   - var_epistemic: variance across passes of mean predictions
#'   - var_aleatoric: average per-pass aleatoric variance (from quantile width)
#'   - var_total: sum of epistemic and aleatoric variances
#'
#' @importFrom stats qnorm
#' @keywords internal
.combine_uncertainties_variance <- function(lwr_mat, upr_mat, mean_mat,
                                            alpha = 0.05, centerline = NULL) {
  # ---- shape checks: expect [passes, n_obs] everywhere ----
  if (!is.matrix(lwr_mat))  lwr_mat  <- as.matrix(lwr_mat)
  if (!is.matrix(upr_mat))  upr_mat  <- as.matrix(upr_mat)
  if (!is.matrix(mean_mat)) mean_mat <- as.matrix(mean_mat)

  same_dim <- function(a, b) identical(dim(a), dim(b))
  stopifnot(
    same_dim(lwr_mat, upr_mat),
    same_dim(lwr_mat, mean_mat),
    nrow(mean_mat) >= 2L  # need >= 2 passes for variance
  )

  passes <- nrow(mean_mat)
  n_obs  <- ncol(mean_mat)

  z <- stats::qnorm(1 - alpha / 2)

  # ---- Aleatoric variance from per-pass quantile width ----
  # sd^(b) = (upr - lwr) / (2 z); then var_ale = mean_b sd^(b)^2
  width   <- pmax(upr_mat - lwr_mat, 0)
  sd_mat  <- width / (2 * z)
  var_ale <- matrixStats::colMeans2(sd_mat^2)   # length n_obs

  # ---- Epistemic variance: variance across passes of the mean head ----
  var_epi <- matrixStats::colVars(mean_mat)     # length n_obs

  # ---- Centerline for the PI ----
  mu_hat <- if (!is.null(centerline)) {
    stopifnot(length(centerline) == n_obs)
    as.numeric(centerline)
  } else {
    matrixStats::colMeans2(mean_mat)            # across-pass mean, length n_obs
  }

  # ---- Total variance and Normal-theory interval ----
  var_tot <- var_epi + var_ale
  se_tot  <- sqrt(pmax(var_tot, 0))
  lwr     <- mu_hat - z * se_tot
  upr     <- mu_hat + z * se_tot

  data.frame(
    lwr = lwr,
    upr = upr,
    var_epistemic = var_epi,
    var_aleatoric = var_ale,
    var_total     = var_tot
  )
}

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
#'
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
    }
  } else {
    # If no parametric part, intercept lives in eta0; keep var_param = 0
    eta_param_fit <- rep(ngam$eta0 %||% 0, n)
  }

  # 2) Nonparametric component: joint MC Dropout over all smooths
  # Build (passes x n) matrix of the summed smooth for each pass
  if (!length(np_terms)) {
    # No smooths: SE = sqrt(var_param)
    return(sqrt(pmax(var_param, 0)))
  }

  passes <- max(2L, as.integer(forward_passes))
  eta_np_pass <- matrix(0.0, nrow = passes, ncol = n)  # each row = one pass

  for (tm in np_terms) {
    mdl <- ngam$model[[tm]]
    Xtm <- x[[tm]]; if (is.null(dim(Xtm))) Xtm <- matrix(Xtm, ncol = 1L)

    # Get output dimension & mean head index
    probe <- try(as.matrix(mdl$predict(Xtm, verbose = 0)), silent = TRUE)
    if (inherits(probe, "try-error") || is.null(dim(probe))) {
      # fallback: deterministic mean only, no extra variance
      mu_det <- as.numeric(mdl$predict(Xtm, verbose = 0))
      eta_np_pass <- sweep(eta_np_pass, 2L, mu_det, `+`)
      next
    }
    nout <- ncol(probe)
    mean_col <- if (nout >= 3L) 3L else 1L

    # MC Dropout forward: returns [passes, n, nout] or [passes, n] if nout==1
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

  # 4) Joint epistemic variance from passes (captures cross-term covariance among smooths)
  var_ep_joint <- matrixStats::colVars(eta_full_pass)  # length n

  # 5) Combine with parametric variance (independence assumption)
  var_eta <- var_ep_joint + var_param
  sqrt(pmax(var_eta, 0))
}
