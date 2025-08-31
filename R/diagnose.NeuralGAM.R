#' @title Diagnosis plots to evaluate a fitted \code{neuralGAM} model.
#'
#' @description
#' Produce a 2x2 diagnostic panel for a fitted \code{neuralGAM} model, mirroring
#' the layout of \pkg{gratia}'s \code{appraise()} for \pkg{mgcv} GAMs:
#' (top-left) a QQ plot of residuals with optional simulation envelope,
#' (top-right) a histogram of residuals,
#' (bottom-left) residuals vs linear predictor \eqn{\eta}, and
#' (bottom-right) observed vs fitted values on the response scale.
#'
#' @details
#' The function uses \code{predict.neuralGAM()} to obtain the linear
#' predictor (\code{type = "link"}) and the fitted mean on the response scale
#' (\code{type = "response"}). Residuals are computed internally for supported
#' families; by default we use \emph{deviance residuals}:
#' \itemize{
#'   \item \strong{Gaussian}: \eqn{r_i = y_i - \hat{\mu}_i}.
#'   \item \strong{Binomial}: \eqn{r_i = \mathrm{sign}(y_i-\hat{\mu}_i)\,
#'         \sqrt{2 w_i \{ y_i \log(y_i/\hat{\mu}_i) + (1-y_i)\log[(1-y_i)/(1-\hat{\mu}_i)] \}}}, with optional per-observation weights \eqn{w_i} (e.g., trials for proportions).
#'   \item \strong{Poisson}: \eqn{r_i = \mathrm{sign}(y_i-\hat{\mu}_i)\,
#'         \sqrt{2 w_i \{ y_i \log(y_i/\hat{\mu}_i) - (y_i-\hat{\mu}_i) \}}}, adopting the convention \eqn{y_i \log(y_i/\hat{\mu}_i)=0} when \eqn{y_i=0}.
#' }
#'
#' For Gaussian models, these plots diagnose symmetry, tail behaviour, and
#' mean/variance misfit similar to standard GLM/GAM diagnostics. For non-Gaussian
#' families (Binomial, Poisson), interpret shapes on the \emph{deviance} scale,
#' which is approximately normal under a well-specified model. For discrete data,
#' \emph{randomized quantile (Dunn-Smyth)} residuals are also available and often
#' yield smoother QQ behaviour.
#'
#' \strong{QQ reference methods.}
#' \code{qq_method} controls how theoretical quantiles are generated (as in \pkg{gratia}):
#' \itemize{
#'   \item \code{"uniform"} (default): draw \eqn{U(0,1)} and map through the inverse CDF of the fitted response distribution
#'   at each observation; convert to residuals and average the sorted curves over \code{n_uniform} draws.
#'   Fast and respects the mean-variance relationship.
#'   \item \code{"simulate"}: simulate \code{n_simulate} datasets from the fitted model at the observed covariates, compute residuals, and average the sorted curves; also provides pointwise \code{level} bands on the QQ plot.
#'   \item \code{"normal"}: use standard normal quantiles; a fallback when a suitable RNG or inverse CDF is unavailable.
#' }
#'
#' For Poisson models, include offsets for exposure in the linear predictor
#' (e.g., \code{log(E)}). The QQ methods use \eqn{\hat{\mu}_i} with
#' \code{qpois}/\code{rpois} for \code{"uniform"}/\code{"simulate"}, respectively.
#'
#' @param object A fitted \code{neuralGAM} model.
#' @param data Optional \code{data.frame} for out-of-sample evaluation. If
#'   supplied, \code{response} must name the response column.
#' @param response Character scalar giving the response variable name in
#'   \code{data} (required when \code{data} is provided).
#' @param qq_method Character; one of \code{"uniform"}, \code{"simulate"}, or
#'   \code{"normal"} for the QQ reference. See Details.
#' @param n_uniform Integer; number of \eqn{U(0,1)} replicates for
#'   \code{qq_method = "uniform"}.
#' @param n_simulate Integer; number of simulated datasets for
#'   \code{qq_method = "simulate"} (also controls the QQ bands).
#' @param residual_type One of \code{"deviance"}, \code{"pearson"}, or
#'   \code{"quantile"}. Quantile (Dunn-Smyth) residuals are recommended for
#'   discrete families (binomial/poisson) because they are continuous and
#'   approximately standard normal under the fitted model, improving QQ
#'   diagnostics.
#' @param level Numeric in (0,1); coverage level for the QQ bands when
#'   \code{qq_method = "simulate"}.
#' @param point_col Character; colour for points in scatter/histogram panels.
#' @param point_alpha Numeric in (0,1); point transparency.
#' @param hist_bins Integer; number of bins in the histogram.
#'
#' @return A \pkg{patchwork} object combining four \pkg{ggplot2} plots. You can
#'   print it, add titles/themes, or extract individual panels if needed.
#'
#' @section Dependencies:
#' Requires \pkg{ggplot2} and \pkg{patchwork}.
#'
#' @import patchwork
#' @importFrom ggplot2 ggplot geom_point geom_ribbon geom_abline labs geom_hline geom_smooth geom_histogram
#' @importFrom stats pbinom pnorm ppoints ppois qbinom qpois quantile rbinom rpois runif sd
#' @references
#'
#' Augustin, N.H., Sauleau, E.A., Wood, S.N. (2012). On quantile-quantile plots
#' for generalized linear models. \emph{Computational Statistics & Data Analysis},
#' \strong{56}, 2404-2409. https://doi.org/10.1016/j.csda.2012.01.026
#'
#' Dunn, P.K., Smyth, G.K. (1996). Randomized quantile residuals.
#' \emph{Journal of Computational and Graphical Statistics}, \strong{5}(3), 236-244.
#'
#' @author Ines Ortega-Fernandez, Marta Sestelo
#' @export
diagnose.neuralGAM <- function(object,
                               data = NULL,
                               response = NULL,
                               qq_method = c("uniform", "simulate", "normal"),
                               n_uniform = 1000,
                               n_simulate = 200,
                               residual_type = c("deviance", "pearson", "quantile"),
                               level = 0.95,
                               point_col = "steelblue",
                               point_alpha = 0.5,
                               hist_bins = 30) {

  stopifnot(inherits(object, "neuralGAM"))
  family <- object$family
  qq_method <- match.arg(qq_method)
  residual_type <- match.arg(residual_type)

  # --- predictions (eta and mu) & response y ------------------------------------
  if (is.null(data)) {
    eta <- as.numeric(predict(object, type = "link"))
    mu  <- as.numeric(predict(object, type = "response"))
    if (!is.null(object$y)) y <- as.numeric(object$y)
    else stop("Supply 'data' and 'response' or fit the model with object$y stored.")
  } else {
    stopifnot(!is.null(response), response %in% names(data))
    eta <- as.numeric(predict(object, newdata = data, type = "link"))
    mu  <- as.numeric(predict(object, newdata = data, type = "response"))
    y   <- as.numeric(data[[response]])
  }

  n <- length(y)
  weights <- object$w_train

  # --- Deviance residuals per family -----------------------------------------
  .eps <- 1e-12

  # CDF helpers per family (for quantile residuals)
  cdf_fun <- switch(
    family,
    "gaussian" = function(yv, muv, disp, wv) stats::pnorm(yv, mean = muv, sd = disp),
    "binomial" = function(yv, muv, disp, wv) {
      size <- ifelse(is.null(weights), 1, round(wv))
      pbinom(round(yv * size), size = size, prob = pmin(pmax(muv, .eps), 1 - .eps))
    },
    "poisson" = function(yv, muv, disp, wv) stats::ppois(round(yv), lambda = pmax(muv, .eps))
  )

  # dispersion estimate for Gaussian only (used by gaussian cdf/quantile)
  sigma_hat <- if (family == "gaussian") sqrt(mean((y - mu)^2)) else NA_real_

  # Deviance residuals (as before)
  dev_resid_fun <- switch(
    family,
    "gaussian" = function(yv, muv, wv) yv - muv,
    "binomial" = function(yv, muv, wv) {
      muv <- pmin(pmax(muv, .eps), 1 - .eps)
      yv  <- pmin(pmax(yv,  .eps), 1 - .eps)
      d <- 2 * wv * (yv * log(yv / muv) + (1 - yv) * log((1 - yv) / (1 - muv)))
      sign(yv - muv) * sqrt(pmax(d, 0))
    },
    "poisson" = function(yv, muv, wv) {
      muv <- pmax(muv, .eps)
      term <- ifelse(yv == 0, 0, yv * log(yv / muv))
      d <- 2 * wv * (term - (yv - muv))
      sign(yv - muv) * sqrt(pmax(d, 0))
    }
  )

  # Pearson residuals (optional)
  pearson_fun <- switch(
    family,
    "gaussian" = function(yv, muv, wv) (yv - muv) / (stats::sd(y - mu) + .eps),
    "binomial" = function(yv, muv, wv) (yv - muv) / sqrt(pmax(muv * (1 - muv) / pmax(wv,1), .eps)),
    "poisson"  = function(yv, muv, wv) (yv - muv) / sqrt(pmax(muv / pmax(wv,1), .eps))
  )

  # Randomized quantile residuals (Dunn-Smyth)
  quantile_resid <- function(yv, muv, wv) {
    if (family == "gaussian") {
      # continuous case: exact
      qnorm(stats::pnorm(yv, mean = muv, sd = sigma_hat))
    } else {
      # discrete: U ~ (F(y-), F(y)] then z = Phi^{-1}(U)
      Fy   <- cdf_fun(yv, muv, sigma_hat, wv)
      Fy_1 <- cdf_fun(yv - 1, muv, sigma_hat, wv)  # for binomial: y*size - 1, but round handles
      u <- stats::runif(length(yv), pmax(Fy_1, 0), pmin(Fy, 1))
      qnorm(pmin(pmax(u, .Machine$double.eps), 1 - .Machine$double.eps))
    }
  }

  # Choose residuals
  dres <- switch(residual_type,
                 "deviance" = dev_resid_fun(y, mu, weights),
                 "pearson"  = pearson_fun(y, mu, weights),
                 "quantile" = quantile_resid(y, mu, weights)
  )

  # --- Distribution helpers for QQ methods -----------------------------------
  if (family == "gaussian") {
    sigma_hat <- sqrt(mean((y - mu)^2))
    qfun <- function(p, i) qnorm(p, mean = mu[i], sd = sigma_hat)
    rfun <- function() rnorm(n, mean = mu, sd = sigma_hat)
  } else if (family == "binomial") {
    size <- if (!is.null(weights)) round(weights) else rep(1, n)
    pmu  <- pmin(pmax(mu, .eps), 1 - .eps)
    qfun <- function(p, i) {
      qi <- stats::qbinom(p, size = size[i], prob = pmu[i])
      if (!is.null(weights)) qi / size[i] else qi
    }
    rfun <- function() {
      ri <- stats::rbinom(n, size = size, prob = pmu)
      if (!is.null(weights)) ri / size else ri
    }
  } else if (family == "poisson") {
    # mu already contains offsets/exposures via the fitted model (log link typical)
    qfun <- function(p, i) stats::qpois(p, lambda = mu[i])
    rfun <- function() stats::rpois(n, lambda = mu)
  }

  # --- QQ reference construction ---------------------------------------------
  make_qq <- function() {
    ord  <- order(dres)
    samp <- dres[ord]

    if (qq_method == "normal") {
      theo <- qnorm(stats::ppoints(n))
      return(data.frame(theo = theo, samp = samp))
    }

    if (qq_method == "uniform") {
      ref_mat <- matrix(NA_real_, nrow = n, ncol = n_uniform)
      for (k in seq_len(n_uniform)) {
        u    <- stats::runif(n)
        yref <- vapply(seq_len(n), function(i) qfun(u[i], i), numeric(1))
        rref <- dev_resid_fun(yref, mu, weights)
        ref_mat[, k] <- sort(rref)
      }
      theo <- rowMeans(ref_mat)
      return(data.frame(theo = theo, samp = samp))
    }

    if (qq_method == "simulate") {
      ref_mat <- matrix(NA_real_, nrow = n, ncol = n_simulate)
      for (k in seq_len(n_simulate)) {
        ysim <- rfun()
        rref <- dev_resid_fun(ysim, mu, weights)
        ref_mat[, k] <- sort(rref)
      }
      lo_prob <- (1 - level) / 2
      hi_prob <- 1 - lo_prob
      theo <- rowMeans(ref_mat)
      lo   <- apply(ref_mat, 1, stats::quantile, probs = lo_prob)
      hi   <- apply(ref_mat, 1, stats::quantile, probs = hi_prob)
      return(data.frame(theo = theo, samp = samp, lo = lo, hi = hi))
    }
  }
  qq_df <- make_qq()

  # --- Plots ------------------------------------------------------------------

  p1 <- ggplot2::ggplot(qq_df, aes(x = .data$theo, y = .data$samp)) +
    ggplot2::geom_point(alpha = point_alpha) +
    {
      if ("lo" %in% names(qq_df))
        ggplot2::geom_ribbon(data = qq_df, aes(x = .data$theo, ymin = .data$lo, ymax = .data$hi),
                    alpha = 0.15, inherit.aes = FALSE)
      else NULL
    } +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2) +
    ggplot2::labs(x = "Reference quantiles", y = "Sample deviance residuals",
         title = sprintf("Q-Q plot of deviance residuals (%s)", qq_method))

  p2 <- ggplot2::ggplot(data.frame(eta = eta, dres = dres), aes(x = .data$eta, y = .data$dres)) +
    ggplot2::geom_hline(yintercept = 0, linetype = 2) +
    ggplot2::geom_point(alpha = point_alpha, color = point_col) +
    ggplot2::geom_smooth(method = "loess", formula = y ~ x, se = FALSE) +
    ggplot2::labs(x = "Linear predictor (eta)", y = "Deviance residuals",
         title = "Residuals vs linear predictor",
         subtitle = sprintf("Family: %s", family))

  p3 <- ggplot2::ggplot(data.frame(dres = dres), aes(x = .data$dres)) +
    ggplot2::geom_histogram(bins = hist_bins, fill = point_col, alpha = 0.7) +
    ggplot2::labs(x = "Deviance residuals", y = "Count",
         title = "Histogram of deviance residuals")

  p4 <- ggplot2::ggplot(data.frame(mu = mu, y = y), aes(x = .data$mu, y = .data$y)) +
    ggplot2::geom_point(alpha = point_alpha, color = point_col) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2) +
    ggplot2::geom_smooth(method = "loess", formula = y ~ x, se = FALSE) +
    ggplot2::labs(x = "Fitted values (mu)", y = "Observed (y)",
         title = "Observed vs fitted")

  (p1 | p3) / (p2 | p4)
}
