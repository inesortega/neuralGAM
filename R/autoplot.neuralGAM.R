#' @title Autoplot method for \code{neuralGAM} objects (single plot)
#'
#' @description
#' Produce diagnostic and effect plots from a fitted \code{neuralGAM} model.
#' Supported plot types:
#' \itemize{
#'   \item \code{which = "response"}: Fitted response vs. index, with optional
#'         \emph{confidence intervals} (CI) and/or \emph{prediction intervals} (PI).
#'   \item \code{which = "link"}: Linear predictor (link scale) vs. index,
#'         with optional CI (PI are not defined on the link scale).
#'   \item \code{which = "terms"}: Single per-term contribution \eqn{g_j(x_j)} on the link scale.
#'         For continuous covariates: line with optional bands. For factor covariates: boxplots per
#'         level with optional mean \eqn{\pm z \cdot SE} error bars and/or aleatoric ranges.
#'         On the terms panel you can display (i) epistemic CI for the smooth \eqn{g_j} and/or
#'         (ii) \emph{aleatoric residual quantile bands} for partial residuals \eqn{R_j}.
#' }
#'
#' @param object A fitted \code{neuralGAM} object.
#' @param newdata Optional \code{data.frame}/list of covariates. If omitted, training data are used.
#' @param which One of \code{c("response","link","terms")}. Default \code{"response"}.
#' @param interval One of \code{c("none","confidence","prediction","both","aleatoric")}.
#'   Default \code{"confidence"}.
#'   \itemize{
#'     \item \strong{Response scale:} \code{"confidence"} draws epistemic CI for the fitted mean
#'           (SEs on the link mapped via the delta method). \code{"prediction"} draws PIs that reflect
#'           aleatoric noise (and, when the model was trained with combined uncertainty, epistemic too).
#'           \code{"both"} overlays CI and PI.
#'     \item \strong{Link scale:} Only \code{"confidence"} is defined; if \code{interval} requests
#'           \code{"prediction"}, \code{"both"}, or \code{"aleatoric"}, it is coerced to
#'           \code{"confidence"} with a warning.
#'     \item \strong{Terms (per-term effects on the link scale):}
#'           \code{"confidence"} draws epistemic CI for the smooth \eqn{g_j(x_j)} (SE-based).
#'           \code{"aleatoric"} draws \emph{diagnostic} residual quantile bands for partial residuals
#'           \eqn{R_j} (from quantile heads), centered on \eqn{\hat g_j}. \code{"both"} overlays
#'           confidence and aleatoric bands. \code{"prediction"} is not defined for terms and is
#'           coerced to \code{"aleatoric"} with a warning.
#'   }
#' @param level Coverage level for intervals (e.g., \code{0.95}). Default \code{0.95}.
#' @param term Single term name to plot when \code{which = "terms"}.
#' @param rug Logical; if \code{TRUE} (default), add rugs to continuous term plots.
#' @param ... Additional arguments passed to \code{predict.neuralGAM}.
#'
#' @return A single \code{ggplot} object.
#'
#' @details
#' \strong{Uncertainty semantics}
#' \itemize{
#'   \item \strong{CI (epistemic):} Uncertainty about the \emph{fitted mean}. For terms, bands are
#'         \eqn{\hat g_j \pm z \cdot SE(\hat g_j)} on the link scale. For the response, SEs are mapped
#'         to the response scale via the delta method.
#'   \item \strong{PI (aleatoric / combined):} Width reflects data noise (aleatoric) and, when the
#'         model was trained with combined uncertainty, also model uncertainty (epistemic). PIs are
#'         provided \emph{only} on the response scale.
#'   \item \strong{Aleatoric residual bands for terms (diagnostic):} For \code{which="terms"},
#'         quantile heads yield residual quantile bands for \eqn{R_j} centered on \eqn{\hat g_j}.
#'         These are intended for diagnostics and \emph{are not} prediction intervals for \eqn{Y}.
#' }
#'
#' @author Ines Ortega-Fernandez, Marta Sestelo
#'
#' @importFrom ggplot2 autoplot ggplot aes geom_line geom_ribbon geom_rug
#' @importFrom ggplot2 geom_boxplot geom_point geom_errorbar geom_blank
#' @importFrom ggplot2 geom_linerange labs theme_bw scale_fill_manual scale_colour_manual
#' @importFrom stats predict qnorm
#' @method autoplot neuralGAM
#' @export
autoplot.neuralGAM <- function(object,
                               newdata = NULL,
                               which = c("response","link","terms"),
                               interval = c("none","confidence","prediction","both","aleatoric"),
                               level = 0.95,
                               term = NULL,
                               rug = TRUE,
                               ...) {
  stopifnot(inherits(object, "neuralGAM"))
  which    <- match.arg(which)
  interval <- match.arg(interval)

  # enforce expected behavior: Normalize interval semantics by panel
  if (which == "link" && interval %in% c("prediction","both","aleatoric")) {
    warning("Only 'confidence' is defined on the link scale; using 'confidence'.")
    interval <- "confidence"
  }
  if (which == "terms" && interval == "prediction") {
    warning("Prediction intervals are not defined for terms; using 'aleatoric' (residual quantile band).")
    interval <- "aleatoric"
  }

  show_legend <- isTRUE(interval == "both")

  x <- if (is.null(newdata)) object$x else as.data.frame(newdata)

  # helper: keep only rows with finite bands
  .finite_band <- function(df, lower, upper) {
    keep <- is.finite(df[[lower]]) & is.finite(df[[upper]])
    df[keep, , drop = FALSE]
  }

  # ---------------------- RESPONSE SCALE ----------------------
  if (which == "response") {
    if (interval == "none") {
      mu <- predict(object, newdata, type = "response", se.fit = FALSE, ...)
      if (length(mu) == 0L) stop("No data to plot.")
      df <- data.frame(.x = seq_along(mu), fit = as.numeric(mu))
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$fit)) +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Response", x = "Index") +
        ggplot2::theme_bw()
      return(p)
    }

    if (interval == "confidence") {
      res <- predict(object, newdata, type = "response", se.fit = TRUE, ...)
      fit <- as.numeric(res$fit); se <- as.numeric(res$se.fit)
      if (length(fit) == 0L) stop("No data to plot.")
      z <- stats::qnorm(1 - (1 - level)/2)
      df <- data.frame(.x = seq_along(fit),
                       fit = fit,
                       lwr = fit - z*se,
                       upr = fit + z*se)
      df_band <- .finite_band(df, "lwr", "upr")
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$fit)) +
        { if (nrow(df_band) > 0L)
          ggplot2::geom_ribbon(data = df_band, ggplot2::aes(ymin = .data$lwr, ymax = .data$upr), alpha = 0.2)
          else { warning("Confidence band unavailable (missing SEs)."); ggplot2::geom_blank() } } +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Response", x = "Index") +
        ggplot2::theme_bw()
      return(p)
    }

    if (interval == "prediction") {
      df <- predict(object, newdata, type = "response", interval = "prediction", level = level, ...)
      if (!all(c("fit","lwr","upr") %in% names(df))) stop("predict() did not return PI columns.")
      if (nrow(df) == 0L) stop("No data to plot.")
      df$.x <- seq_len(nrow(df))
      df_band <- .finite_band(df, "lwr", "upr")
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$fit)) +
        { if (nrow(df_band) > 0L)
          ggplot2::geom_ribbon(data = df_band, ggplot2::aes(ymin = .data$lwr, ymax = .data$upr), alpha = 0.15)
          else { warning("Prediction band unavailable."); ggplot2::geom_blank() } } +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Response", x = "Index") +
        ggplot2::theme_bw()
      return(p)
    }

    if (interval == "both") {
      df <- predict(object, newdata, type = "response", interval = "both", level = level, ...)
      if (!all(c("fit","lwr_ci","upr_ci","lwr_pi","upr_pi") %in% names(df))) {
        stop("predict() did not return columns for 'both'.")
      }
      if (nrow(df) == 0L) stop("No data to plot.")
      df$.x <- seq_len(nrow(df))
      df_ci <- .finite_band(df, "lwr_ci", "upr_ci")
      df_pi <- .finite_band(df, "lwr_pi", "upr_pi")

      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$fit)) +
        # PI first (underneath)
        { if (nrow(df_pi) > 0L)
          ggplot2::geom_ribbon(
            data = df_pi,
            ggplot2::aes(ymin = .data$lwr_pi, ymax = .data$upr_pi, fill = "Prediction interval"),
            col = "grey80", alpha = 0.15,
            show.legend = show_legend
          )
          else { warning("Prediction band unavailable."); ggplot2::geom_blank() } } +
        # CI on top if interval != both (otherwise just PI)
        { if (nrow(df_ci) > 0L)
          ggplot2::geom_ribbon(
            data = df_ci,
            ggplot2::aes(ymin = .data$lwr_ci, ymax = .data$upr_ci, fill = "Confidence interval"),
            col = "grey60", alpha = 0.20,
            show.legend = show_legend
          )
          else { warning("Confidence band unavailable (missing SEs)."); ggplot2::geom_blank() } } +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Response", x = "Index", fill = "Interval") +
        {if(show_legend){
          ggplot2::scale_fill_manual(
            values = c("Prediction interval" = "grey80", "Confidence interval" = "grey60")
          )
        }} +
        ggplot2::theme_bw()

      return(p)
    }
  }

  # ------------------------ LINK SCALE ------------------------
  if (which == "link") {
    res <- predict(object, newdata, type = "link", se.fit = (interval == "confidence"), ...)
    if (interval == "none") {
      fit <- as.numeric(if (is.list(res)) res$fit else res)
      if (length(fit) == 0L) stop("No data to plot.")
      df <- data.frame(.x = seq_along(fit), fit = fit)
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$fit)) +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Link", x = "Index") +
        ggplot2::theme_bw()
      return(p)
    }
    # CI on link
    fit <- as.numeric(res$fit); se <- as.numeric(res$se.fit)
    if (length(fit) == 0L) stop("No data to plot.")
    z <- stats::qnorm(1 - (1 - level)/2)
    df <- data.frame(.x = seq_along(fit),
                     fit = fit,
                     lwr = fit - z*se,
                     upr = fit + z*se)
    df_band <- .finite_band(df, "lwr", "upr")
    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$fit)) +
      { if (nrow(df_band) > 0L)
        ggplot2::geom_ribbon(data = df_band, ggplot2::aes(ymin = .data$lwr, ymax = .data$upr), alpha = 0.2)
        else { warning("Confidence band unavailable (missing SEs)."); ggplot2::geom_blank() } } +
      ggplot2::geom_line() +
      ggplot2::labs(y = "Link", x = "Index") +
      ggplot2::theme_bw()
    return(p)
  }

  # -------------------------- TERMS (single term, link scale) --------------------------
  if (which == "terms") {
    if (is.null(term) || length(term) != 1L)
      stop("When which='terms', provide exactly one 'term'.")

    all_terms <- c(object$formula$p_terms %||% character(0L),
                   object$formula$np_terms %||% character(0L))
    if (!term %in% all_terms)
      stop(sprintf("Unknown term '%s'. Available terms: %s", term, paste(all_terms, collapse = ", ")))

    want_ci  <- interval %in% c("confidence","both")
    want_ale <- interval %in% c("aleatoric","both")

    pr_terms <- predict(object, newdata,
                        type = "terms", terms = term,
                        se.fit = want_ci,
                        diagnostic_bands = want_ale,  # same internal switch as before
                        ...)

    if (!is.list(pr_terms)) pr_terms <- list(fit = pr_terms)
    term_fit <- as.numeric(pr_terms$fit[, term])
    term_se  <- if (!is.null(pr_terms$se.fit)) pr_terms$se.fit[, term] else NA_real_
    lwrA <- if (!is.null(pr_terms$lwr_ale)) pr_terms$lwr_ale[, term] else NA_real_
    uprA <- if (!is.null(pr_terms$upr_ale)) pr_terms$upr_ale[, term] else NA_real_
    xv   <- (if (is.null(newdata)) object$x else as.data.frame(newdata))[[term]]
    z    <- stats::qnorm(1 - (1 - level)/2)

    if (!is.factor(xv)) {
      df <- data.frame(x = xv, fit = term_fit)
      if (want_ci && any(is.finite(term_se))) {
        df$lwr_ci <- term_fit - z * term_se
        df$upr_ci <- term_fit + z * term_se
      }
      if (want_ale && any(is.finite(lwrA) & is.finite(uprA))) {
        df$lwr_ale <- term_fit + lwrA
        df$upr_ale <- term_fit + uprA
      }
      df <- df[order(df$x), , drop = FALSE]

      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$fit)) +
        { if (!is.null(df$lwr_ale) && !is.null(df$upr_ale)) {
          if (show_legend) {
            ggplot2::geom_ribbon(
              ggplot2::aes(ymin = .data$lwr_ale, ymax = .data$upr_ale,
                           fill = "Residual quantiles (aleatoric)"),
              alpha = 0.20,                         # was col=; for ribbons usually just fill/alpha
              show.legend = TRUE
            )
          } else {
            ggplot2::geom_ribbon(
              ggplot2::aes(ymin = .data$lwr_ale, ymax = .data$upr_ale),
              fill = "grey80", alpha = 0.20, show.legend = FALSE
            )
          }
        }
        } +
        { if (!is.null(df$lwr_ci) && !is.null(df$upr_ci)){
          if (show_legend) {
            ggplot2::geom_ribbon(
              ggplot2::aes(ymin = .data$lwr_ci, ymax = .data$upr_ci,
                           fill = "Confidence band (epistemic)"),
              alpha = 0.20, show.legend = TRUE
            )
          } else {
            ggplot2::geom_ribbon(
              ggplot2::aes(ymin = .data$lwr_ci, ymax = .data$upr_ci),
              fill = "grey60", alpha = 0.20, show.legend = FALSE
            )
          }
        }
        } +
        ggplot2::geom_line() +
        { if (rug) ggplot2::geom_rug(sides = "b", alpha = 0.3) } +
        ggplot2::labs(x = term, y = "Partial effect", fill = "Bands") +
        {if(show_legend){
          ggplot2::scale_fill_manual(values = c(
            "Residual quantiles (aleatoric)" = "grey80",
            "Confidence band (epistemic)"    = "grey60"
          ))
        }} +
        ggplot2::theme_bw()

      if (want_ci  && !any(is.finite(term_se))) warning(sprintf("Confidence band unavailable for term '%s'.", term))
      if (want_ale && !any(is.finite(lwrA) & is.finite(uprA))) warning(sprintf("Aleatoric residual band unavailable for term '%s'.", term))
      return(p)
    }

    if(is.factor(xv)){
      # factor term
      df_box <- data.frame(level = xv, fit = term_fit)
      p <- ggplot2::ggplot(df_box, ggplot2::aes(x = .data$level, y = .data$fit)) +
        ggplot2::geom_boxplot(outlier.shape = NA, alpha = 0.15)

      # Aggregate per level
      ag_fit <- tapply(term_fit, xv, function(z) mean(z, na.rm = TRUE))

      # CI per level (epistemic)
      if (want_ci && any(is.finite(term_se))) {
        ag_se  <- tapply(term_se, xv, function(z) {
          z <- z[is.finite(z)]
          if (!length(z)) NA_real_ else sqrt(mean(z^2))
        })
        lev <- names(ag_fit)
        df_levels <- data.frame(
          level = factor(lev, levels = levels(xv)),
          fit   = as.numeric(ag_fit[lev]),
          se    = as.numeric(ag_se[lev])
        )
        df_levels$lwr_ci <- df_levels$fit - z * df_levels$se
        df_levels$upr_ci <- df_levels$fit + z * df_levels$se
      } else df_levels <- NULL

      # Aleatoric per level: average residual bands, then center on the level mean
      df_ale <- NULL
      if (want_ale && !is.null(pr_terms$lwr_ale) && !is.null(pr_terms$upr_ale)) {
        lwr_v <- pr_terms$lwr_ale[, term]
        upr_v <- pr_terms$upr_ale[, term]
        ok    <- is.finite(lwr_v) & is.finite(upr_v)
        if (any(ok)) {
          ag_lwr <- tapply(lwr_v[ok], xv[ok], function(z) mean(z, na.rm = TRUE))
          ag_upr <- tapply(upr_v[ok], xv[ok], function(z) mean(z, na.rm = TRUE))
          lev <- names(ag_fit)
          df_ale <- data.frame(
            level = factor(lev, levels = levels(xv)),
            fit   = as.numeric(ag_fit[lev]),
            lwr_ale = as.numeric(ag_fit[lev] + ag_lwr[lev]),
            upr_ale = as.numeric(ag_fit[lev] + ag_upr[lev])
          )
        }
      }

      # Draw points at level means
      p <- p + ggplot2::geom_point(
        data = data.frame(level = factor(names(ag_fit), levels = levels(xv)),
                          fit   = as.numeric(ag_fit)),
        ggplot2::aes(x = .data$level, y = .data$fit)
      )

      # Draw CI whiskers if present
      if (want_ci && !is.null(df_levels)) {
        df_levels <- df_levels[is.finite(df_levels$lwr_ci) & is.finite(df_levels$upr_ci), , drop = FALSE]
        if (nrow(df_levels)) {
          p <- p + ggplot2::geom_errorbar(
            data = df_levels,
            ggplot2::aes(x = .data$level, ymin = .data$lwr_ci, ymax = .data$upr_ci),
            width = 0.2
          )
        }
      }

      p <- p + ggplot2::labs(x = term, y = "Partial effect",
                             colour = NULL) +
        ggplot2::theme_bw()

      if (want_ci && (is.null(df_levels) || !nrow(df_levels)))
        warning(sprintf("Confidence band unavailable for term '%s' (missing SEs).", term))
      if (want_ale && (is.null(df_ale) || !nrow(df_ale)))
        warning(sprintf("Aleatoric residual band unavailable for term '%s'.", term))

      return(p)
    }

  }

  stop("Unknown 'which' value.")
}
