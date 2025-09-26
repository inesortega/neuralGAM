#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot
#' @title Autoplot method for \code{neuralGAM} objects (epistemic-only)
#'
#' @description
#' Produce effect/diagnostic plots from a fitted \code{neuralGAM} model.
#' Supported panels:
#' \itemize{
#'   \item \code{which = "response"}: fitted response vs. index, with optional
#'         epistemic \emph{confidence intervals} (CI).
#'   \item \code{which = "link"}: linear predictor (link scale) vs. index,
#'         with optional CI.
#'   \item \code{which = "terms"}: single per-term contribution \eqn{g_j(x_j)} on the link scale,
#'         with optional CI band for the smooth (epistemic).
#' }
#'
#' @param object A fitted \code{neuralGAM} object.
#' @param newdata Optional \code{data.frame}/list of covariates. If omitted, training data are used.
#' @param which One of \code{c("response","link","terms")}. Default \code{"response"}.
#' @param interval One of \code{c("none","confidence")}. Default \code{"confidence"}.
#' @param level Coverage level for confidence intervals (e.g., \code{0.95}). Default \code{0.95}.
#' @param forward_passes Integer. Number of MC-dropout forward passes used when
#'   \code{uncertainty_method \%in\% c("epistemic","both")}.
#' @param term Single term name to plot when \code{which = "terms"}.
#' @param rug Logical; if \code{TRUE} (default), add rugs to continuous term plots.
#' @param ... Additional arguments passed to \code{predict.neuralGAM}.
#'
#' @return A single \code{ggplot} object.
#'
#' @details
#' \strong{Uncertainty semantics (epistemic only)}
#' \itemize{
#'   \item \strong{CI:} Uncertainty about the fitted mean.
#'   \item For the response, SEs are mapped via the delta method;
#'   \item For terms, bands are obtained as \eqn{\hat g_j \pm z \cdot SE(\hat g_j)} on the link scale.
#' }
#' @examples \dontrun{
#'
#' library(neuralGAM)
#' dat <- sim_neuralGAM_data()
#' train <- dat$train
#' test  <- dat$test
#'
#' ngam <- neuralGAM(
#'   y ~ s(x1) + x2 + s(x3),
#'   data = train, family = "gaussian", num_units = 128,
#'   uncertainty_method = "epistemic", forward_passes = 10
#' )
#' ## --- Autoplot (epistemic-only) ---
#' # Per-term effect with CI band
#' autoplot(ngam, which = "terms", term = "x1", interval = "confidence")  +
#'   ggplot2::xlab("x1") + ggplot2::ylab("Partial effect")
#'
#' # Request a different number of forward passes or CI level:
#' autoplot(ngam, which = "terms", term = "x1", interval = "confidence",
#' forward_passes = 15, level = 0.7)

#' # Response panel with CI
#' autoplot(ngam, which = "response", interval = "confidence")
#'
#' # Link panel with CI
#' autoplot(ngam, which = "link", interval = "confidence")  +
#'   ggplot2::ggtitle("Main Title")
#'
#' }
#' @importFrom ggplot2 ggplot aes geom_line geom_ribbon geom_rug geom_boxplot
#' @importFrom ggplot2 geom_point geom_errorbar labs theme_bw
#' @importFrom stats predict qnorm
#' @author Ines Ortega-Fernandez, Marta Sestelo
#' @method autoplot neuralGAM
#' @export
autoplot.neuralGAM <- function(object,
                               newdata = NULL,
                               which = c("response","link","terms"),
                               interval = c("none","confidence"),
                               level = 0.95,
                               forward_passes = 150,
                               term = NULL,
                               rug = TRUE,
                               ...) {
  stopifnot(inherits(object, "neuralGAM"))
  which    <- match.arg(which)
  interval <- match.arg(interval)

  xdat <- if (is.null(newdata)) object$x else as.data.frame(newdata)

  # small helper
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
    } else { # confidence
      df <- predict(object, newdata, type = "response", interval = "confidence", level = level, forward_passes = forward_passes, ...)
      if (!all(c("fit","lwr","upr") %in% names(df))) stop("predict() did not return CI columns.")
      if (nrow(df) == 0L) stop("No data to plot.")
      df$.x <- seq_len(nrow(df))
      df_band <- .finite_band(df, "lwr", "upr")
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$fit)) +
        { if (nrow(df_band) > 0L)
          ggplot2::geom_ribbon(data = df_band,
                               ggplot2::aes(ymin = .data$lwr, ymax = .data$upr),
                               alpha = 0.2) else ggplot2::geom_blank() } +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Response", x = "Index") +
        ggplot2::theme_bw()
      return(p)
    }
  }

  # ------------------------ LINK SCALE ------------------------
  if (which == "link") {
    if (interval == "none") {
      fit <- predict(object, newdata, type = "link", se.fit = FALSE, ...)
      if (length(fit) == 0L) stop("No data to plot.")
      df <- data.frame(.x = seq_along(fit), fit = as.numeric(fit))
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$fit)) +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Link", x = "Index") +
        ggplot2::theme_bw()
      return(p)
    } else { # confidence
      df <- predict(object, newdata, type = "link", interval = "confidence", level = level, forward_passes = forward_passes, ...)
      if (!all(c("fit","lwr","upr") %in% names(df))) {
        # fall back to se.fit if CI df not returned for some reason
        res <- predict(object, newdata, type = "link", se.fit = TRUE, level = level, forward_passes = forward_passes, ...)
        fit <- as.numeric(res$fit); se <- as.numeric(res$se.fit)
        z <- stats::qnorm(1 - (1 - level)/2)
        df <- data.frame(.x = seq_along(fit), fit = fit,
                         lwr = fit - z*se, upr = fit + z*se)
      } else {
        df$.x <- seq_len(nrow(df))
      }
      df_band <- .finite_band(df, "lwr", "upr")
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$fit)) +
        { if (nrow(df_band) > 0L)
          ggplot2::geom_ribbon(data = df_band,
                               ggplot2::aes(ymin = .data$lwr, ymax = .data$upr),
                               alpha = 0.2) else ggplot2::geom_blank() } +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Link", x = "Index") +
        ggplot2::theme_bw()
      return(p)
    }
  }

  # -------------------------- TERMS (single term, link scale) --------------------------
  if (which == "terms") {
    if (is.null(term) || length(term) != 1L)
      stop("When which = 'terms', provide exactly one 'term'.")

    all_terms <- c(object$formula$p_terms %||% character(0L),
                   object$formula$np_terms %||% character(0L))
    if (!term %in% all_terms)
      stop(sprintf("Unknown term '%s'. Available terms: %s", term, paste(all_terms, collapse = ", ")))

    want_ci <- interval == "confidence"

    # Ask predict() for per-term CIs when requested (it returns matrices)
    pr <- if (want_ci) {
      predict(object, newdata, type = "terms", terms = term,
              interval = "confidence", level = level, forward_passes = forward_passes, ...)
    } else {
      predict(object, newdata, type = "terms", terms = term, se.fit = FALSE, ...)
    }
    if (!is.list(pr)) pr <- list(fit = pr)

    term_fit <- as.numeric(pr$fit[, term])
    term_se  <- if (!is.null(pr$se.fit)) pr$se.fit[, term] else NULL
    term_lwr <- if (!is.null(pr$lwr))    pr$lwr[, term]    else NULL
    term_upr <- if (!is.null(pr$upr))    pr$upr[, term]    else NULL

    xv <- xdat[[term]]

    if (!is.factor(xv)) {
      df <- data.frame(x = xv, fit = term_fit)
      if (want_ci && !is.null(term_lwr) && !is.null(term_upr)) {
        df$lwr <- term_lwr
        df$upr <- term_upr
      }
      df <- df[order(df$x), , drop = FALSE]

      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$fit)) +
        { if (want_ci && all(c("lwr","upr") %in% names(df))) {
          df_band <- df[is.finite(df$lwr) & is.finite(df$upr), , drop = FALSE]
          if (nrow(df_band) > 0L)
            ggplot2::geom_ribbon(data = df_band,
                                 ggplot2::aes(ymin = .data$lwr, ymax = .data$upr),
                                 alpha = 0.2)
          else ggplot2::geom_blank()
        } else ggplot2::geom_blank() } +
        ggplot2::geom_line() +
        { if (rug) ggplot2::geom_rug(sides = "b", alpha = 0.3) } +
        ggplot2::labs(x = term, y = "Partial effect") +
        ggplot2::theme_bw()
      return(p)
    }

    # factor term: show boxplot per level; overlay level means and CI if available
    df_box <- data.frame(level = xv, fit = term_fit)
    p <- ggplot2::ggplot(df_box, ggplot2::aes(x = .data$level, y = .data$fit)) +
      ggplot2::geom_boxplot(outlier.shape = NA, alpha = 0.15)

    # level means
    ag_fit <- tapply(term_fit, xv, function(z) mean(z, na.rm = TRUE))
    p <- p + ggplot2::geom_point(
      data = data.frame(level = factor(names(ag_fit), levels = levels(xv)),
                        fit   = as.numeric(ag_fit)),
      ggplot2::aes(x = .data$level, y = .data$fit)
    )

    # CI per level (aggregate SE as sqrt(mean(se^2)))
    if (want_ci && !is.null(term_se)) {
      ag_se <- tapply(term_se, xv, function(z) {
        z <- z[is.finite(z)]
        if (!length(z)) NA_real_ else sqrt(mean(z^2))
      })
      z <- stats::qnorm(1 - (1 - level)/2)
      lev <- names(ag_fit)
      df_levels <- data.frame(
        level = factor(lev, levels = levels(xv)),
        fit   = as.numeric(ag_fit[lev]),
        se    = as.numeric(ag_se[lev])
      )
      df_levels$lwr <- df_levels$fit - z * df_levels$se
      df_levels$upr <- df_levels$fit + z * df_levels$se
      df_levels <- df_levels[is.finite(df_levels$lwr) & is.finite(df_levels$upr), , drop = FALSE]
      if (nrow(df_levels))
        p <- p + ggplot2::geom_errorbar(
          data = df_levels,
          ggplot2::aes(x = .data$level, ymin = .data$lwr, ymax = .data$upr),
          width = 0.2
        )
    }

    p <- p + ggplot2::labs(x = term, y = "Partial effect") +
      ggplot2::theme_bw()
    return(p)
  }

  stop("Unknown 'which' value.")
}
