#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot

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
#'         For continuous covariates: line with optional CI ribbon (SE-based).
#'         For factor covariates: boxplots per level with optional mean \eqn{\pm z \cdot SE} error bars.
#' }
#'
#' @param object A fitted \code{neuralGAM} object.
#' @param newdata Optional \code{data.frame}/list of covariates. If omitted, training data are used.
#' @param which One of \code{c("response","link","terms")}. Default \code{"response"}.
#' @param interval One of \code{c("none","confidence","prediction","both")}. Default \code{"confidence"}.
#'   \itemize{
#'     \item \strong{Response scale:} CI reflect epistemic uncertainty about the mean (from SEs;
#'           delta-method on the response). PI reflect aleatoric (and, when \code{pi_method="both"},
#'           combined) uncertainty for a \emph{new observation}.
#'     \item \strong{Link scale:} CI only (PI not defined). If \code{interval} requests PI, it is
#'           coerced to \code{"confidence"} with a warning.
#'     \item \strong{Terms:} \emph{CI only} for the smooth \eqn{g_j(x_j)} (SE-based, epistemic).
#'           If \code{interval} requests PI, it is coerced to \code{"confidence"} with a warning.
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
#'   \item \strong{CI (epistemic)}: uncertainty about the \emph{fitted mean}.
#'         For terms, bands are \eqn{\hat g_j \pm z \cdot SE(\hat g_j)} on the link scale.
#'         For response, SEs are mapped via the delta method.
#'   \item \strong{PI (aleatoric / both)}: width reflects data noise (and model uncertainty
#'         when \code{pi_method="both"}). PIs are provided \emph{only} on the response scale.
#' }
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 500
#' x1 <- runif(n, -2, 2)
#' x2 <- factor(sample(letters[1:3], n, TRUE))
#' x3 <- runif(n, -2, 2)
#'
#' f1 <- sin(x1)
#' f2 <- ifelse(x2 == "a", 1, ifelse(x2 == "b", -1, 0))
#' f3 <- x3^2
#' y <- 2 + f1 + f2 + f3 + rnorm(n, sd = 0.3)
#' dat <- data.frame(x1, x2, x3, y)
#'
#' library(neuralGAM)
#' ngam <- neuralGAM(y ~ s(x1) + x2 + s(x3),
#'                  data = dat, num_units = 128,
#'                  family = "gaussian",
#'                  pi_method = "both",
#'                  seed = 1)
#'
#' # response with PI
#'autoplot(ngam, which = "response", interval = "prediction")
#'
#' # link with CI
#'autoplot(ngam, which = "link", interval = "confidence")
#'
#' # single term (continuous): x1
#'autoplot(ngam, which = "terms", term = "x1", interval = "confidence")
#'
#' # single term (factor): x2
#' autoplot(ngam, which = "terms", term = "x2", interval = "confidence")
#'
#' # user arranges multiple terms manually
#' p1 <- autoplot(ngam, which = "terms", term = "x1")
#' p2 <- autoplot(ngam, which = "terms", term = "x2")
#' # arrange p1, p2 using your preferred layout tooling, such as grid.arrange
#' }
#' @method autoplot neuralGAM
#' @export
autoplot.neuralGAM <- function(object,
                               newdata = NULL,
                               which = c("response","link","terms"),
                               interval = c("none","confidence","prediction","both"),
                               level = 0.95,
                               term = NULL,
                               rug = TRUE,
                               ...) {
  stopifnot(inherits(object, "neuralGAM"))
  which    <- match.arg(which)
  interval <- match.arg(interval)

  # enforce expected behavior:
  if (which == "link"  && interval %in% c("prediction","both")) {
    warning("Prediction intervals are not defined on the link scale; using 'confidence' instead.")
    interval <- "confidence"
  }
  if (which == "terms" && interval %in% c("prediction","both")) {
    warning("Prediction intervals are not defined for term effects; using 'confidence' instead.")
    interval <- "confidence"
  }

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
        { if (nrow(df_pi) > 0L)
          ggplot2::geom_ribbon(data = df_pi, ggplot2::aes(ymin = .data$lwr_pi, ymax = .data$upr_pi), alpha = 0.10)
          else { warning("Prediction band unavailable."); ggplot2::geom_blank() } } +
        { if (nrow(df_ci) > 0L)
          ggplot2::geom_ribbon(data = df_ci, ggplot2::aes(ymin = .data$lwr_ci, ymax = .data$upr_ci), alpha = 0.20)
          else { warning("Confidence band unavailable (missing SEs)."); ggplot2::geom_blank() } } +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Response", x = "Index") +
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
      stop(sprintf("Unknown term '%s'. Available terms: %s",
                   term, paste(all_terms, collapse = ", ")))

    pr_terms <- predict(object, newdata, type = "terms", se.fit = TRUE, ...)
    term_fit <- as.numeric(pr_terms$fit[, term])
    term_se  <- pr_terms$se.fit[, term]
    xv       <- x[[term]]
    z        <- stats::qnorm(1 - (1 - level)/2)

    if (is.factor(xv)) {
      df_box <- data.frame(level = xv, fit = term_fit)
      p <- ggplot2::ggplot(df_box, ggplot2::aes(x = .data$level, y = .data$fit)) +
        ggplot2::geom_boxplot(outlier.shape = NA, alpha = 0.15)

      if (any(is.finite(term_se))) {
        ag_fit <- tapply(term_fit, xv, function(z) mean(z, na.rm = TRUE))
        ag_se  <- tapply(term_se,  xv, function(z) {
          z <- z[is.finite(z)]
          if (!length(z)) NA_real_ else sqrt(mean(z^2))  # mgcv-style aggregation
        })
        lev <- names(ag_fit)
        df_levels <- data.frame(
          level = factor(lev, levels = levels(xv)),
          fit   = as.numeric(ag_fit[lev]),
          se    = as.numeric(ag_se[lev])
        )
        df_levels$lwr <- df_levels$fit - z * df_levels$se
        df_levels$upr <- df_levels$fit + z * df_levels$se
        df_levels <- df_levels[is.finite(df_levels$lwr) & is.finite(df_levels$upr), , drop = FALSE]

        p <- p + ggplot2::geom_point(
          data = df_levels,
          ggplot2::aes(x = .data$level, y = .data$fit)
        )
        if (nrow(df_levels) && interval == "confidence") {
          p <- p + ggplot2::geom_errorbar(
            data = df_levels,
            ggplot2::aes(x = .data$level, ymin = .data$lwr, ymax = .data$upr),
            width = 0.2
          )
        }
      }

      p <- p + ggplot2::labs(x = term, y = "Partial effect") + ggplot2::theme_bw()
      return(p)
    }

    # continuous term: CI ribbon (if requested) + line
    df <- data.frame(x = xv, fit = term_fit)
    ord <- order(df$x)
    df <- df[ord, , drop = FALSE]

    if (interval == "confidence" && any(is.finite(term_se))) {
      lwr_ci <- term_fit - z * term_se
      upr_ci <- term_fit + z * term_se
      df$lwr_ci <- lwr_ci[ord]; df$upr_ci <- upr_ci[ord]
    }

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$fit)) +
      { if (!is.null(df$lwr_ci) && !is.null(df$upr_ci)) {
        df_ci <- df[is.finite(df$lwr_ci) & is.finite(df$upr_ci), , drop = FALSE]
        if (nrow(df_ci)) ggplot2::geom_ribbon(data = df_ci,
                                              ggplot2::aes(ymin = .data$lwr_ci, ymax = .data$upr_ci),
                                              alpha = 0.20)
      } } +
      ggplot2::geom_line() +
      { if (rug) ggplot2::geom_rug(sides = "b", alpha = 0.3) } +
      ggplot2::labs(x = term, y = "Partial effect") +
      ggplot2::theme_bw()

    if (interval == "confidence" && !any(is.finite(term_se)))
      warning(sprintf("Confidence band unavailable for term '%s' (missing SEs).", term))

    return(p)
  }

  stop("Unknown 'which' value.")
}
