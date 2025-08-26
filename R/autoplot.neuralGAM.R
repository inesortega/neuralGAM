#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot
#' @title Autoplot method for \code{neuralGAM} objects. Produces a single plot.
#'
#' @description
#' Produce diagnostic and effect plots from a fitted \code{neuralGAM} model.
#' Supported plot types:
#' \itemize{
#'   \item \code{which = "response"}: Fitted response values vs. observation index,
#'         with optional confidence or prediction intervals.
#'   \item \code{which = "link"}: Linear predictor (link scale) vs. observation index,
#'         with optional confidence intervals.
#'   \item \code{which = "terms"}: Per-term contributions (on the link scale),
#'         displayed in a faceted plot:
#'         \itemize{
#'           \item For continuous covariates: line plot with optional ribbons
#'                 for confidence intervals (CI) and/or prediction intervals (PI).
#'           \item For factor covariates: box plots of per-observation contributions
#'                 by level, with optional mean ± z·SE error bars if available.
#'         }
#' }
#'
#' @param object A fitted \code{neuralGAM} object.
#' @param newdata Optional \code{data.frame}/list of covariates to evaluate.
#'   If omitted, the training data stored in the model are used.
#' @param which One of \code{c("response","link","terms")}. Default \code{"response"}.
#' @param interval One of \code{c("none","confidence","prediction","both")}.
#'   Default \code{"confidence"}.
#'   \itemize{
#'     \item Confidence intervals (CI) are derived from epistemic SEs when the model was trained with `pi_method %in% c("epistemic", "both")`.
#'     \item Prediction intervals (PI) are available only if the model was trained with `pi_method %in% c("aleatoric", "both")`.
#'     \item For \code{which = "terms"} and factor covariates, interval bands are
#'           represented as error bars around group means.
#'   }
#' @param level Coverage level for intervals (e.g., \code{0.95}). Default \code{0.95}.
#' @param term Term name to plot (for \code{which = "terms"}).
#' @param rug Logical; if \code{TRUE} (default), add rugs to continuous term plots.
#' @param ... Additional arguments passed to \code{predict.neuralGAM}.
#'
#' @return
#' A single \code{ggplot} object.
#'
#' @details
#' Intervals:
#' \itemize{
#'   \item \strong{Confidence intervals (CI):} Derived from epistemic SEs
#'         (like in \code{mgcv}) if the model was trained with \code{pi_method = 'epistemic'}.
#'         On the response scale they are propagated through the delta method.
#'   \item \strong{Prediction intervals (PI):} Capture aleatoric uncertainty
#'         if the model was trained with \code{pi_method = 'aleatoric'}. Constructed
#'         by combining per-term bounds on the link scale and applying the
#'         inverse link.
#' }
#'
#' For factor terms, plots display the distribution of per-observation contributions
#' as box plots. If SEs are available, group means are shown with error bars
#' representing \eqn{\text{mean} \pm z \times SE}.
#'
#' Multiple terms are always combined into a single faceted plot (one panel per term).
#'
#' @importFrom ggplot2 autoplot ggplot aes geom_line geom_ribbon geom_rug geom_boxplot
#'   stat_summary geom_errorbar geom_point labs theme_bw facet_wrap
#' @importFrom stats qnorm
#' @importFrom rlang .data
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
#'autoplot(ngam, which = "terms", term = "x1", interval = "both")
#'
#' # single term (factor): x2
#' autoplot(ngam, which = "terms", term = "x2", interval = "confidence")
#'
#' # user arranges multiple terms manually (pseudo-code)
#' p1 <- autoplot(ngam, which = "terms", term = "x1")
#' p2 <- autoplot(ngam, which = "terms", term = "x2")
#' # arrange p1, p2 using your preferred layout tooling, for example using grid.arrange:
#' gridExtra::grid.arrange(grobs = list(p1,p2), ncol = 2)
#' }
#' @method autoplot neuralGAM
#' @export
autoplot.neuralGAM <- function(object,
                     newdata = NULL,
                     which = c("response","link","terms"),
                     interval = c("none","confidence", "prediction","both"),
                     level = 0.95,
                     term = NULL,
                     rug = TRUE,
                     ...) {
  stopifnot(inherits(object, "neuralGAM"))
  which    <- match.arg(which)
  interval <- match.arg(interval)

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
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, .data$fit)) +
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
      p <- ggplot2::ggplot(df, ggplot2::aes(.data$.x, .data$fit)) +
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
      p <- ggplot2::ggplot(df, ggplot2::aes(.data$.x, .data$fit)) +
        { if (nrow(df_band) > 0L)
          ggplot2::geom_ribbon(data = df_band, ggplot2::aes(ymin = .data$lwr, ymax = .data$upr), alpha = 0.15)
          else { warning("Prediction band unavailable."); ggplot2::geom_blank() } } +
        ggplot2::geom_line() +
        ggplot2::labs(y = "Response", x = "Index") +
        ggplot2::theme_bw()
      return(p)
    }
    if (interval == "both"){
      df <- predict(object, newdata, type = "response", interval = "both", level = level, ...)
      if (!all(c("fit","lwr_ci","upr_ci","lwr_pi","upr_pi") %in% names(df))) {
        stop("predict() did not return columns for 'both'.")
      }
      if (nrow(df) == 0L) stop("No data to plot.")
      df$.x <- seq_len(nrow(df))
      df_ci <- .finite_band(df, "lwr_ci", "upr_ci")
      df_pi <- .finite_band(df, "lwr_pi", "upr_pi")
      p <- ggplot2::ggplot(df, ggplot2::aes(.data$.x, .data$fit)) +
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
    if (interval %in% c("prediction","both")) {
      warning("Prediction intervals are not defined on the link scale; plotting CI only.")
    }
    res <- predict(object, newdata, type = "link", se.fit = TRUE, ...)
    fit <- as.numeric(res$fit); se <- as.numeric(res$se.fit)
    if (length(fit) == 0L) stop("No data to plot.")
    z <- stats::qnorm(1 - (1 - level)/2)
    df <- data.frame(.x = seq_along(fit),
                     fit = fit,
                     lwr = fit - z*se,
                     upr = fit + z*se)
    df_band <- .finite_band(df, "lwr", "upr")
    p <- ggplot2::ggplot(df, ggplot2::aes(.data$.x, .data$fit)) +
      { if (nrow(df_band) > 0L)
        ggplot2::geom_ribbon(data = df_band, ggplot2::aes(ymin = .data$lwr, ymax = .data$upr), alpha = 0.2)
        else { warning("Confidence band unavailable (missing SEs)."); ggplot2::geom_blank() } } +
      ggplot2::geom_line() +
      ggplot2::labs(y = "Link", x = "Index") +
      ggplot2::theme_bw()
    return(p)
  }

  # -------------------------- TERMS (single term, link scale) --------------------------
  # enforce exactly one term
  if (which == "terms") {
    if (is.null(term) || length(term) != 1L)
      stop("When which='terms', provide exactly one 'term' (single term per plot).")

    # validate term exists in model
    all_terms <- c(object$formula$p_terms %||% character(0L),
                   object$formula$np_terms %||% character(0L))
    if (!term %in% all_terms)
      stop(sprintf("Unknown term '%s'. Available terms: %s",
                   term, paste(all_terms, collapse = ", ")))

    # get per-term contributions & SEs
    pr_terms <- predict(object, newdata, type = "terms", se.fit = TRUE, ...)
    term_fit <- as.numeric(pr_terms$fit[, term])
    term_se  <- pr_terms$se.fit[, term]
    xv       <- x[[term]]
    z        <- stats::qnorm(1 - (1 - level)/2)

    # PI availability for this term (only when using training cache & build_pi)
    have_pi <- isTRUE(object$build_pi) && is.null(newdata) &&
      term %in% colnames(object$lwr) && term %in% colnames(object$upr)
    lwr_pi <- upr_pi <- NULL
    if (have_pi) {
      lwr_pi <- object$lwr[, term]
      upr_pi <- object$upr[, term]
    }

    if (is.factor(xv)) {
      # -------- factor term: boxplot + mean ± z·SE (if available)
      df_box <- data.frame(level = xv, fit = term_fit)
      p <- ggplot2::ggplot(df_box, ggplot2::aes(x = .data$level, y = .data$fit)) +
        ggplot2::geom_boxplot(outlier.shape = NA, alpha = 0.15)

      # SE bars around group means if SEs are finite
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
        if (nrow(df_levels)) {
          if(!interval %in% c("none"))
          p <- p +
            ggplot2::geom_errorbar(
              data = df_levels,
              ggplot2::aes(x = .data$level, ymin = .data$lwr, ymax = .data$upr),
              width = 0.2
            )
        } else {
          warning(sprintf("Confidence bars unavailable for term '%s' (missing SEs).", term))
        }
      }

      p <- p +
        ggplot2::labs(x = term, y = "Partial effect") +
        ggplot2::theme_bw()
      return(p)
    }

    # -------- continuous term: line + optional ribbons
    df <- data.frame(x = xv, fit = term_fit)
    ord <- order(df$x)
    df <- df[ord, , drop = FALSE]

    # ribbons
    add_ci <- interval %in% c("confidence","both") && any(is.finite(term_se))
    add_pi <- interval %in% c("prediction","both") && have_pi &&
      any(is.finite(lwr_pi)) && any(is.finite(upr_pi))

    if (add_ci) {
      lwr_ci <- term_fit - z * term_se
      upr_ci <- term_fit + z * term_se
      df$lwr_ci <- lwr_ci[ord]; df$upr_ci <- upr_ci[ord]
    }
    if (add_pi) {
      df$lwr_pi <- lwr_pi[ord]; df$upr_pi <- upr_pi[ord]
    }

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$fit)) +
      { if (!is.null(df$lwr_pi) && !is.null(df$upr_pi)) {
        df_pi <- df[is.finite(df$lwr_pi) & is.finite(df$upr_pi), , drop = FALSE]
        if (nrow(df_pi)) ggplot2::geom_ribbon(data = df_pi, ggplot2::aes(ymin = .data$lwr_pi, ymax = .data$upr_pi), alpha = 0.10)
      } } +
      { if (!is.null(df$lwr_ci) && !is.null(df$upr_ci)) {
        df_ci <- df[is.finite(df$lwr_ci) & is.finite(df$upr_ci), , drop = FALSE]
        if (nrow(df_ci)) ggplot2::geom_ribbon(data = df_ci, ggplot2::aes(ymin = .data$lwr_ci, ymax = .data$upr_ci), alpha = 0.20)
      } } +
      ggplot2::geom_line() +
      { if (rug) ggplot2::geom_rug(sides = "b", alpha = 0.3) } +
      ggplot2::labs(x = term, y = "Partial effect") +
      ggplot2::theme_bw()

    # gentle messages when user requested unavailable bands
    if (interval %in% c("confidence","both") && !add_ci)
      warning(sprintf("Confidence band unavailable for term '%s' (missing SEs).", term))
    if (interval %in% c("prediction","both") && !add_pi)
      warning(sprintf("Prediction band unavailable for term '%s'.", term))

    return(p)
  }

  stop("Unknown 'which' value.")
}
