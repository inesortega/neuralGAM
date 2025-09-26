#' @title Visualization of \code{neuralGAM} object with base graphics
#'
#' @description
#' Visualization of a fitted \code{neuralGAM}. Plots learned partial effects, either as
#' scatter/line plots for continuous covariates or s for factor covariates.
#' Confidence and/or prediction intervals can be added if available.
#'
#' @param x A fitted \code{neuralGAM} object as produced by \code{neuralGAM()}.
#' @param select Character vector of terms to plot. If \code{NULL} (default),
#'   all terms are plotted.
#' @param xlab Optional custom x-axis label(s).
#' @param ylab Optional custom y-axis label(s).
#' @param interval One of \code{c("none","confidence","prediction","both")}.
#'   Default \code{"none"}. Controls whether intervals are plotted.
#' @param level Coverage level for intervals (e.g. \code{0.95}). Default \code{0.95}.
#' @param ... Additional graphical arguments passed to \code{plot()}.
#'
#' @return Produces plots on the current graphics device.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @importFrom graphics  arrows lines points
#' @export
plot.neuralGAM <- function(x, select = NULL,
                           xlab = NULL, ylab = NULL,
                           interval = c("none","confidence","prediction","both"),
                           level = 0.95,
                           ...) {

  if (!inherits(x, "neuralGAM"))
    stop("Argument 'x' must be of class 'neuralGAM'")

  interval <- match.arg(interval)
  ngam <- x

  X <- ngam$x
  f <- ngam$partial

  if (!is.null(select)) {
    if (!all(select %in% colnames(X)))
      stop("Invalid select argument. Valid options are: ",
           paste(colnames(X), collapse = ", "))
  }

  # compute SEs and/or PIs if requested
  se_mat <- lwr_mat <- upr_mat <- NULL
  if (interval %in% c("confidence","both")) {
    pr <- predict(ngam, newdata = X, type = "terms", terms = select, se.fit = TRUE)
    se_mat <- pr$se.fit
  }
  if (interval %in% c("prediction","both") && isTRUE(ngam$build_pi) && is.null(select)) {
    lwr_mat <- ngam$lwr[, colnames(X), drop = FALSE]
    upr_mat <- ngam$upr[, colnames(X), drop = FALSE]
  }

  # Filter after predict, since predict with newdata requires all covariates to be present
  X <- X[, select, drop = FALSE]
  f <- f[, select, drop = FALSE]

  plot_names <- colnames(X)
  z <- stats::qnorm(1 - (1 - level)/2)

  for (i in seq_along(plot_names)) {
    term <- plot_names[i]
    xv <- X[[term]]
    yv <- f[[term]]

    if (is.factor(xv)) {
      # -------- factor term:  + optional mean ± SE
      graphics::boxplot(yv ~ xv, xlab = term, ylab = paste("Partial for", term),
              main = term, ...)

      if (!is.null(se_mat)) {
        means <- tapply(yv, xv, mean, na.rm = TRUE)
        ses   <- tapply(se_mat[, term], xv, function(z) mean(z, na.rm = TRUE))
        ci_lwr <- means - z * ses
        ci_upr <- means + z * ses

        graphics::points(seq_along(means), means, pch = 19, col = "red")
        graphics::arrows(seq_along(means), ci_lwr, seq_along(means), ci_upr,
               angle = 90, code = 3, length = 0.05, col = "red")
      }

    } else {
      # -------- continuous term: scatter + line + intervals
      ord <- order(xv)
      plot(xv, yv, type = "p", xlab = term, ylab = paste("s(", term, ")", sep=""),
           main = term, ...)
      graphics::lines(xv[ord], yv[ord], col = "blue")

      if (!is.null(se_mat)) {
        se <- se_mat[, term]
        lwr_ci <- yv - z * se
        upr_ci <- yv + z * se
        graphics::lines(xv[ord], lwr_ci[ord], col = "red", lty = 2)
        graphics::lines(xv[ord], upr_ci[ord], col = "red", lty = 2)
      }

      if (!is.null(lwr_mat) && !is.null(upr_mat)) {
        graphics::lines(xv[ord], lwr_mat[ord, term], col = "darkgreen", lty = 3)
        graphics::lines(xv[ord], upr_mat[ord, term], col = "darkgreen", lty = 3)
      }
    }

    # pause if multiple plots
    if (i < length(plot_names)) {
      message("Hit <Return> to see next plot: ")
      invisible(readLines(n = 1))
    }
  }
}
