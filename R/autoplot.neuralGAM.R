#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot

#' @title Plot smooth terms from a fitted \code{neuralGAM} model
#'
#' @description
#' Produces a \code{ggplot} visualization of a smooth term from a fitted \code{neuralGAM} model.
#' If the model was trained with prediction intervals (\code{BUILD_PI = TRUE}),
#' the plot automatically includes a shaded ribbon showing the prediction interval
#' between the lower (\code{y_L}) and upper (\code{y_U}) bounds.
#'
#' @param object A fitted \code{neuralGAM} object.
#' @param newdata Optional \code{data.frame}. Values of covariates at which the smooth term will be evaluated.
#' If omitted, predictions are computed for the training data.
#' @param select Character string. Name of the smooth term to plot (e.g., \code{"x1"}).
#' Must match a smooth term in the model.
#' @param ... Additional arguments passed to \code{\link{predict.neuralGAM}} or \code{ggplot2} layers.
#'
#' @return A \code{ggplot} object showing:
#' \itemize{
#'   \item The mean prediction for the smooth term (\code{y_hat}) as a line.
#'   \item If available, a ribbon showing the prediction interval between \code{y_L} and \code{y_U}.
#' }
#'
#' @details
#' The function automatically detects if prediction intervals are present in the model's output
#' (i.e., when \code{BUILD_PI = TRUE}) and adds a ribbon layer to the plot.
#' If intervals are not available, only the smooth function is plotted.
#'
#' The \code{select} argument controls which smooth term is visualized.
#' If you want to visualize all smooth terms at once, consider writing a loop or wrapper
#' that calls \code{autoplot()} for each term.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @importFrom ggplot2 ggplot labs aes geom_line
#' @importFrom ggplot2 guide_axis scale_x_discrete geom_boxplot
#' @importFrom gridExtra grid.arrange
#' @examples \dontrun{
#' n <- 24500
#'
#' seed <- 42
#' set.seed(seed)
#'
#' x1 <- runif(n, -2.5, 2.5)
#' x2 <- runif(n, -2.5, 2.5)
#' x3 <- runif(n, -2.5, 2.5)
#'
#' f1 <- x1 ** 2
#' f2 <- 2 * x2
#' f3 <- sin(x3)
#' f1 <- f1 - mean(f1)
#' f2 <- f2 - mean(f2)
#' f3 <- f3 - mean(f3)
#'
#' eta0 <- 2 + f1 + f2 + f3
#' epsilon <- rnorm(n, 0.25)
#' y <- eta0 + epsilon
#' train <- data.frame(x1, x2, x3, y)
#'
#' library(neuralGAM)
#' ngam <- neuralGAM(y ~ s(x1) + x2 + s(x3), data = train,
#'                  num_units = 1024, family = "gaussian",
#'                  activation = "relu",
#'                  learning_rate = 0.001, bf_threshold = 0.001,
#'                  max_iter_backfitting = 10, max_iter_ls = 10,
#'                  seed = seed
#'                  )
#' autoplot(ngam, select="x1")
#'
#' # add custom title
#' autoplot(ngam, select="x1") + ggplot2::ggtitle("Main Title")
#' # add labels
#' autoplot(ngam, select="x1") + ggplot2::xlab("test") + ggplot2::ylab("my y lab")
#' # plot multiple terms:
#' plots <- lapply(c("x1", "x2", "x3"), function(x) autoplot(ngam, select = x))
#' gridExtra::grid.arrange(grobs = plots, ncol = 3, nrow = 1)
#' }
#' @method autoplot neuralGAM
#' @export
autoplot.neuralGAM <-
  function(object,
           select,
           xlab = NULL,
           ylab = NULL,
           ...) {
    if (!inherits(object, "neuralGAM")) {
      stop("Argument 'object' must be of class 'neuralGAM'")
    }

    ngam <- object

    x <- ngam$x
    f <- ngam$partial

    if (is.null(select)) {
      stop("You must provide a single term to be plotted using the select argument")
    }


    # plot only selected terms
    x <- data.frame(x[, select])
    f <- data.frame(f[, select])
    colnames(x) <- select
    colnames(f) <- select

    term <- select

    # Generate custom labels if xlab or ylab is not provided, else use default colnames
    if (is.null(xlab)) {
      xlab <- colnames(x)
    }

    if (is.null(ylab)) {
      if (term %in% ngam$formula$np_terms) {
        ylab <- paste("s(", term, ")", sep = "")
      }
      else{
        # parametric term:
        if (is.factor(x)) {
          ylab <- paste("Partial for ", term)
        }
        else{
          ylab <- term
        }
      }
    }

    # Robust PI check
    has_PI <- (!is.null(ngam$y_L) && select %in% colnames(ngam$y_L) &&
                 ncol(ngam$y_L) > 0 && !all(is.na(ngam$y_L[[select]]))) &&
      (!is.null(ngam$y_U) && select %in% colnames(ngam$y_U) &&
         ncol(ngam$y_U) > 0 && !all(is.na(ngam$y_U[[select]])))

    # Prepare PI layer if available
    if (has_PI) {
      df_ribbon <- data.frame(
        x = x[, select],
        y_L_term = ngam$y_L[, select],
        y_U_term = ngam$y_U[, select]
      )
      pi_layer <- ggplot2::geom_ribbon(data = df_ribbon, aes(x = x, ymin = y_L_term, ymax = y_U_term),
          fill = "gray", alpha = 0.5)
    } else {
      pi_layer <- NULL
    }

    if (is.factor(x[[term]])) {
      return(
        ggplot2::ggplot() +
          ggplot2::geom_boxplot(ggplot2::aes(x = x[[term]], y = f[[term]]), ...) +
          ggplot2::labs(x = xlab, y = ylab) +
          ggplot2::scale_x_discrete(guide = ggplot2::guide_axis(angle = 45))
      )
    }
    else{
      return(
        ggplot2::ggplot() +
          ggplot2::geom_line(ggplot2::aes(x = x[[term]], y = f[[term]]), ...) +
          pi_layer +
          ggplot2::labs(x = xlab, y = ylab)
      )
    }

  }
