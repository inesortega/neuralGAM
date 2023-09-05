#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot

#' Advanced \code{neuralGAM} visualization with ggplot2 library
#' @param object a fitted \code{neuralGAM} object as produced by \code{neuralGAM()}.
#' @param select selects the term to be plotted.
#' @param xlab A title for the \code{x} axis.
#' @param ylab A title for the \code{y} axis.
#' @param \ldots other graphics parameters to pass on to plotting commands.
#' See details for ggplot2::geom_line options
#' @return  A ggplot object, so you can use common features from the ggplot2 package
#' to manipulate the plot.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @importFrom ggplot2 ggplot labs aes geom_line
#' @importFrom ggplot2 guide_axis scale_x_discrete geom_boxplot
#' @importFrom gridExtra grid.arrange
#' @examples \donttest{
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
          ggplot2::labs(x = xlab, y = ylab)
      )
    }

  }
