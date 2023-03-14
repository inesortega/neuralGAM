#' Visualization of \code{NeuralGAM} object
#'
#' This function plots the learned partial effects by the NeuralGAM object.
#' @param x \code{NeuralGAM} object.
#' @param y NULL
#' @param ylab NULL
#' @param \ldots Other options.
#' @return Returns the partial effects plot.
#' @author Ines Ortega-Fernandez, Marta Sestelo and Nora M. Villanueva.
#' @examples
#' library(NeuralGAM)
#' data(train)
#' head(train)
#' X_train <- train[c('X0','X1','X2')]
#' y_train <- train$y
#'
#' ngam <- fit_NeuralGAM(num_units = 1024, learning_rate = 0.001, x=X_train,
#'               y = y_train, family = "gaussian", bf_threshold=0.00001,
#'               ls_threshold = 0.1, max_iter_backfitting = 10,
#'               max_iter_ls=10)
#'
#' plot(ngam)
#' @importFrom ggplot2 ggplot labs theme_minimal
#' @importFrom gridExtra grid.arrange
#' @export
plot.NeuralGAM <- function(x = object, y = NULL,
                           ylab = NULL, ...) {

  library(ggplot2)
  library(gridExtra)

  object <- x

  x <- object$x
  y <- object$partial

  plots_list <- list()

  for(i in 1:ncol(x)){

    p <-ggplot2::ggplot() +
      ggplot2::geom_line(aes(x = x[, i], y = y[,i]), lwd = 0.8) +
      ggplot2::labs(x = bquote(X[i]), y = bquote(f(X[i]))) +
      ggplot2::theme_light()

    plots_list[[i]] <- ggplot_gtable(ggplot_build(p))
  }

  return(gridExtra::grid.arrange(grobs = plots_list, ncol=ncol(x)))

}

