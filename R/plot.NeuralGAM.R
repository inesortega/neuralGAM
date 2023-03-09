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
#' y_train <- train['y']
#'
#' ngam <- fit_NeuralGAM(num_units = 1024, learning_rate = 0.001, x=X_train,
#'               y = y_train, family = "gaussian", bf_threshold=0.00001,
#'               ls_threshold = 0.1, max_iter_backfitting = 10,
#'               max_iter_ls=10)
#'
#' plot(ngam)
#' @import ggplot2
#' @export
plot.NeuralGAM <- function(x = object, y = NULL,
                           ylab = NULL, ...) {
  library(ggplot2)
  object = x

  x = object$x
  y = object$partial

  for(i in 1:ncol(x)) {       # for-loop over columns

    ggplot() +
      geom_point(aes(x = x[, i], y = partial[,i]), col = "blue", size = 2)
      labs(x = paste0("x", i), y = "f(x)") +
      theme_minimal()
  }

}
