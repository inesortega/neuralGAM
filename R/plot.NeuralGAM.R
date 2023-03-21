#' Visualization of \code{NeuralGAM} object
#'
#' @description Visualization of \code{NeuralGAM} object. Plots the learned partial effects by the NeuralGAM object.
#' @param x \code{NeuralGAM} object.
#' @param y NULL
#' @param ylab NULL
#' @param \ldots Other options.
#' @return Returns the partial effects plot.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @importFrom ggplot2 ggplot labs ggplot_gtable ggplot_build aes geom_line
#' @importFrom patchwork wrap_plots plot_layout plot_annotation
#' @export
plot.NeuralGAM <- function(x = object, y = NULL,
                           ylab = NULL, ...) {
  object <- x

  x <- object$x
  f <- object$partial

  plots_list <- vector("list", length = ncol(x))

  for (i in 1:ncol(x)) {
    p <- ggplot2::ggplot() +
      ggplot2::geom_line(ggplot2::aes(x = x[, i], y = f[, i])) +
      ggplot2::labs(x=colnames(x)[i], y=paste("f(", colnames(x)[i], ")", sep="")) +
      ggplot2::ggtitle(colnames(x)[i])
    plots_list[[i]] <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(p))
  }

  return(patchwork::wrap_plots(plotlist = plots_list) +
           patchwork::plot_layout(ncol = ncol(x)) +
           patchwork::plot_annotation(title = "Learnt partial effects for each covariate in the training set")
         )
}
