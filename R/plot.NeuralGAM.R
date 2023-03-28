#' Visualization of \code{NeuralGAM} object
#'
#' @description Visualization of \code{NeuralGAM} object. Plots the learned partial effects by the NeuralGAM object.
#' @param x a fitted \code{NeuralGAM} object as produced by \code{NeuralGAM()}.
#' @param xlab if supplied, this value will be used as the \code{x} label for all plots
#' @param ylab if supplied, this value will be used as the \code{y} label for all plots
#' @param select allows to plot a set of selected terms. e.g. if you just want to plot the first term,
#' select="X0"
#' @param all.terms if set to \code{TRUE} then the partial effects of the parametric model
#' component are also plotted using \code{termplot} from the stats package.
#' @param \ldots other graphics parameters to pass on to plotting commands.
#' See details for ggplot2::geom_line options
#' @return Returns the partial effects plot.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @importFrom ggplot2 ggplot labs ggplot_gtable ggplot_build aes geom_line
#' @importFrom patchwork wrap_plots plot_layout plot_annotation
#' @export
plot.NeuralGAM <- function(x = object, select=NULL, xlab = NULL, ylab = NULL,
                           all.terms=FALSE, ...) {

  if (!inherits(x, "NeuralGAM")) {
    stop("Argument 'x' must be of class 'NeuralGAM'")
  }

  ngam <- x

  x <- ngam$x
  f <- ngam$partial

  if(!is.null(select) && ncol(x) > 10){
    stop("We can only plot 10 terms at the same time. Please use the select argument
         to choose a subset of terms to plot")
  }

  if (!is.null(select) && !all(select %in% colnames(x))){
    stop(paste("Invalid select argument. Valid options are: ", paste(colnames(x), collapse=",")))
  }

  if (ncol(x) != length(ngam$model)){
    stop("The number of columns in 'ngam$x' must be equal to the number of terms in 'model'")
  }

  if (!is.logical(all.terms)) {
    stop("all.terms must be a logical value.")
  }

  if (!is.null(select)){
    # plot only selected terms
    x <- data.frame(x[,select])
    f <- data.frame(f[,select])
    colnames(x) <- select
    colnames(f) <- select
  }

  plots_list <- vector("list", length = ncol(x))


  # Generate custom labels if xlab or ylab is provided
  if(!is.null(xlab)){
    x_lab <- rep(xlab, ncol(x))
  }
  else{
    x_lab <- colnames(x)
  }
  if(!is.null(ylab)){
    y_lab <- rep(ylab, ncol(x))
  }
  else{
    y_lab <- list()
    for (i in 1:length(x_lab)){
      y_lab[i] <- paste("s(", x_lab[i], ")", sep="")
    }
  }

  for (i in 1:ncol(x)) {
    p <- ggplot2::ggplot() +
      ggplot2::geom_line(ggplot2::aes(x = x[, i], y = f[, i]), ...) +
      ggplot2::labs(x=x_lab[i], y=y_lab[i]) +
      ggplot2::ggtitle(x_lab[i])
    plots_list[[i]] <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(p))
  }

  return(patchwork::wrap_plots(plotlist = plots_list) +
           patchwork::plot_layout(ncol = ncol(x)) +
           patchwork::plot_annotation(title = "Learnt partial effects")
         )
}
