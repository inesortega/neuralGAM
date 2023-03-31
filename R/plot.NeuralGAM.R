#' Visualization of \code{NeuralGAM} object
#'
#' @description Visualization of \code{NeuralGAM} object. Plots the learned partial effects by the NeuralGAM object.
#' @param x a fitted \code{NeuralGAM} object as produced by \code{NeuralGAM()}.
#' @param select allows to plot a set of selected terms. e.g. if you just want to plot the first term,
#' select="X0"
#' @param xlab if supplied, this value will be used as the \code{x} label for all plots
#' @param ylab if supplied, this value will be used as the \code{y} label for all plots
#' @param \ldots other graphics parameters to pass on to plotting commands.
#' See details for ggplot2::geom_line options
#' @return Returns the partial effects plot.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @importFrom ggplot2 ggplot labs ggplot_gtable ggplot_build aes geom_line
#' @importFrom patchwork wrap_plots plot_layout plot_annotation
#' @export
#' @examples
#' #' library(NeuralGAM)
#' data(train)
#'
#' ngam <- NeuralGAM( y ~ X1 + s(X0) + s(X2), data = train,
#' num_units = 1024, family = "gaussian",
#' learning_rate = 0.001, bf_threshold = 0.001,
#' max_iter_backfitting = 10, max_iter_ls = 10
#' )
#'
#' plot(ngam)
#'
#' # Plot only a given term
#'
#' plot(ngam, select="X1")
#'
plot.NeuralGAM <- function(x = object, select=NULL, xlab = NULL, ylab = NULL,title = NULL, ...) {

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

  if (!is.null(select)){
    # plot only selected terms
    x <- data.frame(x[,select])
    f <- data.frame(f[,select])
    colnames(x) <- select
    colnames(f) <- select
  }

  plots_list <- (vector("list", length = ncol(x)))


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
      if(x_lab[i] %in% ngam$formula$np_terms){
        y_lab[i] <- paste("s(", x_lab[i], ")", sep="")
      }
      else{ ## todo take into account factor terms!
        if(is.factor(x_lab[i])){
          paste("Partial for ", x_lab[i])
        }
        y_lab[i] <- x_lab[i]
      }
    }
  }

  for (i in 1:ncol(x)) {

    term <- colnames(x)[[i]]

    if(is.factor(x[[term]])){
      p <- ggplot2::ggplot() +
        ggplot2::geom_boxplot(ggplot2::aes(x = x[, i], y = f[, i]), ...) +
        ggplot2::labs(x=x_lab[i], y=y_lab[i]) +
        ggplot2::ggtitle(x_lab[i])
    }
    else{
      p <- ggplot2::ggplot() +
        ggplot2::geom_line(ggplot2::aes(x = x[, i], y = f[, i]), ...) +
        ggplot2::labs(x=x_lab[i], y=y_lab[i]) +
        ggplot2::ggtitle(x_lab[i])
    }
    plots_list[[i]] <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(p))

  }

  return(patchwork::wrap_plots(plotlist = plots_list) +
           patchwork::plot_layout(ncol = ncol(x)) +
           patchwork::plot_annotation(title = title)
         )
}
