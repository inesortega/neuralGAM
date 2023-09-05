#' Visualization of \code{neuralGAM} object with base graphics
#'
#' @description Visualization of \code{neuralGAM} object. Plots the learned partial effects by the neuralGAM object.
#' @param x a fitted \code{neuralGAM} object as produced by \code{neuralGAM()}.
#' @param select allows to plot a set of selected terms. e.g. if you just want to plot the first term,
#' select="X0"
#' @param xlab if supplied, this value will be used as the \code{x} label for all plots
#' @param ylab if supplied, this value will be used as the \code{y} label for all plots
#' @param \ldots other graphics parameters to pass on to plotting commands.
#' @return Returns the partial effects plot.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export
#' @examples \donttest{
#'
#' n <- 24500
#'
#' seed <- 42
#' set.seed(seed)
#'
#' x1 <- runif(n, -2.5, 2.5)
#' x2 <- runif(n, -2.5, 2.5)
#' x3 <- runif(n, -2.5, 2.5)
#'
#' f1 <-x1**2
#' f2 <- 2*x2
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
#' plot(ngam)
#' }
plot.neuralGAM <- function(x, select=NULL, xlab = NULL, ylab = NULL, ...) {

  if (!inherits(x, "neuralGAM")) {
    stop("Argument 'x' must be of class 'neuralGAM'")
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

  if(!is.null(xlab) && length(xlab) != length(ncol(x))){
    stop(paste("xlab must have labels for all the selected columns: "))
  }
  if(!is.null(ylab) && length(ylab) != length(ncol(x))){
    stop(paste("ylab must have labels for all the selected columns: "))
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
    y_lab <- rep(as.character(ngam$formula$y), ncol(x))
    for (i in 1:length(y_lab)){
      if(colnames(x)[i] %in% ngam$formula$np_terms){
        y_lab[i] <- paste("s(", x_lab[i], ")", sep="")
      }
      else{ ## todo take into account factor terms!
        if(is.factor(x[[colnames(x)[i]]])){
          y_lab[i] <- paste("Partial for ", x_lab[i])
        }
        else{
          y_lab[i] <- x_lab[i]
        }
      }
    }
  }

  plot_names <- colnames(x)

  # Loop through the plots
  for (i in seq_along(plot_names)) {

    term <- colnames(x)[[i]]

    # Generate the plot
    plot(x[, i], f[,i], xlab=x_lab[i], ylab=y_lab[i], ...)

    # Prompt the user to continue
    if (i < length(plot_names)) {
      message("Hit <Return> to see next plot: ")
      invisible(readLines(n = 1))
      }
    }

}
