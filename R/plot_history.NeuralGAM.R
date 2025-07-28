#' Plot training loss history for a neuralGAM model
#'
#' This function visualizes the training and/or validation loss over backfitting iterations
#' for each term-specific model in a `neuralGAM` object. It is designed to work with the
#' `history` component of a trained `neuralGAM` model.
#'
#' @param history A named list (e.g. `ngam$history`) where each element is a list of
#' `keras_training_history` objects returned by each `fit()` call for a specific model term.
#' @param select Optional character vector of term names (e.g. `"x1"` or `c("x1", "x3")`) to subset
#' the history. If `NULL` (default), all terms are included.
#' @param metric Character vector indicating which loss metric(s) to plot. Options are
#' `"loss"`, `"val_loss"`, or both. Defaults to both.
#'
#' @return A `ggplot` object showing the loss curves by backfitting iteration, with facets per term.
#' @export
#'
#' @import ggplot2
#'
#' @examples
#' if (requireNamespace("neuralGAM", quietly = TRUE)) {
#'   set.seed(123)
#'   n <- 200
#'   x1 <- runif(n, -2, 2)
#'   x2 <- runif(n, -2, 2)
#'   y <- 2 + x1^2 + sin(x2) + rnorm(n, 0, 0.1)
#'   df <- data.frame(x1 = x1, x2 = x2, y = y)
#'
#'   model <- neuralGAM::neuralGAM(
#'     y ~ s(x1) + s(x2),
#'     data = df,
#'     num_units = 8,
#'     family = "gaussian",
#'     max_iter_backfitting = 2,
#'     max_iter_ls = 1,
#'     learning_rate = 0.01,
#'     seed = 42,
#'     validation_split = 0.2,
#'     verbose = 0
#'   )
#'
#'   plot_history(model$history)                      # Plot all terms
#'   plot_history(model$history, select = "x1")       # Plot just x1
#'   plot_history(model$history, metric = "val_loss") # Plot only validation loss
#' }
plot_history <- function(history, select = NULL, metric = c("loss", "val_loss")) {
  metric <- match.arg(metric, several.ok = TRUE)

  if (!is.null(select)) {
    if (!all(select %in% names(history))) {
      stop("Some selected terms are not in the history object.")
    }
    history <- history[select]
  }

  df_list <- list()

  for (term in names(history)) {
    term_hist <- history[[term]]
    for (i in seq_along(term_hist)) {
      h <- term_hist[[i]]
      row <- list(Term = term, Iteration = i)

      # Handle scalar or length-1 loss
      if ("loss" %in% metric) {
        loss_val <- h$metrics$loss
        if (length(loss_val) > 1) loss_val <- tail(loss_val, 1)
        row$Loss <- as.numeric(loss_val)
      }

      # Handle validation loss if available
      if ("val_loss" %in% metric && "val_loss" %in% names(h$metrics)) {
        val_loss_val <- h$metrics$val_loss
        if (length(val_loss_val) > 1) val_loss_val <- tail(val_loss_val, 1)
        row$ValLoss <- as.numeric(val_loss_val)
      }

      df_list[[length(df_list) + 1]] <- row
    }
  }

  df <- do.call(rbind, lapply(df_list, as.data.frame))

  # Start plotting
  library(ggplot2)
  plt <- ggplot(df, aes(x = .data$Iteration)) +
    facet_wrap(~ Term, scales = "free_y")

  if ("Loss" %in% names(df)) {
    plt <- plt + geom_line(aes(y = .data$Loss, color = "Train Loss")) +
      geom_point(aes(y = .data$Loss, color = "Train Loss"))
  }

  if ("ValLoss" %in% names(df)) {
    plt <- plt + geom_line(aes(y = .data$ValLoss, color = "Validation Loss")) +
      geom_point(aes(y = .data$ValLoss, color = "Validation Loss"))
  }

  plt + labs(title = "Loss per Backfitting Iteration",
             y = "Loss", color = "Metric")
}
