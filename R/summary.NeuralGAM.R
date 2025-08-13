#' @title Summary of a \code{neuralGAM} model
#' @description
#' Summarizes a fitted \code{neuralGAM} object: family, formula, sample size,
#' intercept, training MSE, per-term neural net settings, per-term NN layer
#' configuration, and training history. If a linear component is present, its
#' coefficients are also reported.
#'
#' @param object A \code{neuralGAM} object.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns \code{object}. Prints a human-readable summary.
#' @export
#' @examples
#' \dontrun{
#' n <- 24500
#' seed <- 42
#' set.seed(seed)
#'
#' x1 <- runif(n, -2.5, 2.5)
#' x2 <- runif(n, -2.5, 2.5)
#' x3 <- runif(n, -2.5, 2.5)
#'
#' f1 <- x1**2
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
#' ngam <- neuralGAM(
#'   y ~ s(x1) + x2 + s(x3),
#'   data = train,
#'   num_units = 1024,
#'   family = "gaussian",
#'   activation = "relu",
#'   learning_rate = 0.001,
#'   bf_threshold = 0.001,
#'   max_iter_backfitting = 10,
#'   max_iter_ls = 10,
#'   seed = seed
#' )
#' summary(ngam)
#' }
summary.neuralGAM <- function(object, ...) {
  if (!inherits(object, "neuralGAM")) {
    stop("Argument 'object' must be a neuralGAM object.")
  }
  ngam <- object

  # --- small helpers ----------------------------------------------------------
  .is_py <- function(x) inherits(x, "python.builtin.object")
  .cfg_get <- function(lst, key, default = NA) {
    if (is.null(lst)) return(default)
    if (!is.list(lst)) return(default)
    if (is.null(lst[[key]])) default else lst[[key]]
  }
  .scalar <- function(x) if (length(x) == 1) x else paste(x, collapse = ", ")

  # Extract build_pi/alpha if present in object (stored via your constructor)
  # If you already store them elsewhere, adapt below:
  build_pi <- tryCatch(isTRUE(ngam$build_pi), error = function(e) FALSE)
  alpha    <- tryCatch(ngam$alpha, error = function(e) NA_real_)

  # Global defaults (if you stored them, read from object; else NA)
  globals <- tryCatch(ngam$globals, error = function(e) NULL)
  g_num_units          <- .cfg_get(globals, "num_units", NA)
  g_activation         <- .cfg_get(globals, "activation", NA)
  g_learning_rate      <- .cfg_get(globals, "learning_rate", NA)
  g_kernel_initializer <- .cfg_get(globals, "kernel_initializer", NA)
  g_bias_initializer   <- .cfg_get(globals, "bias_initializer", NA)
  g_kernel_regularizer <- .cfg_get(globals, "kernel_regularizer", NA)
  g_bias_regularizer   <- .cfg_get(globals, "bias_regularizer", NA)
  g_activity_regularizer <- .cfg_get(globals, "activity_regularizer", NA)

  cat("neuralGAM summary\n")
  cat(rep("=", 72), sep = "", "\n")
  cat("Family          : ", ngam$family, "\n", sep = "")
  cat("Formula         : ", deparse(ngam$formula$formula), "\n", sep = "")
  cat("Observations    : ", NROW(ngam$y), "\n", sep = "")
  cat("Intercept (eta0): ", format(ngam$eta0, digits = 6), "\n", sep = "")
  cat("Train MSE       : ", format(ngam$mse, digits = 6), "\n", sep = "")
  if (isTRUE(build_pi)) {
    cat("Prediction Int. : ENABLED (alpha = ", alpha, ")\n", sep = "")
  } else {
    cat("Prediction Int. : disabled\n")
  }
  cat(rep("-", 72), sep = "", "\n")

  # ----- concise per-term configuration print ----------------------------------
  cat("Per-term configuration (parsed from s(...))\n")
  if (length(ngam$formula$np_terms) == 0) {
    cat("  (no smooth terms)\n")
  } else {
    for (term in ngam$formula$np_terms) {
      cfg <- ngam$formula$np_architecture[[term]]

      units <- .scalar(if (!is.null(cfg$num_units)) cfg$num_units else .scalar(g_num_units))
      act   <- .scalar(if (!is.null(cfg$activation)) cfg$activation else .scalar(g_activation))
      lr    <- .scalar(if (!is.null(cfg$learning_rate)) cfg$learning_rate else .scalar(g_learning_rate))

      kinit <- .pretty_keras_obj(if (!is.null(cfg$kernel_initializer))  cfg$kernel_initializer  else g_kernel_initializer)
      binit <- .pretty_keras_obj(if (!is.null(cfg$bias_initializer))    cfg$bias_initializer    else g_bias_initializer)
      kreg  <- .pretty_keras_obj(if (!is.null(cfg$kernel_regularizer))  cfg$kernel_regularizer  else g_kernel_regularizer)
      breg  <- .pretty_keras_obj(if (!is.null(cfg$bias_regularizer))    cfg$bias_regularizer    else g_bias_regularizer)
      areg  <- .pretty_keras_obj(if (!is.null(cfg$activity_regularizer))cfg$activity_regularizer else g_activity_regularizer)

      line <- sprintf(" -- %s - units: %s | activation: %s | learning rate: %s | k_init: %s | b_init: %s | k_reg: %s | b_reg: %s | a_reg: %s",
                      term, units, act, lr, kinit, binit, kreg, breg, areg)
      cat(line, "\n", sep = "")
    }
  }
  cat(rep("-", 72), sep = "", "\n")

  # --- Keras model layer summaries per smooth term ----------------------------
  .layer_summary <- function(keras_model) {
    out <- data.frame(
      layer_index = integer(0),
      class       = character(0),
      units       = character(0),
      activation  = character(0),
      kernel_init = character(0),
      bias_init   = character(0),
      kernel_reg  = character(0),
      bias_reg    = character(0),
      stringsAsFactors = FALSE
    )
    if (is.null(keras_model) || !.is_py(keras_model)) return(out)

    layers <- tryCatch(keras_model$layers, error = function(e) NULL)
    if (is.null(layers)) return(out)

    for (i in seq_along(layers)) {
      lyr <- layers[[i]]
      cls <- tryCatch(reticulate::py_get_attr(reticulate::py_get_attr(lyr, "__class__"), "__name__"), error = function(e) NA_character_)
      cfg <- tryCatch(lyr$get_config(), error = function(e) NULL)

      get_name <- function(entry) {
        if (is.null(entry)) return(NA_character_)
        # In TF2: list with class_name + config; sometimes a plain string
        if (is.list(entry) && !is.null(entry$class_name)) return(as.character(entry$class_name))
        if (is.character(entry)) return(entry[1])
        deparse1(entry)
      }

      units      <- if (!is.null(cfg$units)) as.character(cfg$units) else NA_character_
      activation <- if (!is.null(cfg$activation)) as.character(cfg$activation) else NA_character_
      kinit      <- get_name(cfg$kernel_initializer)
      binit      <- get_name(cfg$bias_initializer)
      kreg       <- get_name(cfg$kernel_regularizer)
      breg       <- get_name(cfg$bias_regularizer)

      out[nrow(out) + 1L, ] <- list(i, cfg$name, .collapse_val(units), .collapse_val(activation), .collapse_val(kinit), .collapse_val(binit), .collapse_val(kreg), .collapse_val(breg))
    }
    out
  }

  cat("Neural network layer configuration per smooth term\n")
  if (length(ngam$formula$np_terms)) {
    for (term in ngam$formula$np_terms) {
      mdl <- ngam$model[[term]]
      cat(" -- ", term, "\n", sep = "")
      tbl <- .layer_summary(mdl)
      if (NROW(tbl) == 0) {
        cat("    (no accessible layer config)\n")
      } else {
        print(tbl, row.names = FALSE)
      }
    }
  } else {
    cat("  (no smooth terms)\n")
  }
  cat(rep("-", 72), sep = "", "\n")

  # --- Linear model (if present) ----------------------------------------------
  if (!is.null(ngam$model$linear)) {
    cat("Linear component coefficients\n")
    print(ngam$model$linear$coefficients)
    cat(rep("-", 72), sep = "", "\n")
  }

  # --- Training history (compact) ---------------------------------------------
  if (!is.null(ngam$stats) && NROW(ngam$stats)) {
    cat("Training history (head)\n")
    print(utils::head(ngam$stats, 10))
  } else {
    cat("Training history: (none)\n")
  }

  invisible(object)
}


# ----- helpers to pretty-print keras objects ---------------------------------
.is_py   <- function(x) inherits(x, "python.builtin.object")
.py_has  <- function(x, attr) reticulate::py_has_attr(x, attr)
.py_get  <- function(x, attr) reticulate::py_get_attr(x, attr)

.pretty_keras_obj <- function(x) {
  if (is.null(x)) return("NA")
  if (is.character(x) && length(x) == 1) return(x)

  if (.is_py(x)) {
    cls <- tryCatch(.py_get(.py_get(x, "__class__"), "__name__"), error = function(e) NA_character_)
    cfg <- NULL
    if (.py_has(x, "get_config")) {
      cfg <- tryCatch(x$get_config(), error = function(e) NULL)
    }

    # Regularizers
    if (!is.null(cfg) && is.list(cfg) && all(c("l1","l2") %in% names(cfg))) {
      l1 <- cfg$l1; l2 <- cfg$l2
      if (!is.null(l2) && length(l2) && l2 != 0) return(sprintf("l2(%g)", l2))
      if (!is.null(l1) && length(l1) && l1 != 0) return(sprintf("l1(%g)", l1))
    }

    # Initializers
    if (!is.null(cls)) {
      if (!is.null(cfg) && is.list(cfg)) {
        keys <- intersect(names(cfg), c("seed","mean","stddev","scale"))
        if (length(keys)) {
          parts <- vapply(
            keys,
            function(k) {
              val <- cfg[[k]]
              if (is.null(val) || length(val) == 0) return(sprintf("%s=NA", k))
              sprintf("%s=%s", k, as.character(val))
            },
            character(1)
          )
          return(sprintf("%s(%s)", cls, paste(parts, collapse = ", ")))
        }
      }
      return(cls)
    }
  }

  # R object fallback
  cl <- class(x)
  hit <- grep("(Initializer|Regularizer)$", cl, value = TRUE)
  if (length(hit)) return(hit[1])
  deparse1(x)
}
.scalar <- function(x) if (length(x) <= 1) as.character(x) else paste(x, collapse = ",")


.collapse_val <- function(x) {
  if (length(x) > 1) {
    paste0("[", paste(x, collapse = ", "), "]")
  } else if (length(x) == 0 || is.null(x)) {
    "NA"
  } else {
    as.character(x)
  }
}
