#' @title Extract structured elements from a model formula
#'
#' @description
#' Parses a GAM-style formula and separates the response, all terms, smooth
#' (non‑parametric) terms declared via `s(...)`, and parametric terms. In addition,
#' it extracts **per‑term neural network specifications** that can be written
#' inline inside each `s(...)` call (e.g., `num_units`, `activation`,
#' `kernel_initializer`, `bias_initializer`, `kernel_regularizer`,
#' `bias_regularizer`, `activity_regularizer`).
#'
#' This function uses an abstract syntax tree (AST) walker (no regex) to read
#' the arguments of each `s(var, ...)`. Arguments are evaluated in the caller’s
#' environment. For `*_regularizer` and `*_initializer` arguments, **proper Keras
#' objects are required** (e.g., `keras::regularizer_l2(1e-4)`,
#' `keras::initializer_glorot_uniform()`).
#'
#' @param formula A model formula. Smooth terms must be written as `s(var, ...)`.
#'
#' @return
#' A list with the following elements:
#' \describe{
#'   \item{y}{Character scalar. The response variable name.}
#'   \item{terms}{Character vector with all variable names on the RHS (both smooth and parametric).}
#'   \item{np_terms}{Character vector with smooth (non‑parametric) variable names extracted from `s(...)`.}
#'   \item{p_terms}{Character vector with parametric terms (i.e., `terms \\ np_terms`).}
#'   \item{np_formula}{A formula containing only the `s(...)` terms (or `NULL` if none).}
#'   \item{p_formula}{A formula containing only the parametric terms (or `NULL` if none).}
#'   \item{np_architecture}{Named list keyed by smooth term (e.g., `"x1"`, `"x3"`). Each entry is
#'     a list of per‑term settings parsed from `s(term, ...)`. Supported keys include:
#'     \code{num_units} (numeric or numeric vector),
#'     \code{activation} (character),
#'     \code{learning_rate} (numeric),
#'     \code{kernel_initializer} (keras initializer object),
#'     \code{bias_initializer} (keras initializer object),
#'     \code{kernel_regularizer} (keras regularizer object),
#'     \code{bias_regularizer} (keras regularizer object),
#'     \code{activity_regularizer} (keras regularizer object).}
#'   \item{formula}{The original formula object.}
#' }
#'
#' @details
#' \strong{Inline per‑term configuration in \code{s(...)}.}
#' You can specify neural network hyperparameters per smooth term, e.g.:
#' \preformatted{
#'   y ~ s(x1, num_units = c(1024, 512),
#'           activation = "tanh",
#'           kernel_regularizer = keras::regularizer_l2(1e-4)) +
#'       x2 +
#'       s(x3, num_units = 1024,
#'           bias_initializer = keras::initializer_zeros())
#' }
#' Values are evaluated in the caller’s environment. For regularizers and
#' initializers you must pass actual Keras objects (not character strings).
#'
#' \strong{Supported keys.}
#' Only the keys listed in \code{np_architecture} above are recognized.
#' Unknown keys are ignored with a warning.
#'
#' \strong{Typical usage.}
#' The returned \code{np_terms} and \code{np_architecture} are consumed by
#' model-building code to construct one neural network per smooth term, applying
#' any per‑term overrides while falling back to global defaults for unspecified keys.
#'
#' @section Errors and validation:
#' \itemize{
#'   \item The first argument of each \code{s(...)} must be a symbol naming the variable.
#'   \item All additional arguments to \code{s(...)} must be named.
#'   \item \code{*_regularizer} must be a Keras regularizer object.
#'   \item \code{*_initializer} must be a Keras initializer object.
#' }
#'
#' @examples
#' \dontrun{
#'   fml <- y ~ s(x1, num_units = c(256,128),
#'                    activation = "relu",
#'                    kernel_regularizer = keras::regularizer_l2(1e-4)) +
#'               x2 +
#'               s(x3, num_units = 512,
#'                    bias_initializer = keras::initializer_zeros())
#'   parts <- get_formula_elements(fml)
#'   str(parts$np_terms)         # "x1" "x3"
#'   str(parts$np_architecture$x1)
#' }
#'
#' @importFrom formula.tools lhs rhs
#' @importFrom stats as.formula terms.formula
#' @keywords internal
get_formula_elements <- function(formula) {
  y <- formula.tools::lhs(formula)
  rhs <- formula.tools::rhs(formula)
  all_terms <- all.vars(rhs)

  parsed <- .parse_s_terms_adv(formula)
  np_terms <- parsed$np_terms
  np_architecture <- parsed$np_architecture

  term_labels <- attr(stats::terms.formula(formula), "term.labels")
  s_labels <- if (length(term_labels)) grep("^s\\(", term_labels, value = TRUE) else character()
  np_formula <- if (length(s_labels)) stats::as.formula(paste("y ~", paste(s_labels, collapse = " + "))) else NULL

  p_terms <- setdiff(all_terms, np_terms)
  p_formula <- if (length(p_terms)) stats::as.formula(paste("y ~", paste(p_terms, collapse = " + "))) else NULL

  list(
    y = y,
    terms = all_terms,
    np_terms = np_terms,
    p_terms = p_terms,
    np_formula = np_formula,
    p_formula = p_formula,
    np_architecture = np_architecture,
    formula = formula
  )
}

.NG_SUPPORTED_KEYS <- c(
  "num_units",              # numeric or numeric vector
  "activation",             # character
  "learning_rate",          # numeric
  "kernel_initializer",     # keras initializer object
  "bias_initializer",       # keras initializer object
  "kernel_regularizer",     # keras regularizer object
  "bias_regularizer",       # keras regularizer object
  "activity_regularizer"    # keras regularizer object
)


# Helpers to validate keras objects

.is_python_obj <- function(x) inherits(x, "python.builtin.object")

.py_class_info <- function(x) {
  out <- list(name = NULL, module = NULL)
  if (!.is_python_obj(x)) return(out)
  if (!reticulate::py_has_attr(x, "__class__")) return(out)
  cls <- reticulate::py_get_attr(x, "__class__")
  out$name   <- tryCatch(reticulate::py_get_attr(cls, "__name__"),   error = function(e) NULL)
  out$module <- tryCatch(reticulate::py_get_attr(cls, "__module__"), error = function(e) NULL)
  out
}

.is_keras_initializer <- function(x) {
  if (is.null(x)) return(FALSE)
  if (.is_python_obj(x)) {
    ci <- .py_class_info(x)
    if (!is.null(ci$name) && grepl("Initializer$", ci$name)) return(TRUE)
    if (!is.null(ci$module) && grepl("keras", ci$module) && grepl("initializers", ci$module)) return(TRUE)
  }
  # Fallback for older wrappers where R classes are labeled
  any(grepl("Initializer", class(x), fixed = TRUE))
}

.is_keras_regularizer <- function(x) {
  if (is.null(x)) return(FALSE)
  if (.is_python_obj(x)) {
    ci <- .py_class_info(x)
    if (!is.null(ci$name) && grepl("Regularizer$", ci$name)) return(TRUE)
    if (!is.null(ci$module) && grepl("keras", ci$module) && grepl("regularizers", ci$module)) return(TRUE)
  }
  any(grepl("Regularizer", class(x), fixed = TRUE))
}

# Advanced parser for s(...) terms
.parse_s_terms_adv <- function(formula, env = rlang::caller_env()) {
  rhs <- rlang::f_rhs(formula)
  np_terms <- character()
  np_architecture <- list()

  walk <- function(expr) {
    if (rlang::is_symbol(expr) || rlang::is_atomic(expr) || rlang::is_null(expr))
      return(invisible())

    if (rlang::is_call(expr)) {
      fname <- rlang::call_name(expr)

      # Handle binary ops like +
      if (identical(fname, "+")) {
        lapply(rlang::call_args(expr), walk)
        return(invisible())
      }

      # Found s(...)
      if (identical(fname, "s")) {
        args <- rlang::call_args(expr)
        if (length(args) < 1L)
          stop("s(...) must have at least one argument (the variable).")

        # First arg must be variable name
        var_expr <- args[[1]]
        if (!rlang::is_symbol(var_expr))
          stop("First argument of s(...) must be a symbol naming the variable.")
        var_name <- rlang::as_string(var_expr)
        np_terms <<- unique(c(np_terms, var_name))

        # Evaluate named args in caller's env
        spec <- list()
        if (length(args) > 1L) {
          named <- args[-1]
          arg_names <- names(named)
          if (is.null(arg_names) || any(arg_names == "")) {
            stop(sprintf("All arguments to s(%s, ...) after the first must be named.", var_name))
          }

          for (nm in arg_names) {
            if (!(nm %in% .NG_SUPPORTED_KEYS)) {
              warning(sprintf("Ignoring unsupported argument '%s' in s(%s, ...).", nm, var_name))
              next
            }

            val <- rlang::eval_bare(named[[nm]], env = env)

            # Validation for keras object types
            if (grepl("regularizer", nm)) {
              if (!.is_keras_regularizer(val)) {
                stop(sprintf(
                  "Argument '%s' for s(%s, ...) must be a keras regularizer object, e.g. keras::regularizer_l2(1e-4).",
                  nm, var_name
                ))
              }
            }
            if (grepl("initializer", nm)) {
              if (!.is_keras_initializer(val)) {
                stop(sprintf(
                  "Argument '%s' for s(%s, ...) must be a keras initializer object, e.g. keras::initializer_glorot_uniform().",
                  nm, var_name
                ))
              }
            }

            spec[[nm]] <- val
          }
        }

        np_architecture[[var_name]] <<- spec

        return(invisible())
      }

      # Otherwise, walk deeper
      lapply(rlang::call_args(expr), walk)
    }
  }

  walk(rhs)
  list(
    np_terms = if (length(np_terms)) np_terms else NULL,
    np_architecture = np_architecture
  )
}

# Turn common inputs into proper regularizer objects.
# Accept:
#   NULL
#   numeric scalar -> l2(weight)
#   call/object -> returned as-is
#   character -> parse+eval if it looks like call text, or map short names
.coerce_regularizer <- function(x) {
  if (is.null(x) || .is_keras_regularizer(x)) return(x)
  if (is.numeric(x) && length(x) == 1L) return(keras::regularizer_l2(x))
  if (is.character(x) && length(x) == 1L) {
    m <- tolower(trimws(x))
    if (m == "l2") return(keras::regularizer_l2(1e-4))
    if (m == "l1") return(keras::regularizer_l1(1e-4))
    stop("Unknown regularizer string: '", x, "'. Use keras::regularizer_* or numeric scalar.")
  }
  stop("Unsupported regularizer specification. Provide NULL, a keras regularizer object, a numeric scalar, or 'l1'/'l2'.")
}

.convert_regularizer <- function(x) {
  if (is.character(x)) {
    if (tolower(x) == "l2") return(keras::regularizer_l2())
    if (tolower(x) == "l1") return(keras::regularizer_l1())
    stop(sprintf("Unknown regularizer string '%s'. Use a keras::regularizer_* object instead.", x))
  }
  x
}
