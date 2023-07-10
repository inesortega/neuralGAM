#' Extract formula elements
#'
#' This function separates the model terms of a given formula into response,
#' all_terms, non-parametric terms and parametric terms.
#'
#' @param formula A formula object
#'
#' @return A list with the following elements:
#' \itemize{
#' \item y: The response variable
#' \item terms: A character vector with all model terms
#' \item np_terms: A character vector with non-parametric terms
#' \item p_terms: A character vector with parametric terms
#' \item np_formula: The formula for the non-parametric terms
#' \item p_formula: The formula for the parametric terms
#' \item formula: The original formula object
#' }
#' @importFrom formula.tools lhs rhs
#' @importFrom stats as.formula
#' @keywords internal
get_formula_elements <- function(formula) {
  # Separate model terms (response, all_terms, smooth_terms)
  formula <- formula
  y <- formula.tools::lhs(formula)
  all_terms <- all.vars(formula.tools::rhs(formula))
  terms <- formula.tools::rhs(formula)

  pattern <- "s\\(.*\\)"
  smooth_terms <-
    grep(pattern, attr(terms(formula), "term.labels"), value = TRUE)

  if (length(smooth_terms) > 0) {
    smooth_formula <-
      stats::as.formula(paste("y ~ ", paste(smooth_terms, collapse = " + ")))
    smooth_terms <- all.vars(formula.tools::rhs(smooth_formula))
  }
  else{
    smooth_formula <- NULL
    smooth_terms <- NULL
  }

  linear_terms = setdiff(all_terms, smooth_terms)
  if (length(linear_terms) > 0) {
    linear_formula <-
      stats::as.formula(paste("y ~ ", paste(linear_terms, collapse = " + ")))
  }
  else{
    linear_formula <- NULL
    linear_terms <- NULL
  }


  return(
    list(
      y = y,
      terms = all_terms,
      np_terms = smooth_terms,
      p_terms = linear_terms,
      np_formula = smooth_formula,
      p_formula = linear_formula,
      formula = formula
    )
  )

}
