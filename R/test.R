#' Homoscedastic Uniform Gaussian synthetic test data.
#'
#' Homoscedastic Uniform Gaussian synthetic data frame with 7 variables
#' and 5000 measurements for test/evaluation. This dataset was generated to build a
#' simulation scenario to evaluate NeuralGAM.
#'
#' We considered the predictor: \eqn{\eta = \alpha + \sum^3_{j=1} f_j(X_j)}, with
#'
#' \deqn{
#' f_j(X_j) =
#' \begin{cases}
#' X_j^2 \hspace{1.4cm} \text{ if } j=1 \\
#' 2X_j \hspace{1.3cm} \text{ if } j=2 \\
#' \sin{X_j} \hspace{1cm} \text{ if } j=3,
#' \end{cases}
#' }
#' \eqn{\alpha = 2}, and covariates \eqn{X_1, X_2, X_3} drawn from an uniform
#' distribution \eqn{U\left[-2.5, 2.5\right]} and response variable
#' \eqn{Y = \eta + \varepsilon} where \eqn{\varepsilon} is an homoscedastic error term
#' distributed in accordance to a \eqn{N(0,\sigma(x)}
#'
#' @name test
#' @docType data
#' @usage data(test)
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @format \code{test} is a data frame with 7
#' variables (columns).
#' The first three columns of the data frame contains the covariates X0, X1, X2.
#' The next three columns of the data frame contains the corresponding f(X) for
#' each covariate
#' The last column of the data frame contains the response variable.
#' contains the response variable
#' @examples
#' library(NeuralGAM)
#' data(test)
#' head(test)
#'
#' X_train <- test[c("X0", "X1", "X2")]
#' fs_train <- test[c("f(X0)", "f(X1)", "f(X2)")]
#' y_train <- test["y"]
NULL
