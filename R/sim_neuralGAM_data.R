#' Simulate Example Data for NeuralGAM
#'
#' @description
#' Generate a synthetic dataset for demonstrating and testing
#' \code{\link{neuralGAM}}. The response is constructed from three covariates:
#' a quadratic effect, a linear effect, and a sinusoidal effect, plus Gaussian noise.
#'
#' @param n Integer. Number of observations to generate. Default \code{2000}.
#' @param seed Integer. Random seed for reproducibility. Default \code{42}.
#' @param test_prop Numeric in \eqn{[0,1]}. Proportion of data to reserve for the test set. Default \code{0.3}.
#'
#' @return A list with two elements:
#' \itemize{
#'   \item \code{train}: data.frame with training data.
#'   \item \code{test}: data.frame with test data.
#' }
#'
#' @details
#' The data generating process is:
#' \deqn{y = 2 + x1^2 + 2 x2 + \sin(x3) + \varepsilon,}
#' where \eqn{\varepsilon \sim N(0, 0.25^2)}.
#'
#' Covariates \eqn{x1}, \eqn{x2}, \eqn{x3} are drawn independently from
#' \eqn{U(-2.5, 2.5)}.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' dat <- sim_neuralGAM_data(n = 500, test_prop = 0.2)
#'
#' train <- dat$train
#' test  <- dat$test
#'
#' }
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @export
sim_neuralGAM_data <- function(n = 2000, seed = 42, test_prop = 0.3) {
  set.seed(seed)
  x1 <- runif(n, -2.5, 2.5)
  x2 <- runif(n, -2.5, 2.5)
  x3 <- runif(n, -2.5, 2.5)

  f1 <- x1^2
  f2 <- 2 * x2
  f3 <- sin(x3)

  y <- 2 + f1 + f2 + f3 + rnorm(n, 0.25)
  df <- data.frame(x1, x2, x3, y)

  # train/test split
  n_test <- floor(test_prop * n)
  idx <- sample(seq_len(n), n_test)

  list(
    train = df[-idx, , drop = FALSE],
    test  = df[ idx, , drop = FALSE]
  )
}
