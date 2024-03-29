% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predict.NeuralGAM.R
\name{predict.neuralGAM}
\alias{predict.neuralGAM}
\title{Produces predictions from a fitted \code{neuralGAM} object}
\usage{
\method{predict}{neuralGAM}(object, newdata = NULL, type = "link", terms = NULL, verbose = 1, ...)
}
\arguments{
\item{object}{a fitted `neuralGAM` object}

\item{newdata}{A data frame or list containing the values of covariates at which
predictions are required. If not provided, the function returns the predictions
for the original training data.}

\item{type}{when \code{type="link"} (default), the linear
predictor is returned. When \code{type="terms"} each component of the linear
predictor is returned separately on each column of a \code{data.frame}. When
\code{type="response"} predictions on the scale of the response are returned.}

\item{terms}{If \code{type="terms"}, then only results for the terms named
in this list will be returned. If \code{NULL} then no terms are excluded (default).}

\item{verbose}{Verbosity mode (0 = silent, 1 = print messages). Defaults to 1.}

\item{\ldots}{Other options.}
}
\value{
Predicted values according to \code{type} parameter.
}
\description{
Takes a fitted \code{neuralGAM} object produced by
\code{neuralGAM()} and produces predictions given a new set of values for the model covariates.
}
\examples{
\dontrun{

n <- 24500

seed <- 42
set.seed(seed)

x1 <- runif(n, -2.5, 2.5)
x2 <- runif(n, -2.5, 2.5)
x3 <- runif(n, -2.5, 2.5)

f1 <-x1**2
f2 <- 2*x2
f3 <- sin(x3)
f1 <- f1 - mean(f1)
f2 <- f2 - mean(f2)
f3 <- f3 - mean(f3)

eta0 <- 2 + f1 + f2 + f3
epsilon <- rnorm(n, 0.25)
y <- eta0 + epsilon
train <- data.frame(x1, x2, x3, y)

library(neuralGAM)
ngam <- neuralGAM(y ~ s(x1) + x2 + s(x3), data = train,
                 num_units = 1024, family = "gaussian",
                 activation = "relu",
                 learning_rate = 0.001, bf_threshold = 0.001,
                 max_iter_backfitting = 10, max_iter_ls = 10,
                 seed = seed
                 )
n <- 5000
x1 <- runif(n, -2.5, 2.5)
x2 <- runif(n, -2.5, 2.5)
x3 <- runif(n, -2.5, 2.5)
test <- data.frame(x1, x2, x3)

# Obtain linear predictor
eta <- predict(ngam, test, type = "link")

# Obtain predicted response
yhat <- predict(ngam, test, type = "response")

# Obtain each component of the linear predictor
terms <- predict(ngam, test, type = "terms")

# Obtain only certain terms:
terms <- predict(ngam, test, type = "terms", terms = c("x1", "x2"))
}
}
