% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dev.R
\name{dev}
\alias{dev}
\title{Deviance of the model}
\usage{
dev(muhat, y, family)
}
\arguments{
\item{muhat}{current estimation of the response variable}

\item{y}{response variable}

\item{family}{A description of the link function used in the model:
\code{"gaussian"} or \code{"binomial"}}
}
\value{
the deviance of the model
}
\description{
Computes the deviance of the model according to the distribution
family specified in the \code{"family"} parameter.
}
\author{
Ines Ortega-Fernandez, Marta Sestelo.
}
\keyword{internal}
