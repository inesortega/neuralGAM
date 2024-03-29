% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/inv_link.R
\name{inv_link}
\alias{inv_link}
\title{Inverse of the link functions}
\usage{
inv_link(family, muhat)
}
\arguments{
\item{family}{A description of the link function used in the model:
\code{"gaussian"} or \code{"binomial"}}

\item{muhat}{fitted values}
}
\value{
the inverse link function specified by the \code{"family"}
distribution for the given fitted values
}
\description{
Computes the inverse of the link function according to the
distribution family specified in the \code{"family"} parameter.
}
\author{
Ines Ortega-Fernandez, Marta Sestelo.
}
\keyword{internal}
