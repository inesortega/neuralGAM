n <- 24500

set.seed(0)
## fake some data...
f1 <- function(x) {exp(2 * x)}
f2 <- function(x) {
  0.2*x^11*(10*(1-x))^6+10*(10*x)^3*(1-x)^10
}
f3 <- function(x) {x*0}

sig2<-4

x0 <- rep(1:4,24500/4)
x1 <- runif(n, 0, 1)
x2 <- runif(n, 0, 1)
x3 <- runif(n, 0, 1)
e <- rnorm(n, 0, sqrt(sig2))
y <- 2*x0 + f1(x1) + f2(x2) + f3(x3) + e
x0 <- factor(x0)

train <- data.frame(x0,x1,x2,x3,y)

library(neuralGAM)

ngam <- neuralGAM( y~x0+s(x1)+s(x2)+s(x3) , data = train,
                   num_units = 1024, family = "gaussian",
                   learning_rate = 0.001, bf_threshold = 0.001,
                   max_iter_backfitting = 10, max_iter_ls = 10
)
