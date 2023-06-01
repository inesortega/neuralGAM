n <- 24500

x1 <- runif(n, -3, 3)
x2 <- runif(n, -3, 3)
x3 <- runif(n, -3, 3)
x4 <- runif(n, -3, 3)

f1 <-x1**2
f2 <- 2*x2
f3 <- sin(x3)
f4 <- x4

f1 <- f1 - mean(f1)
f2 <- f2 - mean(f2)
f3 <- f3 - mean(f3)
f4 <- f4 - mean(f4)

eta0 <- 2 + f1 + f2 + f3 + f4

epsilon <- rnorm(n, 0.25)
y <- eta0 + epsilon

train <- data.frame(x1, x2, x3, x4, y, f1, f2, f3, f4)

library(neuralGAM)
ngam <- neuralGAM( y ~ s(x1) + x2 + s(x3) + x4 , data = train,
                   num_units = 1024, family = "gaussian",
                   learning_rate = 0.001, bf_threshold = 0.01,
                   max_iter_backfitting = 10, max_iter_ls = 10
)
