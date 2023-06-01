n <- 24500

# binomial

n <- 25000
x1 <- runif(n, -5, 5)
x2 <- runif(n, -5, 5)
x3 <- runif(n, -5, 5)
f1 <-x1**2
f2 <- 2*x2
f3 <- sin(x3)

f1 <- f1 - mean(f1)
f2 <- f2 - mean(f2)
f3 <- f3 - mean(f3)
eta0 <- 2 + f1 + f2 + f3
eta0 <- eta0 - mean(eta0) # recentrar eta 0 para balancear eta final
# para gaussian no es necesario
true_eta <- exp(eta0)/(1 + exp(eta0))
y <- rbinom(n, 1, true_eta)
table(y)

train <- data.frame(x1, x2, x3, f1, f2, f3, y)

library(neuralGAM)
ngam <- neuralGAM( y ~ s(x1) + x2 + s(x3), data = train,
                   num_units = 1024, family = "binomial",
                   learning_rate = 0.001, bf_threshold = 0.001,
                   max_iter_backfitting = 10, max_iter_ls = 10
)
