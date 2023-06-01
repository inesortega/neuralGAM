seed <- 423
set.seed(seed)

n <- 30625

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

data <- data.frame(x1, x2, x3, y)
sample <- sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(0.8,0.2))
train <- data[sample, ]
test <- data[!sample, ]

library(neuralGAM)
ngam <- neuralGAM(y ~ s(x1) + x2 + s(x3), data = train,
                  num_units = 1024, family = "gaussian",
                  activation = "relu",
                  learning_rate = 0.001, bf_threshold = 0.001,
                  max_iter_backfitting = 10, max_iter_ls = 10,
                  seed = seed
)

ngam

plot(ngam)

summary(ngam)

eta <- predict(ngam, test, type = "link")
head(eta)

yhat <- predict(ngam, test, type = "response")
head(yhat)

terms <- predict(ngam, test, type = "terms")
head(terms)

terms <- predict(ngam, test, type = "terms", terms = "x1")
head(terms)
