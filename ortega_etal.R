######### APPLICATION TO SIMULATED DATA ###############
seed <- 42
set.seed(seed)
n <- 30625
x1 <- runif(n,-2.5, 2.5)
x2 <- runif(n,-2.5, 2.5)
x3 <- runif(n,-2.5, 2.5)
f1 <- x1 ** 2
f2 <- 2 * x2
f3 <- sin(x3)
f1 <- f1 - mean(f1)
f2 <- f2 - mean(f2)
f3 <- f3 - mean(f3)
eta0 <- 2 + f1 + f2 + f3
epsilon <- rnorm(n, 0.25)
y <- eta0 + epsilon
dat <- data.frame(x1, x2, x3, y)
sample <-
  sample(c(TRUE, FALSE),
         n,
         replace = TRUE,
         prob = c(0.8, 0.2))
train  <- dat[sample, ]
test <- dat[!sample, ]
summary(dat)
library(neuralGAM)
ngam <- neuralGAM(y ~ s(x1) + x2 + s(x3), data = train,
                  num_units = 1024,
                  learning_rate = 0.001,
                  bf_threshold = 0.001,
                  seed = seed)
ngam
plots <- lapply(c("x1", "x2", "x3"), function(x) autoplot(ngam, select = x))
gridExtra::grid.arrange(grobs = plots, ncol = 3, nrow = 1)
eta <- predict(ngam, newdata = test, type = "link")
head(eta)
yhat <- predict(ngam, newdata = test, type = "response")
head(yhat)
terms <- predict(ngam, newdata = test, type = "terms")
head(terms)
terms <- predict(ngam, newdata = test, type = "terms", terms = c("x1", "x3"))
head(terms)
mse <- mean((yhat - test$y)^2)
mse
r_squared <- cor(yhat, test$y)^2
r_squared

######### APPLICATION TO REAL DATA ###############
library(magrittr)
library(dplyr)
library(ggplot2)
library(gridExtra)

seed <- 1234
set.seed(seed)
data(flights, package = "nycflights13")
data(weather, package = "nycflights13")
data(airlines, package = "nycflights13")
dat <-
  filter(flights, origin == "EWR" & month %in% c(12, 11, 10)) %>%
  left_join(weather, by = c("origin", "time_hour"))  %>%
  select(
    arr_delay,
    dep_delay,
    air_time,
    temp,
    humid
  ) %>%
  data.frame
# Convert temperature to Celsius
dat$temp <- (dat$temp - 32) * (5/9)
dat$delay = ifelse(dat$arr_delay > 0, 1, 0)
dat <- dat[!rowSums(is.na(dat)),]
print(dat %>% count(delay))
head(dat)

sample <- sample(nrow(dat), 0.8 * nrow(dat))
train <- dat[sample, ]
test <- dat[-sample, ]

library(neuralGAM)
ngam <-
  neuralGAM(
    delay ~ s(air_time) + s(dep_delay) + s(temp) + s(humid),
    data = train,
    num_units = c(256, 128),
    family = "binomial",
    seed = seed,
    bf_threshold = 1e-4,
    ls_threshold = 0.01,
    loss = "mse"
  )

ngam

p1 <- autoplot(ngam, select = "air_time", xlab = "Air Time (min)")
p2 <- autoplot(ngam, select = "dep_delay", xlab = "Departure Delay (min)")
p3 <- autoplot(ngam, select = "temp", xlab = "Temperature (ºC)")
p4 <- autoplot(ngam, select = "air_time", xlab = "Relative Humidity")
gridExtra::grid.arrange(grobs = list(p1,p2,p3,p4), ncol = 2, nrow = 2)
predictions <- predict(ngam, newdata = test, type = "response")
library(pROC)
roc_data <- roc(test$delay, predictions)
roc_data$auc
