
# Linear Regression with Homoscedastic error and Uniform distribution

type <- "homoscedastic_uniform_gaussian"
X_train <- read.csv(paste("./dataset/Scenario_I/", type, "/X_train.csv", sep=""))
train <- data.frame(X_train[c("X0","X1","X2")])

fs_train <- read.csv(paste("./dataset/Scenario_I/", type, "/fs_train.csv", sep=""))
fs_train <- fs_train[,-1]
fs_train <- scale(fs_train, center = TRUE, scale = FALSE)

train['f(X0)'] = fs_train[,1]
train['f(X1)'] = fs_train[,2]
train['f(X2)'] = fs_train[,2]

y_train <- read.csv(paste("./dataset/Scenario_I/", type, "/y_train.csv", sep=""))
y_train <- y_train[,-1]
train['y'] = y_train

save(train, file="./data/train.RData")

type <- "homoscedastic_uniform_gaussian"
X_test <- read.csv(paste("./dataset/Scenario_I/", type, "/X_test.csv", sep=""))
test <- data.frame(X_test[c("X0","X1","X2")])

fs_test <- read.csv(paste("./dataset/Scenario_I/", type, "/fs_test.csv", sep=""))
fs_test <- fs_test[,-1]
fs_test <- scale(fs_test, center = TRUE, scale = FALSE)
test['f(X0)'] = fs_test[,1]
test['f(X1)'] = fs_test[,2]
test['f(X2)'] = fs_test[,2]

y_test <- read.csv(paste("./dataset/Scenario_I/", type, "/y_test.csv", sep=""))
y_test <- y_test[,-1]
test['y'] = y_test

save(test, file="./data/test.RData")
