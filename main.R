data(train)

X_train <- train[c('X0','X1','X2')]
y_train <- train$y

ngam <- fit_NeuralGAM(x=X_train, y=y_train, num_units = 1024, learning_rate = 0.001, family = "gaussian")


