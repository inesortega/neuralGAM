# NeuralGAM

NeuralGAM, a neural network based on Generalized Additive Models, which trains a different neural network to estimate the contribution of each feature to the response variable. 

The networks are trained independently leveraging the local scoring and backfitting algorithms to ensure that the Generalized Additive Model converges and it is additive. 

The resultant Neural Network is a highly accurate and interpretable deep learning model, which can be used for high-risk AI practices where decision-making should be based on accountable and interpretable algorithms. 
            
## Requirements

NeuralGAM is based on Deep Neural Networks, and depends on Tensorflow and Keras packages. Therefore, a working Python>3.9 installation is required.

When loading the package, it will automatically generate a working conda environment with 
Keras and Tensorflow installed. 

## Sample usage

In the following example, we use the sample synthetic dataset and fit a NeuralGAM model
with a single layer with 1024 units.  

```
n <- 24500
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
train <- data.frame(x1, x2, x3, y, f1, f2, f3)

library(NeuralGAM)

X_train <- train[c('X0','X1','X2')]
y_train <- train$y

ngam <- NeuralGAM(num_units = 1024, learning_rate = 0.0053, x=X_train,
              y = y_train, family = "gaussian", bf_threshold=0.001,
              ls_threshold = 0.1, max_iter_backfitting = 10,
              max_iter_ls=10)

```
You can then use the `plot` function to visualize the learnt partial effects: 

```
plot(ngam)
```

To obtain predictions from new data, use the `predict` function: 

```
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

```

