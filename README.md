# NeuralGAM
An R package which provides a Generalized Additive Model implementation with Neural Networks

## Requirements

<<<<<<< HEAD
NeuralGAM is based on Deep Neural Networks, and depends on Tensorflow and Keras packages. Therefore, a working Python installation with Keras and Tensorflow installed is required.

## Sample usage

In the following example, we use the sample synthetic dataset and fit a NeuralGAM model
with a single layer with 1024 units.  

```
library(NeuralGAM)

data(train)

X_train <- train[c('X0','X1','X2')]
y_train <- train$y

ngam <- fit_NeuralGAM(num_units = 1024, learning_rate = 0.001, x=X_train,
              y = y_train, family = "gaussian", bf_threshold=0.00001,
              ls_threshold = 0.1, max_iter_backfitting = 10,
              max_iter_ls=10)
=======
NeuralGAM is based on Deep Neural Networks, and depends on Tensorflow and Keras packages. Therefore, a working Python installation with Keras and Tensorflow installed is required. During installation, the package will generate a customized `conda` environment and install the required libraries. 

```
library(NeuralGAM)
>>>>>>> bfc67d5d2193f4d84c6b1027f65c6c628bce47df
```

You can then use the `plot` function to visualize the learnt partial effects: 

```
plot(ngam)
```

