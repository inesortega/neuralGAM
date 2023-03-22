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
library(NeuralGAM)

data(train)

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
plot(ngam)
```

