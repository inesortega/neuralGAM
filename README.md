<!-- badges: start -->
[![R-CMD-check](https://github.com/inesortega/NeuralGAM/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/inesortega/NeuralGAM/actions/workflows/R-CMD-check.yaml)
[![test-coverage](https://github.com/inesortega/NeuralGAM/actions/workflows/test-coverage.yaml/badge.svg)](https://github.com/inesortega/NeuralGAM/actions/workflows/test-coverage.yaml)
[![Codecov test coverage](https://codecov.io/gh/inesortega/neuralGAM/branch/main/graph/badge.svg)](https://app.codecov.io/gh/inesortega/neuralGAM?branch=main)
[![CRAN Downloads](https://cranlogs.r-pkg.org/badges/grand-total/neuralGAM)](https://cranlogs.r-pkg.org/downloads/total/2023-09-01:2024-09-25/neuralGAM)
<!-- badges: end -->

# neuralGAM: Interpretable Neural Network Based on Generalized Additive Models

neuralGAM is a fully explainable Deep Learning framework based on Generalized Additive Models, which trains a different neural network to estimate the contribution of each feature to the response variable. 

The networks are trained independently leveraging the local scoring and backfitting algorithms to ensure that the Generalized Additive Model converges and it is additive. The resultant Neural Network is a highly accurate and interpretable deep learning model, which can be used for high-risk AI practices where decision-making should be based on accountable and interpretable algorithms. 

The full methodology of the method to train Generalized Additive Models using Deep Neural Networks is published in the following paper: 

> Ortega-Fernandez, I., Sestelo, M. & Villanueva, N.M. _Explainable generalized additive neural networks with independent neural network training_. Statistics & Computing 34, 6 (2024). https://doi.org/10.1007/s11222-023-10320-5

and is also available in Python at the following [Github repository](https://github.com/inesortega/pyNeuralGAM/).
  
## Requirements

neuralGAM is based on Deep Neural Networks, and depends on Tensorflow and Keras packages. Therefore, a working Python>3.9 installation with those packages installed is required. 

We provide a helper function to get a working python installation from RStudio, which creates a miniconda environment with all the required packages.   

```r
library(neuralGAM)
install_neuralGAM()
```

## Quick start

### Fit a neuralGAM with deep smooth terms

In the following example, we use synthetic data to showcase the performance of neuralGAM by fitting a model with a single layer with 1024 units.


```r
n <- 24500
seed <- 42
set.seed(seed)

x1 <- runif(n, -2.5, 2.5)
x2 <- runif(n, -2.5, 2.5)
x3 <- runif(n, -2.5, 2.5)

f1 <- x1**2
f2 <- 2 * x2
f3 <- sin(x3)
f1 <- f1 - mean(f1)
f2 <- f2 - mean(f2)
f3 <- f3 - mean(f3)

eta0 <- 2 + f1 + f2 + f3
y <- eta0 + rnorm(n, 0.25)
train <- data.frame(x1, x2, x3, y)

ngam <- neuralGAM(
  y ~ s(x1) + x2 + s(x3), 
  data = train,
  num_units = 1024, family = "gaussian",
  activation = "relu",
  learning_rate = 0.001, bf_threshold = 0.001,
  max_iter_backfitting = 10, max_iter_ls = 10,
  seed = seed
)

summary(ngam)

```

You can then use the `plot` function to visualize the learnt partial effects: 

```
plot(ngam)
```
Or the custom `autoplot` function for more advanced graphics using the ggplot2 library, including Prediction Intervals (if available)

```
autoplot(ngam, select="x1")
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

### Per-term configuration

`neuralGAM` from version 2.0 allows to specify hyperparameters per smooth term inside s(), overriding global defaults, and the `summary()` now print each smooth term configuraion:

```r
ngam <- neuralGAM(
  y ~ s(x1, num_units = 32, activation = "tanh") + s(x2), 
  data = train,
  num_units = 64,  # default for terms without explicit num_units
  seed = seed
)
```
Per-term configuration supports custom initializers and regularizers for both weights and biases, enabling fine control over model complexity and stability. 
For example, you can set one of the neural networks to use L2 regularization and He initialization using Keras functions directly. 

This is useful for:

- Preventing overfitting (L1/L2 regularization)
- Stabilizing training for deep smooth terms (He/Glorot initializers)
- Applying different constraints to specific smooth terms (for example, more complex functions might require more complex / deeper architectures)

```r
ngam <- neuralGAM(
  y ~ s(
         x1, 
         kernel_initializer = keras::initializer_he_normal(),
         bias_initializer   = keras::initializer_zeros(),
         kernel_regularizer = keras::regularizer_l2(0.01),
         bias_regularizer   = keras::regularizer_l1(0.001)
       ) +
       s(x2),
  data = train,
  num_units = 64,
  activation = "relu",
  seed = seed
)
```


The `summary()` now prints each smooth terms configuration and the essential parameters of each network's architecture: 

```r
summary(ngam)
> summary(ngam)
# neuralGAM summary
# ========================================================================
# Family          : gaussian
# Formula         : y ~ s(x1, kernel_initializer = keras::initializer_he_normal(), bias_initializer = keras::initializer_zeros(), kernel_regularizer = keras::regularizer_l2(0.01), bias_regularizer = keras::regularizer_l1(0.001)) + s(x2)
# Observations    : 24500
# Intercept (eta0): 2.2449
# Train MSE       : 1.69926
# Prediction Int. : disabled
# ------------------------------------------------------------------------
# Per-term configuration (parsed from s(...))
#  • x1 — units: 64 | activation: relu | learning rate: 0.001 | k_init: HeNormal(seed=NA) | b_init: Zeros | k_reg: L2 | b_reg: L1 | a_reg: NA
#  • x2 — units: 64 | activation: relu | learning rate: 0.001 | k_init: glorot_normal | b_init: zeros | k_reg: NA | b_reg: NA | a_reg: NA
# ------------------------------------------------------------------------
# Neural network layer configuration per smooth term
#  • x1
#  layer_index class  units    activation kernel_init bias_init kernel_reg bias_reg
#            1     1 linear GlorotUniform       Zeros      <NA>       <NA>        1
#            2    64   relu      HeNormal       Zeros        L2         L1        2
#            3     1 linear GlorotUniform       Zeros      <NA>       <NA>        3
#  • x2
#  layer_index class  units    activation kernel_init bias_init kernel_reg bias_reg
#            1     1 linear GlorotUniform       Zeros      <NA>       <NA>        1
#            2    64   relu  GlorotNormal       Zeros      <NA>       <NA>        2
#            3     1 linear GlorotUniform       Zeros      <NA>       <NA>        3
# ------------------------------------------------------------------------
# Training history (head)
#             Timestamp Model Epoch TrainLoss
# 1 2025-08-11 15:10:16    x1     1   12.8216
# 2 2025-08-11 15:10:18    x2     1    2.3671
# 3 2025-08-11 15:10:19    x1     2    2.9200
# 4 2025-08-11 15:10:21    x2     2    4.0063
# 5 2025-08-11 15:10:22    x1     3    2.6484
# 6 2025-08-11 15:10:24    x2     3    4.9794
```

### Prediction Intervals

Enable predictive intervals by setting `build_pi = TRUE` and specifying a confidence level via `alpha`:

```r
ngam <- neuralGAM(
  y ~ s(x1) + s(x2),
  data = train,
  build_pi = TRUE,
  alpha = 0.95,
  num_units = 1024,
  seed = seed
)
pred <- predict(ngam, newdata = test, type = "response")
head(terms)
#          lwr       upr      fit
# 1 -0.3361656 13.512428  7.2075512
# 2 -1.4482141 12.116602  5.7794316
# 3 -5.5030289  7.775545  1.4813798
# 4 -0.8146979 13.162940  6.7727171
# 5 -6.6520148  6.589276  0.2617679
# 6 -9.5601383  3.261372 -2.7478739
```

### Cross-validation and Training History

You can monitor the validation loss during training using the `validation_split` parameter. You can then visualize the backfitting loss history using the `plot_history()` function. 
The function `plot_history()` will produce faceted loss curves for each neural network (one per non-parametric term), showing how the training and validation loss evolve over backfitting iterations.

```
ngam <- neuralGAM(y ~ s(x1) + x2 + s(x3),
                  data = train,
                  num_units = 1024, family = "gaussian",
                  activation = "relu",
                  learning_rate = 0.001, bf_threshold = 0.001,
                  max_iter_backfitting = 10, max_iter_ls = 10,
                  validation_split = 0.2,
                  seed = seed)

# Plot loss per backfitting iteration
plot_history(ngam)
plot_history(ngam, select = "x1")       # Plot just x1
plot_history(ngam, metric = "val_loss") # Plot only validation loss
```

### Detailed model inspection

The enhanced summary() shows:

- Family, formula, sample size, intercept, MSE
- Per-term hyperparameters (units, activation, learning rate, initializers, regularizers)
- Layer configuration for each Keras model
- Linear coefficients (if a parametric part exists)
- Compact training history

## Citation

If you use neuralGAM in your research, please cite the following paper:

> Ortega-Fernandez, I., Sestelo, M. & Villanueva, N.M. _Explainable generalized additive neural networks with independent neural network training_. Statistics & Computing 34, 6 (2024). https://doi.org/10.1007/s11222-023-10320-5

```
@article{ortega2024explainable,
author = {Ortega-Fernandez, Ines and Sestelo, Marta and Villanueva, Nora M},
doi = {10.1007/s11222-023-10320-5},
issn = {1573-1375},
journal = {Statistics and Computing},
number = {1},
pages = {6},
title = {{Explainable generalized additive neural networks with independent neural network training}},
url = {https://doi.org/10.1007/s11222-023-10320-5},
volume = {34},
year = {2023}
}
```          

