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

neuralGAM is based on Deep Neural Networks, and depends on Tensorflow and Keras packages. Therefore, a working Python>3.10 installation with those packages installed is required. 

We provide a helper function to get a working python installation from RStudio, which creates a miniconda environment with all the required packages.   

```r
library(neuralGAM)
install_neuralGAM()
```

## Quick start

### Fit a neuralGAM with deep smooth terms

In the following example, we use synthetic data to showcase the performance of neuralGAM by fitting a model with a single layer with 1024 units.

```r
n <- 5000
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
  num_units = 128, family = "gaussian",
  activation = "relu",
  learning_rate = 0.001, bf_threshold = 0.001,
  max_iter_backfitting = 10, max_iter_ls = 10,
  uncertainty_method = "epistemic", forward_passes = 100,
  seed = seed
)

summary(ngam)
```

You can then use the `plot` function to visualize the learnt partial effects: 

```r
plot(ngam)
```
Or the custom `autoplot` function for more advanced graphics using the ggplot2 library, including Confidence / Prediction Intervals (if available)

```r
autoplot(ngam, which="terms", term = "x1", interval = "confidence")
```
To obtain predictions from new data, use the `predict` function: 

```r
n <- 5000
x1 <- runif(n, -2.5, 2.5)
x2 <- runif(n, -2.5, 2.5)
x3 <- runif(n, -2.5, 2.5)

test <- data.frame(x1, x2, x3)

# Obtain linear predictor
eta <- predict(ngam, newdata = test, type = "link")

# Obtain predicted response using se.fit = TRUE to obtain standard errors:
yhat <- predict(ngam, newdata = test, type = "response", se.fit = TRUE)

head(yhat$fit)
head(yhat$se.fit)

# Obtain each component of the linear predictor 
terms <- predict(ngam, newdata = test, type = "terms")

# Obtain only certain terms: 
terms <- predict(ngam, newdata = test, type = "terms", terms = c("x1", "x2"))
```

### Per-term configuration

`neuralGAM` from version 2.0 allows to specify hyperparameters per smooth term inside s(), overriding global defaults, and the `summary()` now print each smooth term configuraion:

```r
ngam <- neuralGAM(
  y ~ s(x1, num_units = 32) + x2 + s(x3, activation = "tanh"), 
  data = train,
  num_units = 64,  # default for terms without explicit num_units
  seed = seed
)
```

Per-term configuration supports custom initializers and regularizers for both weights and biases, enabling fine control over model complexity and stability. 
For example, you can set one of the neural networks to use L2 regularization and He initialization using Keras functions directly (i.e. `keras::regularizer_l1()`). 

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

The `summary()` now prints each smooth terms configuration and the essential parameters of each network's architecture. 

### Prediction Intervals

Enable predictive intervals by setting `uncertainty_method` and specifying a confidence level via `alpha`. For epistemic variance, `forward_passes > 100` is recommended.

```r
ngam <- neuralGAM(
  y ~ s(x1) + s(x2),
  data = train,
  uncertainty_method = "epistemic",
  forward_passes = 100,
  alpha = 0.95,
  num_units = 1024,
  seed = seed
)
pred <- predict(ngam, newdata = test, type = "response")
head(pred)
```

### Cross-validation and Training History

You can monitor the validation loss during training using the `validation_split` parameter. You can then visualize the how the loss evolves per backfitting iteration using the `plot_history()` function. 

```r
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

- Family, formula, sample size, intercept, deviance explained, MSE
- Per-term hyperparameters (units, activation, learning rate, initializers, regularizers)
- Layer configuration for each Keras model
- Linear coefficients (if a parametric part exists)
- Compact training history

Moreover, after fitting a `neuralGAM` model, it is important to evaluate whether the model assumptions are reasonable and whether predictions are well calibrated.  

The helper function `diagnose()` provides a **2×2 diagnostic panel** similar to `gratia::appraise()` for `mgcv` models.

The four panels are:

1. **QQ plot of residuals** (top-left)  
   Compares sample residuals to theoretical quantiles. A straight line indicates a good fit.  
   Deviations suggest skewness, heavy tails, or outliers.

2. **Residuals vs linear predictor η** (top-right)  
   Shows residuals against the fitted linear predictor, with a LOESS smoother.  
   A flat trend near 0 is ideal. Systematic curvature means the model missed a trend; funnel shapes suggest heteroscedasticity.

3. **Histogram of residuals** (bottom-left)  
   Displays the distribution of residuals. Ideally symmetric and centered at 0.  
   Skewness or multimodality may indicate model misspecification.

4. **Observed vs fitted** (bottom-right)  
   Compares predicted values with observed outcomes. For continuous data, points should align with the 45° line.  
   For binary outcomes, this acts as a calibration plot: predicted probabilities should match observed frequencies.

**Notes** 

1. `residual_type` can be `deviance` (default), `pearson`, or `quantile`.
Quantile residuals (Dunn–Smyth) are recommended for discrete families (binomial, Poisson) because they are continuous and approximately normal.

2. `qq_method` controls reference quantiles: 

    - `uniform`: fast and default. 
    - `simulate`: most faithful and provides bands, but slower
    - `normal`: fallback

Together, these diagnosis plots help assess whether residuals behave like noise, whether systematic trends remain and if predictions are unbiased and calibrated. 

## Citation

If you use neuralGAM in your research, please cite the following paper:

> Ortega-Fernandez, I., Sestelo, M. & Villanueva, N.M. _Explainable generalized additive neural networks with independent neural network training_. Statistics & Computing 34, 6 (2024). https://doi.org/10.1007/s11222-023-10320-5

```bibtex
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
