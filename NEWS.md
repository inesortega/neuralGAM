
# neuralGAM 2.0


* Adds support for additional distribution family **Poisson**.
* New **per-term architecture configuration**: hyperparameters (units, activation, initializers, regularizers) can be specified inside `s()`, overriding global defaults.
* Adds **confidence intervals ** for fitted models, available for all supported families via `uncertainty_method`. Supports epistemic uncertainty estimation via MC Dropout. 
* Adds **cross-validation support** with `validation_split` and a new helper `plot_history()` function to visualize training and validation losses per backfitting iteration.
* Improves **summary()** with per-term architecture details, layer configuration, linear coefficients, and compact training history.
* Adds deviance explained by the model in `summary()` and `print()`. 
* Enhances **autoplot()**: ggplot2-based diagnostic/effect plots with support for confidence intervals, continuous vs. factor terms, and response / link visualization.
* New **diagnosis plots** via `diagnose()` to evaluate fitted models using QQ plots, residual histogram, residuals vs linear predictor and observed vs fitted values.  
* Internal refactoring for family-specific deviance/link functions, consistent handling of sample weights, and improved numerical stability (clamping in log/exp/probabilities).
* Expanded **test coverage** for new features: confidence intervals, Poisson/multinomial families, cross-validation, plotting, and per-term configs achieving a 80% test coverage. 

# neuralGAM 1.1.1

* `verbose` parameter is now used along all the required functions.

* Tensorflow and Keras are now loaded when `library(neuralGAM)` is invoked for the first time, and therefore the first run of the  `neuralGAM()` function has all the required packages ready.

# neuralGAM 1.1.0

* Initial CRAN submission.
