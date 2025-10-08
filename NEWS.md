# neuralGAM 2.0.0

* **Major update** with expanded flexibility, improved diagnosis tools, and uncertainty quantification.
* **Additional distribution families**: now supports `poisson` in addition to `gaussian` and `binomial`.
* **Per-term architecture configuration**: hyperparameters (units, activation, learning rate, initializers, regularizers) can now be set per smooth term inside `s()`.  
* **Confidence Intervals (CI)**:  
  - `uncertainty_method` argument allows estimation of *epistemic* uncertainty.  
  - Intervals integrated into `predict()` and `autoplot()`.  
* **Cross-validation support**: new `validation_split` parameter for monitoring validation losses during training.  
* **Training diagnostics**: new `plot_history()` function for visualizing training/validation loss curves per term and per backfitting iteration.  
* **Improved summary()**: displays per-term configuration, layer architectures, linear coefficients, and compact training history.  
* **Diagnosis plots**: new `diagnose()` function which provides a 2×2 diagnostic panel.
* **Autoplot enhancements**: ggplot2-based diagnostic and effect plots with support for CI ribbons, per-term inspection, and factor vs continuous term visualization.  
* **Testing**: expanded test coverage for new families, CI estimation, plotting, and per-term configuration.  
* **Internal refactoring**:  
  - Clean separation of deviance and link functions per family.  
  - Consistent handling of sample weights.  
  - Improved numerical stability (clamping in log/exp/probabilities).
  
# neuralGAM 1.1.1

* `verbose` parameter is now used along all the required functions.

* Tensorflow and Keras are now loaded when `library(neuralGAM)` is invoked for the first time, and therefore the first run of the  `neuralGAM()` function has all the required packages ready.

# neuralGAM 1.1.0

* Initial CRAN submission.
