# neuralGAM 2.0.0

This version introduces major new functionality and internal improvements:

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

## R CMD check results

── R CMD check results ──── neuralGAM 2.0 ────

❯ checking for future file timestamps ... NOTE
  unable to verify current time

0 errors ✔ | 0 warnings ✔ | 1 notes ✖

* Local: `R CMD check` passed on Windows 11 (x86_64, mingw32, R 4.4.1 2024-06-14 ucrt).
* NOTE related to unability of R to verify current time, which is a known issue reported [here](https://forum.posit.co/t/r-devel-r-cmd-check-failing-because-of-time-unable-to-verify-current-time/25589):
* GitHub Actions: [R-CMD-check](https://github.com/inesortega/neuralGAM/actions/workflows/R-CMD-check.yaml) passes on macOS, Windows, and Ubuntu (R-release, R-devel, R-oldrel).
* Coverage: >80% test coverage confirmed with [codecov](https://app.codecov.io/gh/inesortega/neuralGAM).

# neuralGAM 1.1.1

This version fixes a minor issue, regarding the verbosity of the package outputs. Now verbose is considered in all the required function calls.

In this version, the package is loaded without performing the Python dependencies installation, warning the user using a package start-up message that required dependencies are not found. If the required dependencies are found, the Tensorflow library version is checked, so all the dependencies are loaded and the first call to `neuralGAM()` is faster than before. 

# neuralGAM 1.1.0

This version fixes a policy violation regarding writing the package Python dependencies to a non-standard location without the user's consent. 

In this version, the package is loaded without performing the Python dependencies installation, warning the user using a package start-up message that required dependencies are not found. To assist the user with Python dependencies installation, we have included a new function `install_neuralGAM()` which helps the user set up a working Python environment using `miniconda`. 

We have included a instruction on the tests and examples to skip them on CRAN since a working Python installation with the required dependencies is not guaranteed on CRAN machines. Tests have been included for all the functions of the library. 

## CRAN comments after initial submission

> Please add \value to .Rd files regarding exported methods and explain the functions results in the documentation. Please write about the structure of the output (class) and also what the output means. (If a function does not return a value, please document that too, e.g. \value{No return value, called for side effects} or similar)
>Missing Rd-tags:
>     install_neuralGAM.Rd: \value

Added return value to install_neuralGAM.Rd

>\dontrun{} should only be used if the example really cannot be executed (e.g. because of missing additional software, missing API keys, ...) by the user. That's why wrapping examples in \dontrun{} adds the comment ("# Not run:") as a warning for the user. Does not seem necessary. Please replace \dontrun with \donttest.

\dontrun is needed since the library needs additional software (Python dependencies which can be installed using `install_neuralGAM()`), following a similar strategy as other R packages with Python dependencies such as (Keras)[https://github.com/rstudio/keras] and (Tensorflow)[https://github.com/rstudio/tensorflow]. 

> Please unwrap the examples if they are executable in < 5 sec, or replace dontrun{} with \donttest{}.

As in the previous case, examples cannot be unwrapped since additional software is needed to run the tests. 

## Local test execution results

ℹ Testing neuralGAM
✔ | F W S  OK | Context
✔ |         7 | build_feature_NN [14.4s]  
✔ |         6 | dev                       
✔ |         6 | diriv                     
✔ |        35 | formula                   
✔ |         5 | inv_link                  
✔ |         7 | link                      
✔ |        11 | NeuralGAM [6.9s]          
✔ |         7 | weight

[ FAIL 0 | WARN 0 | SKIP 0 | PASS 84 ]

## Local R CMD check results

0 errors ✔ | 0 warnings ✔ | 0 notes ✔

## R-Hub Check for CRAN

Run on the following environments:

* Windows Server 2022, R-devel, 64 bit
* Ubuntu Linux 20.04.1 LTS, R-release, GCC
* Fedora Linux, R-devel, clang, gfortran

There were no errors/warnings.

There were 4 notes:

First one related to the archival of this package:

```
* checking CRAN incoming feasibility ... [6s/21s] NOTE
Maintainer: ‘Ines Ortega-Fernandez <iortega@gradiant.org>’

New submission

Package was archived on CRAN

Possibly misspelled words in DESCRIPTION:
  Hastie (14:81)
  interpretable (14:465, 14:602)
  Interpretable (3:8)
  Tibshirani (14:90)

CRAN repository db overrides:
  X-CRAN-Comment: Archived on 2023-09-10 for policy violation.

  Checking leaves behind 3.6G in the user data directory.
```

Another likely due to a bug/crash in MiKTeX in Windows as noted in [R-hub issue#503](https://github.com/r-hub/rhub/issues/503) and can likely be ignored.
```
* checking for detritus in the temp directory ... NOTE
Found the following files/directories:
  'lastMiKTeXException'
```

The third is related to the [R-hub issue#560](https://github.com/r-hub/rhub/issues/560), and
seems to be an Rhub issue and so can likely be ignored.

```
* checking for non-standard things in the check directory ... NOTE
  ''NULL''
Found the following files/directories:
```

This NOTE is found on the Linux-based distributions, and it is likely an Rhub issue that can be ignored (reported on [R-hub issue#548](https://github.com/r-hub/rhub/issues/548):   

```
* checking HTML version of manual ... NOTE
Skipping checking HTML validation: no command 'tidy' found
```

