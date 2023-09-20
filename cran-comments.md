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

