# neuralGAM 1.1.0

This version fixes a policy violation regarding writing the package Python dependencies to a non-standard location without the user's consent. 

This version includes a new function `install_neuralGAM()` which helps the user set up a working 
Python environment using miniconda. 

In this version, the package can be loaded without performing the Python dependencies installation, warning the user using a package startup message that required dependencies are not found. 

## Local R CMD check results

0 errors ✔ | 0 warnings ✔ | 0 notes ✔

## R-Hub Check for CRAN

Run on the following environments:

* Windows Server 2022, R-devel, 64 bit
* Fedora Linux, R-devel, clang, gfortran
* 

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

