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

There were 3 notes:

First one related to the archival of this package:

```

```

Another likely due to a bug/crash in MiKTeX as noted in [R-hub issue#503](https://github.com/r-hub/rhub/issues/503) and can likely be ignored.
```

```

The third is related to the [R-hub issue#560](https://github.com/r-hub/rhub/issues/560), and
seems to be an Rhub issue and so can likely be ignored.

```

```

This NOTE is found on the Linux-based distributions, and it is likely an Rhub issue that can be ignored (reported on [R-hub issue#548](https://github.com/r-hub/rhub/issues/548):   

```
* checking HTML version of manual ...
  NOTE
Skipping checking HTML validation: no command
  'tidy' found
```

