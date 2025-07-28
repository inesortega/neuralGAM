
# neuralGAM 2.0

* Additional distribution families. In particular, the `poisson` and `multinomial` families are now supported. 
* Added support for cross validation using the `validation_split` parameter. Moreover, we provide the `plot_history()` function to visualize the training and validation losses at the end of each backfitting iteration. 

# neuralGAM 1.1.1

* `verbose` parameter is now used along all the required functions.

* Tensorflow and Keras are now loaded when `library(neuralGAM)` is invoked for the first time, and therefore the first run of the  `neuralGAM()` function has all the required packages ready.

# neuralGAM 1.1.0

* Initial CRAN submission.
