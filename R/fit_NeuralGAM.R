#' Fit a NeuralGAM model
#'
#' @description Main function to fit a NeuralGAM model. The function builds one
#' neural network to attend to each to each feature in x, using the
#' backfitting and local scoring algorithms to fit a weighted additive model
#' using neural networks as function approximators. The adjustment of the
#' dependent variable and the weights is determined by the distribution of the
#' response \code{y} (\code{gaussian} or \code{binomial}).
#' @author Ines Ortega-Fernandez, Marta Sestelo and Nora M. Villanueva.
#' @param x A data frame containing all the covariates.
#' @param y A numeric vector with the response values.
#' @param num_units defines the architecture of each neural network:
#'  * `numeric` value: for shallow neural networks, number of hidden units.
#'  * `list()` of `numeric` values: number of hidden units on each layer
#'  (i.e. \code{list(32,32,32)} generates a DNN
#'  with three layers and \code{32} neurons per layer).
#' @param family A description of the link function used in the model
#' (defaults to \code{gaussian})
#'  * `gaussian` for linear regression.
#'  * `binomial` for logistic regression.
#' @param learning_rate learning rate for the neural network optimizer.
#' @param kernel_initializer kernel initializer for the Dense layers.
#' Defaults to Xavier Initializer (\code{glorot_normal}).
#' @param w_train optional sample weights.
#' @param bf_threshold convergence criterion of the backfitting algorithm.
#' Defaults to \code{0.001}
#' @param ls_threshold convergence criterion of the local scoring algorithm.
#' Defaults to \code{0.1}
#' @param max_iter_backfitting an integer with the maximum number of iterations
#' of the backfitting algorithm. Defaults to \code{10}.
#' @param max_iter_ls an integer with the maximum number of iterations of the
#' local scoring Algorithm. Defaults to \code{10}.
#' @param \ldots Other parameters.
#' @return NeuralGAM object. See  \code{summary(ngam)} for details
#' @export
#'
#' @examples
#'
#' library(NeuralGAM)
#' data(train)
#' head(train)
#' X_train <- train[c('X0','X1','X2')]
#' y_train <- train$y
#'
#' ngam <- fit_NeuralGAM(x=X_train, y = y_train, num_units = 1024,family = "gaussian",
#'                       learning_rate = 0.001, bf_threshold=0.001,
#'                       max_iter_backfitting = 10,max_iter_ls=10)
#'
fit_NeuralGAM <- function(x, y, num_units, family = "gaussian", learning_rate = 0.001,
                          kernel_initializer = "glorot_normal", w_train = NULL,
                          bf_threshold=0.001, ls_threshold = 0.1,
                          max_iter_backfitting = 10,max_iter_ls=10, ...){

  #Initialization
  converged <- FALSE
  f <- x*0
  g <- x*0

  if(is.data.frame(y)){
    y <- as.numeric(y)
  }

  nvars <- dim(f)[2]

  if (nvars == 0) stop("No terms available")

  it <- 1

  if(is.null(w_train)) w <- rep(1, length(y))
  if(family == "gaussian") max_iter_ls <- 1

  print("Initializing NeuralGAM...")

  model <- list()
  for (k in 1:nvars){
    model[[k]] <- build_feature_NN(num_units, learning_rate, ...)
    print(paste("Model ", k ," Summary "))
    model[[k]] %>% summary()
  }


  muhat <- mean(y)
  eta0 <- inv_link(family, muhat)

  eta <- eta0 #initial estimation as the mean of y
  eta_prev <- eta0

  dev_new <- deviance(muhat, y, family)

  hat <- list()

  # Start local scoring algorithm
  while (!converged & (it <= max_iter_ls)) {
    print(paste("ITER LOCAL SCORING", it))

    if(family == "gaussian"){
      Z <- y
      W <- w
    }else{
      der <- diriv(family, muhat)
      Z <- eta + (y - muhat) * der
      W <- weight(w, muhat, family)
    }

    # Start backfitting  algorithm
    it_back <- 1
    err <- bf_threshold + 0.1 #Force backfitting iteration

    while ((err > bf_threshold) & (it_back <= max_iter_backfitting)) {
      for (k in 1:nvars){

        eta <- eta - f[, k]
        residuals <- Z - eta

        # Fit network k with x[k] towards residuals
        if(family == "gaussian"){
          hat[[k]] <- model[[k]] %>% fit(x[, k], residuals, epochs = 1)
        }else{
          model[[k]] %>% compile(
            loss = 'mean_squared_error',
            optimizer = optimizer_adam(learning_rate = learning_rate),
            loss_weights = list(W)
          )
          hat[[k]] <- model[[k]] %>% fit(x[, k], residuals, epochs = 1, sample_weight = list(W))
        }

        # Update f with current learned function for predictor k
        f[, k] <- model[[k]] %>% predict(x[, k])
        f[, k] <- f[, k] - mean(f[, k])
        eta <- eta + f[, k]
      }

      # update current estimations
      g <- data.frame(f)
      eta <- eta0 + rowSums(g)


      #compute the differences in the predictor at each iteration
      err <- sum((eta - eta_prev)**2)/sum(eta_prev**2)
      eta_prev <- eta
      print(paste("BACKFITTING Iteration", it_back, "- Current Err = ", err, "BF Threshold = ", bf_threshold, "Converged = ", err < bf_threshold))
      it_back <- it_back + 1

    }

    muhat <- link(family, eta)
    dev_old <- dev_new
    dev_new <- deviance(muhat, y, family)

    dev_delta <- abs((dev_old-dev_new)/dev_old)

    print(paste("ITERATION_LOCAL_SCORING", it, dev_delta))
    if ((dev_delta < ls_threshold)&(it > 0)){
      print("Z and f(x) converged...")
      converged <- TRUE
    }
    it <- it + 1

  }
  res <- list(muhat = muhat, partial = g, eta=eta, x=x, model=model, eta0=eta0)
  class(res) <- "NeuralGAM"
  return(res)

}


.onLoad <- function(libname, pkgname) {
  # set conda environment for tensorflow and keras
  envs <- reticulate::conda_list()
  if (is.element("NeuralGAM-env", envs$name)){
    if (.isConda()) {
      print("Loading environment...")
      i <- which(envs$name == "NeuralGAM-env")
      tryCatch(
        expr = reticulate::use_condaenv("NeuralGAM-env", required = TRUE),
        error = function(e) NULL
      )
    }
    Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
    Sys.setenv(RETICULATE_PYTHON = envs$python[i])
  }
  else{
    print("Set-up environment NeuralGAM-env not found...")
    installNeuralGAMDeps()
    envs <- reticulate::conda_list()
    i <- which(envs$name == "NeuralGAM-env")
    Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
    Sys.setenv(RETICULATE_PYTHON = envs$python[i])

  }


}
