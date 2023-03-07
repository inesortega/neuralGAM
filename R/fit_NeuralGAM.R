#' Fit a NeuralGAM model
#'
#' @description Main function to fit a NeuralGAM model. The function builds one
#' neural network to attend to each to each feature in x, using the
#' backfitting and local scoring algorithms to fit a weighted additive model
#' using neural networks as function approximators. The adjustment of the
#' dependent variable and the weights is determined by the distribution of the
#' response \code{"y"} (\code{"gaussian"} or #' \code{"binomial"}).
#'
#' @param x A data frame containing all the covariates.
#' @param y A vector with the response values.
#' @param num_units number of hidden units (for shallow neural networks) or
#' list of hidden units per layer (i.e. \code{"list(32,32,32)"} generates a DNN
#' with three layers and \code{32} neurons per layer).
#' @param learning_rate learning rate for the neural network optimizer
#' @param family A description of the link function used in the model:
#' \code{"gaussian"} or \code{"binomial"}
#' Defaults to \code{"gaussian"}
#' @param w_train optional sample weights.
#' @param bf_threshold convergence criterion of the backfitting algorithm.
#' Defaults to \code{0.00001}
#' @param ls_threshold convergence criterion of the local scoring algorithm.
#' Defaults to \code{0.1}
#' @param max_iter_backfitting an integer with the maximum number of iterations
#' of the backfitting algorithm. Defaults to \code{10}.
#' @param max_iter_ls an integer with the maximum number of iterations of the
#' local scoring Algorithm. Defaults to \code{10}.
#' @return y_hat, partial effects and learned eta
#' @export
#'
#' @import tensorflow
#' @import keras

#' @examples
#'
#' library(NeuralGAM)
#' data(train)
#' head(train)
#' X_train = train[c('X0','X1','X2')]
#' fs_train = train[c('f(X0)','f(X1)','f(X2)')]
#' y_train = train['y']
#'
#' ngam <- fit_NeuralGAM(num_units = 1024, learning_rate = 0.001, x=X_train,
#'               y = y_train, family = "gaussian", bf_threshold=0.00001,
#'               ls_threshold = 0.1, max_iter_backfitting = 10,
#'               max_iter_ls=10)
fit_NeuralGAM <- function(x, y, num_units, learning_rate, family = "gaussian",
                          w_train = NULL, bf_threshold=0.00001,
                          ls_threshold = 0.1, max_iter_backfitting = 10,
                          max_iter_ls=10){

  #Initialization
  converged <- FALSE
  f <- x*0
  g <- x*0

  nvars <- dim(f)[2]

  if (nvars == 0) stop("No terms available")

  it <- 1

  if(is.null(w_train)) w_train <- rep(1, length(y))
  if(family == "gaussian") max_iter_ls <- 1

  model <- list()
  for (k in 1:nvars){
    model[[k]] <- build_feature_NN(num_units, learning_rate)
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
      W <- w_train
    }else{
      der <- diriv(family, muhat)
      Z <- eta + (y - muhat) * der
      W <- weight(w_train, muhat, family)
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
          adam <- optimizer_adam(learning_rate = learning_rate)
          model[[k]] %>% compile(
            loss = 'mean_squared_error',
            optimizer = adam,
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
      g <- f
      eta <- eta0 + rowSums(g)


      #compute the differences in the predictor at each iter
      err <- sum((eta-eta_prev)**2)/sum(eta_prev**2)
      eta_prev <- eta
      print(paste("ITERATION_BACK", it_back, "- Current ERR = ", err, "Threshold = ", bf_threshold))
      print(paste("ERR > bf_threshold ?", (err > bf_threshold)))
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

  yhat <- link(family, eta)
  res <- list(y = yhat, partial = g, eta=eta)
  class(res) <- "NeuralGAM"
  return(res)

}
