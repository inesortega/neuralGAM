#' Fit a NeuralGAM model
#'
#' @description Main function to fit a NeuralGAM model. The function builds one
#' neural network to attend to each feature in x, using the
#' backfitting and local scoring algorithms to fit a weighted additive model
#' using neural networks as function approximators. The adjustment of the
#' dependent variable and the weights is determined by the distribution of the
#' response \code{y}, adjusted by the \code{family} parameter.
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#' @param formula A GAM formula. You can add smooth terms using \code{s()}.
#' @param data A data frame containing the model response variable and covariates
#' required by the formula. Additional terms not present in the formula will be ignored.
#' @param num_units Defines the architecture of each neural network.
#' If a scalar value is provided, a single hidden layer neural network with that number of units is used.
#' If a list of values is provided, a multi-layer neural network with each element of the list defining
#' the number of hidden units on each hidden layer is used.
#' @param family A description of the link function used in the model
#' (defaults to \code{gaussian}). Set \code{family="gaussian"} for linear
#' regression and \code{family="binomial"} for logistic regression.
#' @param learning_rate Learning rate for the neural network optimizer.
#' @param kernel_initializer Kernel initializer for the Dense layers.
#' Defaults to Xavier Initializer (\code{glorot_normal}).
#' @param activation Activation function of the neural network. Defaults to \code{relu}
#' @param kernel_initializer Kernel initializer for the Dense layers.
#' Defaults to Xavier Initializer (\code{glorot_normal}).
#' @param kernel_regularizer Optional regularizer function applied to the kernel weights matrix.
#' @param bias_regularizer Optional regularizer function applied to the bias vector.
#' @param bias_initializer Optional initializer for the bias vector.
#' @param activity_regularizer Optional regularizer function applied to the output of the layer
#' @param bf_threshold Convergence criterion of the backfitting algorithm.
#' Defaults to \code{0.001}
#' @param ls_threshold Convergence criterion of the local scoring algorithm.
#' Defaults to \code{0.1}
#' @param max_iter_backfitting An integer with the maximum number of iterations
#' of the backfitting algorithm. Defaults to \code{10}.
#' @param max_iter_ls An integer with the maximum number of iterations of the
#' local scoring Algorithm. Defaults to \code{10}.
#' @param w_train Optional sample weights
#' @param seed A positive integer which specifies the random number generator
#' seed for algorithms dependent on randomization.
#' @param \ldots Additional parameters for the Adam optimizer (see ?keras::optimizer_adam)
#' @return A trained NeuralGAM object. Use \code{summary(ngam)} to see details.
#' @importFrom keras fit
#' @importFrom keras compile
#' @importFrom tensorflow set_random_seed
#' @importFrom stats predict lm
#' @importFrom reticulate conda_list use_condaenv
#' @importFrom magrittr %>%
#' @importFrom formula.tools lhs rhs
#' @export
#'
#' @examples
#'
#' n <- 24500
#'
#' seed <- 42
#' set.seed(seed)
#'
#' x1 <- runif(n, -2.5, 2.5)
#' x2 <- runif(n, -2.5, 2.5)
#' x3 <- runif(n, -2.5, 2.5)
#'
#' f1 <- x1 ** 2
#' f2 <- 2 * x2
#' f3 <- sin(x3)
#' f1 <- f1 - mean(f1)
#' f2 <- f2 - mean(f2)
#' f3 <- f3 - mean(f3)
#'
#' eta0 <- 2 + f1 + f2 + f3
#' epsilon <- rnorm(n, 0.25)
#' y <- eta0 + epsilon
#' train <- data.frame(x1, x2, x3, y)
#'
#' library(NeuralGAM)
#' ngam <- NeuralGAM(y ~ s(x1) + x2 + s(x3), data = train,
#'                  num_units = 1024, family = "gaussian",
#'                  activation = "relu",
#'                  learning_rate = 0.001, bf_threshold = 0.001,
#'                  max_iter_backfitting = 10, max_iter_ls = 10,
#'                  seed = seed
#'                  )
#'
#' ngam

NeuralGAM <-
  function(formula,
           data,
           num_units,
           family = "gaussian",
           learning_rate = 0.001,
           activation = "relu",
           kernel_initializer = "glorot_normal",
           kernel_regularizer = NULL,
           bias_regularizer = NULL,
           bias_initializer = 'zeros',
           activity_regularizer = NULL,
           w_train = NULL,
           bf_threshold = 0.001,
           ls_threshold = 0.1,
           max_iter_backfitting = 10,
           max_iter_ls = 10,
           seed = NULL,
           ...) {
    formula <- get_formula_elements(formula)

    if (is.null(formula$np_terms)) {
      stop("No smooth terms defined in formula. Use s() to define smooth terms.")
    }

    if (!is.data.frame(data))
      stop("data should be a data.frame")

    if (is.null(num_units))
      stop("num_units should not be null")

    if (!is.numeric(num_units) && !(is.list(num_units))) {
      stop("Argument \"num_units\" must be a positive integer or a list of positive  of integers")
    }

    if (is.numeric(num_units)) {
      if (num_units < 1) {
        stop("Argument \"num_units\" must be a positive integer or a list of positive  of integers")
      }
    }
    if (is.list(num_units)) {
      for (units in num_units) {
        if (!is.numeric(units)) {
          stop("Argument \"num_units\" must be a positive integer or a list of positive integers")
        }
      }
    }

    if (!is.numeric(learning_rate)) {
      stop("learning_rate should be a numeric value")
    }

    if (family != "gaussian" && family != "binomial")
      stop("family must be 'gaussian' or 'binomial'")

    if (!is.null(w_train) && !is.numeric(w_train)) {
      stop("w_train should be a numeric vector")
    }

    if (!is.numeric(bf_threshold))
      stop("bf_threshold should be a numeric value")

    if (!is.numeric(ls_threshold))
      stop("ls_threshold should be a numeric value")

    if (!is.numeric(max_iter_backfitting))
      stop("max_iter_backfitting should be a numeric value")

    if (!is.numeric(max_iter_ls))
      stop("max_iter_ls should be a numeric value")

    if (!is.null(seed) && !is.numeric(seed)) {
      stop("seed should be a positive integer value")
    }

    if (!is.null(seed)) {
      tensorflow::set_random_seed(seed)
    }

    # Initialization
    converged <- FALSE

    n <- nrow(data)
    eta <- rep(0, n)


    # extract x and y from data

    y <- data[[formula$y]]
    x <- data[formula$terms]

    x_np <- data[formula$np_terms]

    f <- g <- data.frame(matrix(0, nrow = nrow(x), ncol = ncol(x)))
    colnames(f) <- colnames(g) <- colnames(x)

    epochs <- c()
    mse <- c()
    timestamp <- c()
    model_i <- c()

    if (dim(f)[2] == 0)
      stop("No terms available")

    it <- 1

    if (is.null(w_train))
      w <- rep(1, length(y))
    if (family == "gaussian")
      max_iter_ls <- 1

    print("Initializing NeuralGAM...")
    model <- list()
    for (k in 1:ncol(x_np)) {
      term <- colnames(x_np)[[k]]
      if (term %in% formula$np_terms) {
        model[[term]] <- build_feature_NN(
          num_units = num_units,
          learning_rate = learning_rate,
          activation = activation,
          kernel_initializer = kernel_initializer,
          kernel_regularizer = kernel_regularizer,
          bias_regularizer = bias_regularizer,
          bias_initializer = bias_initializer,
          activity_regularizer = activity_regularizer,
          name = term,
          ...
        )
      }
    }

    muhat <- mean(y)
    eta <-
      inv_link(family, muhat) #initially estimate eta as the mean of y

    eta_prev <- eta
    dev_new <- dev(muhat, y, family)

    # Start local scoring algorithm
    while (!converged & (it <= max_iter_ls)) {
      if (family == "gaussian") {
        Z <- y
        W <- w
      } else {
        print(paste("ITER LOCAL SCORING", it))
        der <- diriv(family, muhat)
        Z <- eta + (y - muhat) * der
        W <- weight(w, muhat, family)
      }

      # Estimate parametric components
      if (length(formula$p_terms) > 0) {
        parametric <- data.frame(x[formula$p_terms])
        colnames(parametric) <- formula$p_terms
        parametric$y <- Z

        linear_model <- stats::lm(formula$p_formula, parametric)
        eta0 <- linear_model$coefficients["(Intercept)"]
        model[["linear"]] <- linear_model

        # Update eta with parametric component
        f[formula$p_terms] <- predict(linear_model, type = "terms")
        eta <- eta0 + rowSums(f)

      }
      else{
        # if no parametric components, keep the mean of the adjusted dependen var.
        eta0 <- mean(Z)
        eta <- eta0
      }
      eta_prev <- eta

      # Start backfitting  algorithm
      it_back <- 1
      err <- bf_threshold + 0.1 # Force backfitting iteration

      ## Non parametric part -- BF Algorithm to estimate the non-parametric components with NN

      while ((err > bf_threshold) &
             (it_back <= max_iter_backfitting)) {
        for (k in 1:ncol(x_np)) {
          term <- colnames(x_np)[[k]]

          eta <- eta - f[[term]]
          residuals <- Z - eta

          # Fit network k with x[k] towards residuals
          if (family == "gaussian") {
            t <- Sys.time()
            history <-
              model[[term]] %>% fit(x_np[[term]], residuals, epochs = 1)

          } else {
            model[[term]] %>% compile(
              loss = "mean_squared_error",
              optimizer = optimizer_adam(learning_rate = learning_rate, ...),
              loss_weights = list(W)
            )
            t <- Sys.time()
            history <-
              model[[term]] %>% fit(x_np[[term]],
                                    residuals,
                                    epochs = 1,
                                    sample_weight = list(W))
          }

          epochs <- c(epochs, it_back)
          mse <- c(mse, round(history$metrics$loss, 4))
          timestamp <- c(timestamp, format(t, "%Y-%m-%d %H:%M:%S"))
          model_i <- c(model_i, term)

          # Update f with current learned function for predictor k
          f[[term]] <- model[[term]] %>% predict(x_np[[term]])
          f[[term]] <- f[[term]] - mean(f[[term]])
          eta <- eta + f[[term]]

        }

        # update current estimations
        g <- data.frame(f)
        eta <- eta0 + rowSums(g)

        # compute the differences in the predictor at each iteration
        err <- sum((eta - eta_prev) ** 2) / sum(eta_prev ** 2)
        eta_prev <- eta
        print(
          paste(
            "BACKFITTING Iteration",
            it_back,
            "- Current Err = ",
            err,
            "BF Threshold = ",
            bf_threshold,
            "Converged = ",
            err < bf_threshold
          )
        )
        it_back <- it_back + 1

      }

      muhat <- link(family, eta)
      dev_old <- dev_new
      dev_new <- dev(muhat, y, family)

      dev_delta <- abs((dev_old - dev_new) / dev_old)
      if (family == "binomial") {
        if ((dev_delta < ls_threshold) & (it > 0)) {
          converged <- TRUE
        }
      }
      print(
        paste(
          "Current delta ",
          dev_delta,
          " LS Threshold = ",
          ls_threshold,
          "Converged = ",
          dev_delta < ls_threshold
        )
      )
      it <- it + 1
    }

    stats <-
      data.frame(
        Timestamp = timestamp,
        Model = model_i,
        Epoch = epochs,
        MSE = mse
      )

    mse <- mean((y - muhat) ^ 2)

    res <-
      list(
        muhat = muhat,
        partial = g,
        eta = eta,
        x = x,
        model = model,
        eta0 = eta0,
        family = family,
        stats = stats,
        mse = mse,
        formula = formula
      )
    class(res) <- "NeuralGAM"
    return(res)
  }




.onLoad <- function(libname, pkgname) {
  # set conda environment for tensorflow and keras
  paste("Installing requirements....")
  envs <- reticulate::conda_list()
  if (is.element("NeuralGAM-env", envs$name)) {
    if (.isConda()) {
      i <- which(envs$name == "NeuralGAM-env")
      Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
      Sys.setenv(RETICULATE_PYTHON = envs$python[i])
      tryCatch(
        expr = reticulate::use_condaenv("NeuralGAM-env", required = TRUE),
        error = function(e)
          NULL
      )
    }
  } else {
    installNeuralGAMDeps()
    envs <- reticulate::conda_list()
    i <- which(envs$name == "NeuralGAM-env")
    Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
    Sys.setenv(RETICULATE_PYTHON = envs$python[i])
  }
}