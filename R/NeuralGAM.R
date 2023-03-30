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
#' required by the formula. Additonal terms not present in the formula will be ignored.
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
#' @param sample_weights Optional sample weights.
#' @param bf_threshold Convergence criterion of the backfitting algorithm.
#' Defaults to \code{0.001}
#' @param ls_threshold Convergence criterion of the local scoring algorithm.
#' Defaults to \code{0.1}
#' @param max_iter_backfitting An integer with the maximum number of iterations
#' of the backfitting algorithm. Defaults to \code{10}.
#' @param max_iter_ls An integer with the maximum number of iterations of the
#' local scoring Algorithm. Defaults to \code{10}.
#' @param ... Other parameters.
#' @return A trained NeuralGAM object. Use \code{summary(ngam)} to see details.
#' @importFrom keras fit
#' @importFrom keras compile
#' @importFrom stats predict
#' @importFrom reticulate conda_list use_condaenv
#' @importFrom magrittr %>%
#' @importFrom formula.tools lhs rhs
#' @export
#'
#' @examples
#'
#' library(NeuralGAM)
#' data(train)
#' head(train)
#'
#' ngam <- NeuralGAM( y ~ X1 + s(X0) + s(X2), data = train,
#' num_units = 1024, family = "gaussian",
#' learning_rate = 0.001, bf_threshold = 0.001,
#' max_iter_backfitting = 10, max_iter_ls = 10
#' )
#'
#' plot(ngam)
#'
#' data(test)
#' X_test <- test[c("X0", "X1", "X2")]
#' # Obtain linear predictor
#' eta <- predict(object = ngam, x = X_test, type = "link")
#' # Obtain each component of the linear predictor separately on each column of a data.frame
#' terms <- predict(object = ngam, x = X_test, type = "terms")

NeuralGAM <- function(formula, data, num_units, family = "gaussian", learning_rate = 0.001,
                          kernel_initializer = "glorot_normal", w_train = NULL,
                          bf_threshold = 0.001, ls_threshold = 0.1,
                          max_iter_backfitting = 10, max_iter_ls = 10, ...) {


  if (!is.data.frame(data)) stop("data should be a data.frame")

  if (is.null(num_units)) stop("num_units should not be null")

  if (family != "gaussian" & family != "binomial") stop("family must be 'gaussian' or 'binomial'")

  if (!is.numeric(learning_rate)) stop("learning_rate should be a numeric value")

  if (!is.null(kernel_initializer)) {
    if (!is.character(kernel_initializer)) stop("kernel_initializer should be a character value")
  }

  if (!is.null(w_train)) {
    if (!is.numeric(w_train)) stop("w_train should be a numeric vector")
  }

  if (!is.numeric(bf_threshold)) stop("bf_threshold should be a numeric value")

  if (!is.numeric(ls_threshold)) stop("ls_threshold should be a numeric value")

  if (!is.numeric(max_iter_backfitting)) stop("max_iter_backfitting should be a numeric value")
  if (!is.numeric(max_iter_ls)) stop("max_iter_ls should be a numeric value")


  library(magrittr)
  library(keras)


  # all vars get.vars(form)

  # Initialization
  converged <- FALSE

  n <- nrow(data)
  eta <- rep(0, n)

  form <- get_formula_elements(formula)

  # extract x and y from data

  y <- data[[form$y]]
  x <- data[form$terms]
  x_p <- data[form$p_terms]
  x_np <- data[form$np_terms]

  f <- g <- data.frame(matrix(0, nrow = nrow(x), ncol = ncol(x)))
  colnames(f) <- colnames(g) <- colnames(x)

  epochs <- c()
  mse <- c()
  timestamp <- c()
  model_i <- c()

  if (dim(f)[2] == 0) stop("No terms available")

  it <- 1

  if (is.null(w_train)) w <- rep(1, length(y))
  if (family == "gaussian") max_iter_ls <- 1

  print("Initializing NeuralGAM...")
  model <- list()
  for (k in 1:ncol(x)) {
    term <- colnames(x)[[k]]
    if(term %in% form$np_terms){
      model[[term]] <- build_feature_NN(num_units = num_units, name=term,
                                     learning_rate = learning_rate, ...)
    }
    if(term %in% form$p_terms){
      model[[term]] <- NULL # will be fitted in LS
    }
  }

  parametric <- data.frame(x[form$p_terms])
  colnames(parametric) <- form$p_terms
  parametric$y <- y

  ## Parametric part -- Use LM to estimate the parametric components
  linear_model <- lm(form$p_formula, parametric)
  muhat <- linear_model$coefficients["(Intercept)"]
  #muhat <- mean(linear_model$fitted.values)

  #muhat <- linear_model$fitted.values
  eta0 <- inv_link(family, muhat)

  model[[term]]

  eta <- eta0
  eta_prev <- eta0
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

    # Start backfitting  algorithm
    it_back <- 1
    err <- bf_threshold + 0.1 # Force backfitting iteration


    for (k in 1:ncol(x_p)){

      term <- colnames(x_p)[[k]]

      eta <- eta - g[[term]]
      residuals <- Z - eta

      lm_formula <- as.formula(paste("residuals ~ ", term))
      lm_data <- data.frame(x_p[, k])
      colnames(lm_data) <- term

      t <- Sys.time()
      model[[term]] <- lm(lm_formula, lm_data)

      # Update g with current learned function for linear predictor
      g[[term]] <- model[[term]]$fitted.values
      g[[term]] <- g[[term]] - mean(g[[term]])
      eta <- eta + g[[term]]

      # add metrics

      epochs <- c(epochs, 1)
      mse <- c(mse, round(mean(model[[term]]$residuals^2), 4))
      timestamp <- c(timestamp, format(t, "%Y-%m-%d %H:%M:%S"))
      model_i <- c(model_i, term)

    }

    # Update current eta with parametric part
    eta <- eta0 + rowSums(g)
    f <- data.frame(g)

    ## Non parametric part -- BF Algorithm to estimate the non-parametric components with NN

    while ((err > bf_threshold) & (it_back <= max_iter_backfitting)) {

      for (k in 1:ncol(x_np)) {

        term <- colnames(x_np)[[k]]

        eta <- eta - f[[term]]
        residuals <- Z - eta

        # Fit network k with x[k] towards residuals
        if (family == "gaussian") {
          t <- Sys.time()
          history <- model[[term]] %>% fit(x_np[[term]], residuals, epochs = 1)

        } else {
          model[[term]] %>% compile(
            loss = "mean_squared_error",
            optimizer = optimizer_adam(learning_rate = learning_rate),
            loss_weights = list(W)
          )
          t <- Sys.time()
          history <- model[[term]] %>% fit(x_np[[term]], residuals, epochs = 1, sample_weight = list(W))
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
      err <- sum((eta - eta_prev)**2) / sum(eta_prev**2)
      eta_prev <- eta
      print(paste("BACKFITTING Iteration", it_back, "- Current Err = ", err, "BF Threshold = ", bf_threshold, "Converged = ", err < bf_threshold))
      it_back <- it_back + 1

    }

    muhat <- link(family, eta)
    dev_old <- dev_new
    dev_new <- dev(muhat, y, family)

    dev_delta <- abs((dev_old - dev_new) / dev_old)

    if ((dev_delta < ls_threshold) & (it > 0)) {
      converged <- TRUE
    }
    it <- it + 1
  }

  stats <- data.frame(Timestamp=timestamp, Model=model_i, Epoch=epochs,MSE=mse)

  mse <- mean((y - muhat)^2)

  res <- list(muhat = muhat, partial = g, eta = eta, x = x, model = model,
              eta0 = eta0, family = family, beta0 = link(family, eta0),
              stats = stats, mse = mse, formula = formula)
  class(res) <- "NeuralGAM"
  return(res)
}


get_formula_elements <- function(formula) {

  # Separate model terms (response, all_terms, smooth_terms)
  y <- formula.tools::lhs(formula)
  all_terms <- all.vars(formula.tools::rhs(formula))
  terms <- formula.tools::rhs(formula)

  smooth_terms <- attr(terms(formula), "term.labels")[grepl("^s", attr(terms(formula), "term.labels"))]
  smooth_formula <- as.formula(paste("y ~ ", paste(smooth_terms, collapse = " + ")))
  smooth_terms <- all.vars(formula.tools::rhs(smooth_formula))

  linear_terms =setdiff(all_terms, smooth_terms)
  linear_formula <- as.formula(paste("y ~ ", paste(linear_terms, collapse = " + ")))

  return(list(y=y, terms=all_terms, np_terms=smooth_terms,
              p_terms =linear_terms, np_formula=smooth_formula,
              p_formula=linear_formula))

}


.onLoad <- function(libname, pkgname) {
  # set conda environment for tensorflow and keras
  envs <- reticulate::conda_list()
  if (is.element("NeuralGAM-env", envs$name)) {
    if (.isConda()) {
      i <- which(envs$name == "NeuralGAM-env")
      Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
      Sys.setenv(RETICULATE_PYTHON = envs$python[i])
      tryCatch(
        expr = reticulate::use_condaenv("NeuralGAM-env", required = TRUE),
        error = function(e) NULL
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

