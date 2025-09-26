#' @title Fit a neuralGAM model
#'
#' @description
#' Fits a Generalized Additive Model where smooth terms are modeled by `keras` neural networks.
#' In addition to point predictions, the model can optionally estimate **uncertainty bands** via Monte Carlo Dropout across forward passes.
#'
#' @param formula Model formula. Smooth terms must be wrapped in `s(...)`.
#'   You can specify per-term NN settings, e.g.:
#'   `y ~ s(x1, num_units = 1024) + s(x3, num_units = c(1024, 512))`.
#' @param data Data frame containing the variables.
#' @param num_units Default hidden layer sizes for smooth terms (integer or vector).
#'   **Mandatory** unless every `s(...)` specifies its own `num_units`.
#' @param family Response distribution: `"gaussian"`, `"binomial"`, `"poisson"`.
#' @param learning_rate Learning rate for Adam optimizer.
#' @param activation Activation function for hidden layers. Either a string understood by
#'   `tf$keras$activations$get()` or a function.
#' @param kernel_initializer,bias_initializer Initializers for weights and biases.
#' @param kernel_regularizer,bias_regularizer,activity_regularizer Optional Keras regularizers.
#' @param uncertainty_method Character string indicating the type of uncertainty to estimate.
#'   One of:
#'   \itemize{
#'     \item \code{"none"} (default): no uncertainty estimation.
#'     \item \code{"epistemic"}: MC Dropout for mean uncertainty (CIs)
#'   }
#' @param loss Loss function to use. Can be any Keras built-in (e.g., `"mse"`, `"mae"`,
#'     `"huber"`, `"logcosh"`) or a custom function, passed directly to `keras::compile()`.
#' @param alpha Significance level for prediction intervals, e.g. \code{0.05} for 95% coverage.
#' @param dropout_rate Dropout probability in smooth-term NNs (0,1).
#'   \itemize{
#'     \item During training: acts as a regularizer.
#'     \item During prediction (if \code{uncertainty_method} is "epistemic"): enables MC Dropout sampling.
#'   }
#' @param forward_passes Integer. Number of MC-dropout forward passes used when
#'   \code{uncertainty_method \%in\% c("epistemic","both")}.
#' @param validation_split Optional fraction of training data used for validation.
#' @param w_train Optional training weights.
#' @param bf_threshold Convergence criterion of the backfitting algorithm. Defaults to \code{0.001}
#' @param ls_threshold Convergence criterion of the local scoring algorithm. Defaults to \code{0.1}
#' @param max_iter_backfitting An integer with the maximum number of iterations
#' of the backfitting algorithm. Defaults to \code{10}.
#' @param max_iter_ls An integer with the maximum number of iterations of the local scoring Algorithm. Defaults to \code{10}.
#' @param seed Random seed.
#' @param verbose Verbosity: `0` silent, `1` progress messages.
#' @param ... Additional arguments passed to `keras::optimizer_adam()`.
#'
#' @return An object of class `"neuralGAM"`, a list with elements including:
#' \describe{
#'   \item{muhat}{ Numeric vector of fitted mean predictions (training data).}
#'   \item{partial}{ Data frame of partial contributions \eqn{g_j(x_j)} per smooth term.}
#'   \item{y}{ Observed response values.}
#'   \item{eta}{ Linear predictor \eqn{\eta = \eta_0 + \sum_j g_j(x_j)}.}
#'   \item{lwr,upr}{ Lower/upper confidence interval bounds (response scale)}
#'   \item{x}{ Training covariates (inputs).}
#'   \item{model}{ List of fitted Keras models, one per smooth term (+ `"linear"` if present).}
#'   \item{eta0}{ Intercept estimate \eqn{\eta_0}.}
#'   \item{family}{ Model family.}
#'   \item{stats}{ Data frame of training/validation losses per backfitting iteration.}
#'   \item{mse}{ Training mean squared error.}
#'   \item{formula}{ Parsed model formula (via `get_formula_elements()`).}
#'   \item{history}{ List of Keras training histories per term.}
#'   \item{globals}{ Global hyperparameter defaults.}
#'   \item{alpha}{ PI significance level (if trained with uncertainty).}
#'   \item{build_pi}{ Logical; whether the model was trained with uancertainty estimation enabled}
#'   \item{uncertainty_method}{ Type of predictive uncertainty used ("none","epistemic").}
#'   \item{var_epistemic}{ Matrix of per-term epistemic variances (if computed).}
#' }
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#'
#' @importFrom keras fit
#' @importFrom keras compile
#' @importFrom tensorflow set_random_seed tf
#' @importFrom stats predict lm
#' @importFrom reticulate py_available
#' @importFrom magrittr %>%
#' @importFrom formula.tools lhs rhs
#' @importFrom matrixStats colVars colMeans2
#' @export
#' @examples \dontrun{
#'
#' library(neuralGAM)
#' dat <- sim_neuralGAM_data()
#' train <- dat$train
#' test  <- dat$test
#'
#' # Per-term architecture and confidence intervals
#' ngam <- neuralGAM(
#'   y ~ s(x1, num_units = c(128,64), activation = "tanh") +
#'        s(x2, num_units = 256),
#'   data = train,
#'   uncertainty_method = "epistemic",
#'   forward_passes = 10,
#'   alpha = 0.05
#' )
#' ngam
#' # Visualize point prediction and confidence intervals using autoplot:
#' autoplot(ngam, which = "terms", term = "x1", interval = "confidence")
#' }
neuralGAM <-
  function(formula,
           data,
           family = "gaussian",
           num_units = 64,
           learning_rate = 0.001,
           activation = "relu",
           kernel_initializer = "glorot_normal",
           kernel_regularizer = NULL,
           bias_regularizer = NULL,
           bias_initializer = 'zeros',
           activity_regularizer = NULL,
           loss = "mse",
           uncertainty_method = c("none", "epistemic"),
           alpha = 0.05,
           forward_passes = 100,
           dropout_rate = 0.1,
           validation_split = NULL,
           w_train = NULL,
           bf_threshold = 0.001,
           ls_threshold = 0.1,
           max_iter_backfitting = 10,
           max_iter_ls = 10,
           seed = NULL,
           verbose = 1,
           ...) {

    uncertainty_method <- match.arg(uncertainty_method)

    global_defaults <- list(
      num_units          = num_units,          # still required globally unless every s(...) overrides
      activation         = activation,
      learning_rate      = learning_rate,
      kernel_initializer = kernel_initializer,
      bias_initializer   = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer   = bias_regularizer,
      activity_regularizer = activity_regularizer,
      dropout_rate = dropout_rate,
      loss = loss
    )

    # --- Formula ---
    if (!inherits(formula, "formula")) {
      stop("Argument 'formula' must be a valid R formula object.")
    }

    formula <- get_formula_elements(formula)
    if (is.null(formula$np_terms)) {
      stop("No smooth terms defined in formula. Use s() to define smooth terms.")
    }

    if (!uncertainty_method %in% c("none", "epistemic")) {
      stop("`uncertainty_method` must be one of 'none', 'epistemic'")
    }

    if (uncertainty_method == "none") {
      build_pi <- FALSE
    }
    else{
      build_pi <- TRUE
    }

    if (missing(num_units) || is.null(num_units)) {
      # ensure all np_terms provide num_units in formula
      missing_any <- any(vapply(formula$np_terms, function(t)
        is.null(formula$np_architecture[[t]]$num_units), logical(1)))
      if (missing_any) {
        stop("Provide global `num_units` or specify `num_units` inside each s(...) term.")
      }
    } else if (!is.numeric(num_units)) {
      stop("Argument `num_units` must be numeric (integer or vector).")
    }


    # --- Data ---
    if (!is.data.frame(data)) {
      stop("Argument 'data' must be a data.frame.")
    }

    # --- Family ---
    if (!family %in% c("gaussian", "binomial", "poisson")) {
      stop("Unsupported distribution family. Supported values are 'gaussian', 'binomial', and 'poisson'.")
    }

    # --- Learning rate ---
    if (!is.numeric(learning_rate) || learning_rate <= 0) {
      stop("Argument 'learning_rate' must be a positive numeric value.")
    }


    # --- Regularizers ---
    valid_regularizer <- function(reg) {
      is.null(reg) || inherits(reg, "keras.regularizers.Regularizer")
    }
    if (!valid_regularizer(kernel_regularizer)) {
      stop("Argument 'kernel_regularizer' must be NULL or a valid keras regularizer object.")
    }
    if (!valid_regularizer(bias_regularizer)) {
      stop("Argument 'bias_regularizer' must be NULL or a valid keras regularizer object.")
    }
    if (!valid_regularizer(activity_regularizer)) {
      stop("Argument 'activity_regularizer' must be NULL or a valid keras regularizer object.")
    }

    # --- alpha ---
    if (!is.numeric(alpha) || alpha <= 0 || alpha >= 1) {
      stop("Argument 'alpha' must be a numeric value strictly between 0 and 1.")
    }

    # --- dropout rate ---
    if (!is.numeric(dropout_rate) || dropout_rate <= 0 || dropout_rate >= 1) {
      stop("Argument 'dropout_rate' must be a numeric value strictly between 0 and 1.")
    }

    # --- Validation split ---
    if (!is.null(validation_split) && (!is.numeric(validation_split) || validation_split < 0 || validation_split >= 1)) {
      stop("Argument 'validation_split' must be NULL or a numeric value in (0, 1).")
    }

    # --- Training weights ---
    if (!is.null(w_train) && (!is.numeric(w_train) || length(w_train) != nrow(data))) {
      stop("Argument 'w_train' must be NULL or a numeric vector of length equal to number of observations.")
    }

    # --- Thresholds ---
    if (!is.numeric(bf_threshold) || bf_threshold <= 0) {
      stop("Argument 'bf_threshold' must be a positive numeric value.")
    }
    if (!is.numeric(ls_threshold) || ls_threshold <= 0) {
      stop("Argument 'ls_threshold' must be a positive numeric value.")
    }

    # --- Iterations ---
    if (!is.numeric(max_iter_backfitting) || max_iter_backfitting < 1) {
      stop("Argument 'max_iter_backfitting' must be a positive integer.")
    }
    if (!is.numeric(max_iter_ls) || max_iter_ls < 1) {
      stop("Argument 'max_iter_ls' must be a positive integer.")
    }

    # --- Seed ---
    if (!is.null(seed) && (!is.numeric(seed) || seed %% 1 != 0)) {
      stop("Argument 'seed' must be NULL or an integer.")
    }
    if (!is.null(seed)) {
      tensorflow::set_random_seed(seed)
    }

    # --- Verbosity ---
    if (!is.numeric(verbose) || !verbose %in% c(0, 1)) {
      stop("Argument 'verbose' must be 0 (silent) or 1 (verbose).")
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
    lwr <- upr <- data.frame(matrix(NA, nrow = nrow(x), ncol = ncol(x)))
    colnames(lwr) <- colnames(upr) <- colnames(f) <- colnames(g) <- colnames(x)
    colnames(f) <- colnames(g) <- colnames(x)


    if(build_pi == TRUE){
      lwr <- upr <- f
    }
    # Uncertainty holders on LINK scale (to keep components separate)
    var_epistemic  <- data.frame(matrix(NA_real_, nrow = nrow(x), ncol = ncol(x)))
    var_aleatoric  <- data.frame(matrix(NA_real_, nrow = nrow(x), ncol = ncol(x)))
    var_total      <- data.frame(matrix(NA_real_, nrow = nrow(x), ncol = ncol(x)))
    colnames(var_epistemic) <- colnames(var_aleatoric) <- colnames(var_total) <- colnames(x)

    epochs <- c()
    loss_metric <- c()
    timestamp <- c()
    model_i <- c()
    model_history <- list()

    if (dim(f)[2] == 0)
      stop("No terms available")

    it <- 1

    if (is.null(w_train))
      w <- rep(1, length(y))
    else{
      w <- w_train
    }
    if (family == "gaussian")
      max_iter_ls <- 1

    if (verbose == 1) {
      print("Initializing neuralGAM...")
    }
    model <- list()
    for (term in formula$np_terms) {
      formula$np_architecture[[term]] <- .get_term_config(formula, term, global_defaults, require_num_units_per_term = FALSE)

      model[[term]] <- build_feature_NN(
        num_units = formula$np_architecture[[term]]$num_units,
        learning_rate = formula$np_architecture[[term]]$learning_rate,
        activation = formula$np_architecture[[term]]$activation,
        kernel_initializer = formula$np_architecture[[term]]$kernel_initializer,
        kernel_regularizer = formula$np_architecture[[term]]$kernel_regularizer,
        bias_regularizer = formula$np_architecture[[term]]$bias_regularizer,
        bias_initializer = formula$np_architecture[[term]]$bias_initializer,
        activity_regularizer = formula$np_architecture[[term]]$activity_regularizer,
        loss = formula$np_architecture[[term]]$loss,
        name = term,
        alpha = alpha,
        uncertainty_method = uncertainty_method,
        dropout_rate = dropout_rate,
        ...
      )
      model_history[[term]] <- list()
    }

    muhat <- mean(y)
    eta <- link(family, muhat) #initially estimate eta as the mean of y

    eta_prev <- eta
    dev_new <- dev(muhat = muhat, y = y, family = family, w = w)

    # Start local scoring algorithm
    while (!converged & (it <= max_iter_ls)) {
      if (family == "gaussian") {
        Z <- y
        W <- w
      } else {
        if (verbose == 1) {
          print(paste("ITER LOCAL SCORING", it))
        }
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
        f[formula$p_terms] <- predict(linear_model, type = "terms", verbose = verbose)
        eta <- eta0 + rowSums(f)

        # Do NOT create per-term PIs for parametric effects
        # (prediction intervals are response-level; not meaningful per term)

        # Instead, store epistemic variance for CI of the mean contribution
        # using predict.lm(type="terms", se.fit=TRUE)
        for (tm in formula$p_terms) {
          pr_tm <- stats::predict(linear_model,
                                  newdata = parametric,
                                  type = "terms",
                                  terms = tm,
                                  se.fit = TRUE)
          var_epistemic[[tm]] <- (as.numeric(pr_tm$se.fit))^2
        }

      }
      else{
        # if no parametric components, keep the mean of the adjusted dependent var.
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

          #### Update model and obtain predictions
          t <- Sys.time()
          nonparametric_update = .update_nonparametric_component(model = model, family = family,
                                                                 term = term, eta = eta, f = g, W = W, Z = Z,
                                                                 x_np = x_np,
                                                                 validation_split = validation_split,
                                                                 verbose = verbose,
                                                                 loss = loss, learning_rate = learning_rate,
                                                                 alpha = alpha,
                                                                 loss_weights = list(W),
                                                                 uncertainty_method = uncertainty_method,
                                                                 forward_passes = forward_passes,
                                                                 ...)

          model <- nonparametric_update$model
          history <- nonparametric_update$history
          fit <- nonparametric_update$fit

          # Update f with current learned function for predictor k
          f[[term]] <- fit
          term_center <- vapply(f, mean, numeric(1))

          f[[term]] <- f[[term]] - term_center[[term]]
          eta <- eta + f[[term]]

          ## Store training metrics for current backfitting iteration
          epochs <- c(epochs, it_back)
          loss_metric <- c(loss_metric, round(history$metrics$loss, 4))

          model_history[[term]][[it_back]] <- history

          timestamp <- c(timestamp, format(t, "%Y-%m-%d %H:%M:%S"))
          model_i <- c(model_i, term)

        }

        # update current estimations
        g <- data.frame(f)
        eta <- eta0 + rowSums(g)

        # compute the differences in the predictor at each iteration
        err <- sum((eta - eta_prev) ** 2) / sum(eta_prev ** 2)
        eta_prev <- eta
        if (verbose == 1) {
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
        }
        it_back <- it_back + 1

      }

      muhat <- inv_link(family, eta)
      dev_old <- dev_new
      dev_new <- dev(muhat = muhat, y = y, family = family, w = w)

      dev_delta <- abs((dev_old - dev_new) / dev_old)
      if (family == "binomial") {
        if (verbose == 1) {
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
        }
        if ((dev_delta < ls_threshold) & (it > 0)) {
          converged <- TRUE
        }
      }
      it <- it + 1
    }

    #### Compute uncertainty when all models are fitted
    for (k in 1:ncol(x_np)) {
      term <- colnames(x_np)[[k]]
      mdl <- model[[term]]

      if (verbose == 1) {
        sprintf("Computing CI/PI using uncertainty_method = %s, at alpha = %s", uncertainty_method, alpha)
      }
      preds <- .compute_uncertainty(model = mdl,
                                    x = x[[term]],
                                    uncertainty_method = uncertainty_method, alpha = alpha,
                                    forward_passes = forward_passes)
      if(build_pi == TRUE){
        # Update prediction intervals
        lwr[[term]] <- preds$lwr - term_center[[term]]
        upr[[term]] <- preds$upr - term_center[[term]]
      }
      # store variances (centering does NOT change variance)
      var_epistemic[[term]] <- preds$var_epistemic
      var_aleatoric[[term]] <- preds$var_aleatoric
      var_total[[term]]     <- preds$var_total
    }

    stats <-
      data.frame(
        Timestamp = timestamp,
        Model = model_i,
        Epoch = epochs,
        TrainLoss = loss_metric
      )

    res <-
      list(
        muhat = muhat,
        partial = g,
        term_center = term_center,
        y = y,
        w_train = w,
        var_aleatoric = var_aleatoric,
        var_epistemic = var_epistemic,
        var_total = var_total,
        eta = eta,
        lwr = lwr,
        upr = upr,
        x = data[formula$terms],
        model = model,
        eta0 = eta0,
        family = family,
        stats = stats,
        mse = mean((y - muhat)^2),
        formula = formula,
        history = model_history,
        globals = global_defaults,
        alpha = alpha,
        build_pi = build_pi,
        uncertainty_method = uncertainty_method,
        forward_passes = forward_passes
      )
    class(res) <- "neuralGAM"
    return(res)
  }


.update_nonparametric_component <- function(model, family, term, eta, f, W, Z, x_np,
                                            validation_split, verbose, loss, learning_rate,
                                            loss_weights, alpha,
                                            uncertainty_method,
                                            forward_passes, ...) {
  # Remove the term's current contribution from eta
  eta <- eta - f[[term]]
  residuals <- Z - eta
  # ---- Fit - one epoch ----
  history <- model[[term]] %>% fit(
    x_np[[term]],
    residuals,
    validation_split = validation_split,
    epochs = 1,
    verbose = verbose
  )

  mu_hat <- model[[term]] %>% predict(x_np[[term]], verbose = verbose)

  if(uncertainty_method %in% c("aleatoric", "both")){
    mu_hat <- mu_hat[, 3]  # obtain mean prediction only
  }
  else{
    mu_hat <- as.numeric(mu_hat)
  }

  list(
    model = model,
    history = history,
    fit = mu_hat
  )
}

.onAttach <- function(libname, pkgname) {
  envname <- getOption("neuralGAM.envName")
  if(is.null(envname)){
    envname = "neuralGAM-venv"
  }
  .setup_virtualenv(envname = envname)
}
