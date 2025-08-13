#' @title Fit a neuralGAM model
#'
#' @description
#' Fits a Generalized Additive Model where the smooth terms are modeled using `keras` neural networks.
#' The model can optionally output **prediction intervals** (lower bound, upper bound, and mean prediction)
#' using a custom quantile loss (`make_quantile_loss()`), or a standard single-output point prediction
#' using any user-specified loss function.
#'
#' When `build_pi = TRUE`, each smooth term's network outputs three units corresponding to the lower bound,
#' upper bound, and mean prediction, and is compiled with the `make_quantile_loss()` custom loss.
#' The `loss` argument in this case is passed to `mean_loss` inside `make_quantile_loss()` and can be
#' `"mse"`, `"mae"`, or a custom Keras loss function.
#'
#' When `build_pi = FALSE`, each smooth term's network outputs a single unit (point prediction)
#' and uses the `loss` argument directly in `compile()`.
#'
#' @param formula Model formula. Smooth terms must be wrapped in `s(...)`.
#'   You can specify per-term network settings, e.g.:
#'   `y ~ s(x1, num_units = 1024) + s(x3, num_units = c(1024, 512))`.
#' @param data Data frame containing the variables.
#' @param num_units Default hidden layer sizes for smooth terms (integer or vector).
#'   **Mandatory** unless every `s(...)` specifies its own `num_units`.
#' @param family Response distribution: `"gaussian"`, `"binomial"`, `"poisson"`.
#' @param learning_rate Learning rate for Adam optimizer.
#' @param activation Activation function for hidden layers.
#' @param kernel_initializer,bias_initializer Initializers for weights and biases.
#' @param kernel_regularizer,bias_regularizer,activity_regularizer Optional Keras regularizers.
#' @param loss Loss function.
#'   - If `build_pi = FALSE`: used directly for training.
#'   - If `build_pi = TRUE`: must be `"mse"`, `"mae"`, or `"huber"` (applies to mean prediction inside PI loss).
#' @param build_pi Logical. If `TRUE`, trains networks to predict lower bound, upper bound, and mean.
#' @param alpha PI coverage (only used if `build_pi = TRUE`), e.g. `0.95` for 95% PI.
#' @param validation_split Fraction of training data used for validation.
#' @param w_train Optional training weights.
#' @param bf_threshold,ls_threshold Convergence thresholds for backfitting and local scoring.
#' @param max_iter_backfitting,max_iter_ls Maximum iterations for backfitting and local scoring.
#' @param seed Random seed.
#' @param verbose Verbosity: `0` silent, `1` progress messages.
#' @param ... Additional arguments passed to `keras::optimizer_adam()`.
#'
#' @return
#' An object of class `"neuralGAM"`, which is a list containing:
#' \describe{
#'   \item{muhat}{ Numeric vector of fitted mean predictions on the training data.}
#'   \item{partial}{ List of partial contributions \eqn{g_j(x_j)} for each smooth term.}
#'   \item{y}{ Observed response values.}
#'   \item{eta}{ Numeric vector of the linear predictor \eqn{\eta = \eta_0 + \sum_j g_j(x_j)}.}
#'   \item{lwr}{ Numeric vector of lower prediction interval bounds (if `build_pi = TRUE`), otherwise `NULL`.}
#'   \item{upr}{ Numeric vector of upper prediction interval bounds (if `build_pi = TRUE`), otherwise `NULL`.}
#'   \item{x}{ List of model inputs (covariates) used in training.}
#'   \item{model}{L ist of fitted Keras models, one per smooth term (plus `"linear"` if a linear component is present).}
#'   \item{eta0}{ Intercept estimate \eqn{\eta_0}.}
#'   \item{family}{ Model family (`"gaussian"`, `"binomial"`, `"poisson"`).}
#'   \item{stats}{ Data frame of training/validation losses per backfitting iteration.}
#'   \item{mse}{ Training mean squared error.}
#'   \item{formula}{ The original model formula, as parsed by `get_formula_elements()`.}
#'   \item{history}{ List of Keras training histories for each fitted term.}
#'   \item{globals}{ List of global default hyperparameters used for architecture and training.}
#'   \item{alpha}{ PI coverage level (only relevant if `build_pi = TRUE`).}
#'   \item{build_pi}{ Logical; whether the model was trained to produce prediction intervals.}
#' }
#'
#' @details
#' **Defining per-term architectures**
#' You can pass most Keras architecture/training parameters inside each `s(...)` call:
#' ```
#' y ~ s(x1, num_units = 512, activation = "tanh") +
#'     s(x2, num_units = c(256,128), kernel_regularizer = regularizer_l2(1e-4))
#' ```
#' Any term without its own setting will use the global defaults.
#'
#' **Prediction intervals (`build_pi = TRUE`)**
#' - Output: lower bound, upper bound, mean.
#' - Loss: combined quantile loss + mean prediction loss.
#' - `alpha` controls coverage.
#'
#' **Point prediction (`build_pi = FALSE`)**
#' - Output: single value.
#' - Loss: exactly as given in `loss`.
#' @references
#' Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
#' arXiv preprint arXiv:1412.6980.
#' Koenker, R., & Bassett, G. (1978). Regression quantiles.
#' *Econometrica*, 46(1), 33-50.
#' #'
#' @author Ines Ortega-Fernandez, Marta Sestelo.
#'
#' @keywords internal
#'
#' @importFrom keras fit
#' @importFrom keras compile
#' @importFrom tensorflow set_random_seed
#' @importFrom stats predict lm
#' @importFrom reticulate py_available
#' @importFrom magrittr %>%
#' @importFrom formula.tools lhs rhs
#' @export
#' @examples \dontrun{
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
#' library(neuralGAM)
#' # Global architecture
#' ngam <- neuralGAM(
#'   y ~ s(x1) + x2,
#'   data = train,
#'   num_units = 128
#' )
#' ngam
#' # Per-term architecture
#' ngam <- neuralGAM(
#'   y ~ s(x1, num_units = c(128,64), activation = "tanh") +
#'        s(x2, num_units = 256),
#'   data = train
#' )
#' ngam
#' # Construct prediction intervals
#' ngam <- neuralGAM(
#'   y ~ s(x1, num_units = c(128,64), activation = "tanh") +
#'        s(x2, num_units = 256),
#'   data = train,
#'   build_pi = TRUE,
#'   alpha = 0.95
#' )
#' # Visualize point prediction and prediction intervals using autoplot:
#' autoplot(ngam, "x1")
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
           build_pi = FALSE,
           alpha = 0.95,
           validation_split = NULL,
           w_train = NULL,
           bf_threshold = 0.001,
           ls_threshold = 0.1,
           max_iter_backfitting = 10,
           max_iter_ls = 10,
           seed = NULL,
           verbose = 1,
           ...) {


    global_defaults <- list(
      num_units          = num_units,          # still required globally unless every s(...) overrides
      activation         = activation,
      learning_rate      = learning_rate,
      kernel_initializer = kernel_initializer,
      bias_initializer   = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer   = bias_regularizer,
      activity_regularizer = activity_regularizer
    )

    # --- Formula ---
    if (!inherits(formula, "formula")) {
      stop("Argument 'formula' must be a valid R formula object.")
    }

    formula <- get_formula_elements(formula)
    if (is.null(formula$np_terms)) {
      stop("No smooth terms defined in formula. Use s() to define smooth terms.")
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

    # --- Activation & Initializers ---
    if (!is.character(activation) || length(activation) != 1) {
      stop("Argument 'activation' must be a single character string.")
    }
    if (!is.character(kernel_initializer) || length(kernel_initializer) != 1) {
      stop("Argument 'kernel_initializer' must be a single character string.")
    }
    if (!is.character(bias_initializer) || length(bias_initializer) != 1) {
      stop("Argument 'bias_initializer' must be a single character string.")
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

    # --- Loss ---
    if (!is.character(loss) && !is.function(loss)) {
      stop("Argument 'loss' must be a character string (keras built-in) or a custom loss function.")
    }
    if (build_pi) {
      if (is.character(loss)) {
        if (!loss %in% c("mse", "mae", "huber")) {
          stop("When 'build_pi = TRUE', 'loss' must be 'mse', 'mae', or 'huber' to be used in make_quantile_loss().")
        }
      } else {
        stop("When 'build_pi = TRUE', 'loss' must be a supported character string ('mse', 'mae', 'huber').")
      }
    }

    # --- build_pi ---
    if (!is.logical(build_pi) || length(build_pi) != 1) {
      stop("Argument 'build_pi' must be a single logical value (TRUE or FALSE).")
    }

    # --- alpha ---
    if (!is.numeric(alpha) || alpha <= 0 || alpha >= 1) {
      stop("Argument 'alpha' must be a numeric value strictly between 0 and 1.")
    }
    if (build_pi && (alpha < 0.8 || alpha > 0.99)) {
      warning("Alpha values outside 0.8-0.99 may lead to overly narrow or wide prediction intervals.")
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


    if(build_pi == TRUE){
      lwr <- upr <- f
    }

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
        name = term,
        alpha = alpha,
        build_pi = build_pi,
        ...
      )
      model_history[[term]] <- list()
    }

    muhat <- mean(y)
    eta <- inv_link(family, muhat) #initially estimate eta as the mean of y

    eta_prev <- eta
    dev_new <- dev(muhat, y, w, family)

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

        if(build_pi == TRUE){
          # Update yL and yU with parametric component intervals:

          fit <- suppressWarnings(data.frame(predict(linear_model, interval="prediction", level = alpha)))

          # Update prediction intervals
          lwr[formula$p_terms] <- fit$lwr
          upr[formula$p_terms] <- fit$upr
        }

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

          #### Update model and obtain predictions
          t <- Sys.time()
          nonparametric_update = .update_nonparametric_component(model = model, family = family,
                                                                 term = term, eta = eta, f = f, W = W, Z = Z,
                                                                 x_np = x_np,
                                                                 validation_split = validation_split,
                                                                 verbose = verbose,
                                                                 loss = loss, learning_rate = learning_rate,
                                                                 build_pi = build_pi, alpha = alpha,
                                                                 loss_weights = list(W),
                                                                 ...)

          model <- nonparametric_update$model
          history <- nonparametric_update$history
          y_hat <- nonparametric_update$fit

          # Update f with current learned function y_hat for predictor k

          if(build_pi == TRUE){
            f[[term]] <- y_hat$fit # y_hat or mean prediction
            f[[term]] <- f[[term]] - mean(f[[term]])
            eta <- eta + f[[term]]

            # Update prediction intervals
            lwr[[term]] <- y_hat$lwr
            upr[[term]] <- y_hat$upr
          }
          else{
            f[[term]] <- y_hat # y_hat or mean prediction
            f[[term]] <- f[[term]] - mean(f[[term]])
            eta <- eta + f[[term]]
          }

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

      muhat <- link(family, eta)
      dev_old <- dev_new
      dev_new <- dev(muhat, y, w, family)

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
        y = y,
        eta = eta,
        lwr = lwr,
        upr = upr,
        x = x,
        model = model,
        eta0 = eta0,
        family = family,
        stats = stats,
        mse = mean((y - muhat)^2),
        formula = formula,
        history = model_history,
        globals = global_defaults,
        alpha = alpha,
        build_pi = build_pi
      )
    class(res) <- "neuralGAM"
    return(res)
  }

.update_nonparametric_component <- function(model, family, term, eta, f, W, Z, x_np,
                                            validation_split, verbose, loss, learning_rate,
                                            build_pi, loss_weights, alpha,
                                            mc_dropout = TRUE, forward_passes = 15, ...) {
  # Remove the term's current contribution from eta
  eta <- eta - f[[term]]
  residuals <- Z - eta

  # Compile for non-Gaussian families
  if (family == "gaussian") {
    history <- model[[term]] %>% fit(
      x_np[[term]],
      residuals,
      validation_split = validation_split,
      epochs = 1,
      verbose = verbose
    )
  } else {
    model[[term]] <- set_compile(
      model[[term]], build_pi, alpha, learning_rate, loss,
      loss_weights = loss_weights, ...
    )

    history <- model[[term]] %>% fit(
      x_np[[term]],
      residuals,
      epochs = 1,
      sample_weight = list(W),
      verbose = verbose,
      validation_split = validation_split
    )
  }

  # ---- Monte Carlo Dropout forward passes ----
  if (isTRUE(mc_dropout) && forward_passes > 1) {
    if(verbose){
      print("Runing MC...")
    }
    preds_list <- replicate(
      forward_passes,
      {
        # Force dropout active: predict(..., training = TRUE)
        model[[term]] %>% predict(x_np[[term]],
                                  verbose = 0,
                                  training = TRUE)
      },
      simplify = FALSE
    )

    preds_array <- abind::abind(preds_list, along = 3)  # shape: [n_obs, 1, forward_passes]
    y_hat_mean <- apply(preds_array, c(1, 2), mean)
    y_hat_var  <- apply(preds_array, c(1, 2), var)

    # Compute lower and upper bounds of PI using empirical quantiles (non-parametric percentile approach from MC samples):
    # adjust alpha (i.e from 95% desired coverage, we want alpha set to 0.05)
    alpha <- 1 - alpha
    lower_q <- alpha / 2
    upper_q <- 1 - alpha / 2

    ci_lower <- apply(preds_array, 1, quantile, probs = lower_q)
    ci_upper <- apply(preds_array, 1, quantile, probs = upper_q)

    preds <- data.frame(fit = y_hat_mean,
                        lwr = ci_lower,
                        upr = ci_upper)
  } else {
    # Standard deterministic prediction
    preds <- model[[term]] %>% predict(x_np[[term]], verbose = verbose)
    # y_hat_sd <- rep(NA_real_, length(y_hat))
  }

  res = list("model" = model,
             "history" = history,
             "fit" = preds)
  return(res)
}

.merge_term_config <- function(global, per_term) {
  out <- global
  if (length(per_term)) for (nm in names(per_term)) out[[nm]] <- per_term[[nm]]
  out
}

.get_term_config <- function(formula_parsed, term, global_defaults, require_num_units_per_term = FALSE) {
  per_term <- formula_parsed$np_architecture[[term]]
  cfg <- .merge_term_config(global_defaults, per_term)

  if (require_num_units_per_term) {
    if (is.null(per_term$num_units)) {
      stop(sprintf("Missing `num_units` in s(%s, ...) - per-term `num_units` is mandatory.", term))
    }
  } else {
    if (is.null(cfg$num_units))
      stop(sprintf("Provide `num_units` globally or inside s(%s, num_units = ...).", term))
  }

  cfg$kernel_regularizer   <- .coerce_regularizer(cfg$kernel_regularizer)
  cfg$bias_regularizer     <- .coerce_regularizer(cfg$bias_regularizer)
  cfg$activity_regularizer <- .coerce_regularizer(cfg$activity_regularizer)

  cfg
}

.onAttach <- function(libname, pkgname) {
  Sys.unsetenv("RETICULATE_PYTHON")
  .setupConda(.getConda())
}
