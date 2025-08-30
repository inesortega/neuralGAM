#' @title Fit a neuralGAM model
#'
#' @description
#' Fits a Generalized Additive Model where the smooth terms are modeled using `keras` neural networks.
#' The model can optionally output **prediction intervals** (lower bound, upper bound, and mean prediction)
#' using a custom quantile loss (`make_quantile_loss()`), or a standard single-output point prediction
#' using any user-specified loss function.
#''
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
#' @param pi_method Character string indicating the type of uncertainty to estimate in prediction intervals.
#'   Must be one of `"none"`, `"aleatoric"`, `"epistemic"`, or `"both"`:
#'   \itemize{
#'     \item \code{"none"}: None (default).
#'     \item \code{"aleatoric"}: Use quantile regression loss to capture data-dependent (heteroscedastic) noise.
#'     \item \code{"epistemic"}: Use MC Dropout with multiple forward passes to capture model uncertainty.
#'     \item \code{"both"}: Combine both quantile estimation and MC Dropout to estimate total predictive uncertainty.
#'   }
#' @param loss Loss function.
#'   - If `pi_method = "none"`: used directly for training.
#'   - Else: must be `"mse"`, `"mae"`, or a custom Keras loss function (applies to mean prediction inside PI loss).
#' @param alpha PI significance level, e.g. `0.05` for 95% PI.
#' @param dropout_rate Numeric in (0,1). Dropout probability applied to hidden layers of each
#'   smooth-term network (when \code{pi_method} is \code{"epistemic"} or \code{"both"}). It serves two purposes:
#'   \itemize{
#'     \item During training: acts as a regularizer to prevent overfitting.
#'     \item During prediction (when \code{pi_method = "epistemic"} or \code{"both"}):
#'           enables Monte Carlo Dropout sampling to approximate epistemic uncertainty.
#'   }
#'   Typical values are between \code{0.1} and \code{0.3}.
#' @param forward_passes Integer, number of forward passess to run MC-dropout when computing
#'   epistemic uncertainty (\code{pi_method = "epistemic"}) or both aleatoric and epistemic.
#' @param inner_samples Integer, number of draws per MC-dropout pass used when combining
#'   aleatoric and epistemic uncertainty (\code{pi_method = "both"}).
#'   For each dropout mask, \code{inner_samples} values are generated from the Normal
#'   approximation defined by the predicted quantile bounds.
#'   Larger values improve stability of the sampled prediction intervals at the cost of speed.
#' @param validation_split Optional fraction of training data used for validation.
#' @param w_train Optional training weights.
#' @param bf_threshold,ls_threshold Convergence thresholds for backfitting and local scoring.
#' @param max_iter_backfitting,max_iter_ls Maximum iterations for backfitting and local scoring.
#' @param seed Random seed.
#' @param verbose Verbosity: `0` silent, `1` progress messages.
#' @param ... Additional arguments passed to `keras::optimizer_adam()`.
#' @return
#' An object of class `"neuralGAM"`, which is a list containing:
#' \describe{
#'   \item{muhat}{ Numeric vector of fitted mean predictions on the training data.}
#'   \item{partial}{ List of partial contributions \eqn{g_j(x_j)} for each smooth term.}
#'   \item{y}{ Observed response values.}
#'   \item{eta}{ Numeric vector of the linear predictor \eqn{\eta = \eta_0 + \sum_j g_j(x_j)}.}
#'   \item{lwr}{ Numeric vector of lower prediction interval bounds if neuralGAM was trained with epistemic/aleatoric unecertainty, otherwise `NULL`.}
#'   \item{upr}{ Numeric vector of upper prediction interval bounds if neuralGAM was trained with epistemic/aleatoric unecertainty, otherwise `NULL`.}
#'   \item{x}{ List of model inputs (covariates) used in training.}
#'   \item{model}{L ist of fitted Keras models, one per smooth term (plus `"linear"` if a linear component is present).}
#'   \item{eta0}{ Intercept estimate \eqn{\eta_0}.}
#'   \item{family}{ Model family (`"gaussian"`, `"binomial"`, `"poisson"`).}
#'   \item{stats}{ Data frame of training/validation losses per backfitting iteration.}
#'   \item{mse}{ Training mean squared error.}
#'   \item{formula}{ The original model formula, as parsed by `get_formula_elements()`.}
#'   \item{history}{ List of Keras training histories for each fitted term.}
#'   \item{globals}{ List of global default hyperparameters used for architecture and training.}
#'   \item{alpha}{ PI significance level (only relevant when model was trained with uncertainty).}
#'   \item{build_pi}{ Logical; whether the model was trained to produce prediction/confidence intervals.}
#'   \item{pi_method}{ Character string: type of predictive uncertainty estimated}
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
#' **Prediction intervals **
#' The package supports three PI mechanisms via `pi_method`:
#' \itemize{
#'   \item \code{"aleatoric"}: per-term networks output \emph{lower}, \emph{upper}, and \emph{mean}
#'         using a combined quantile loss + mean loss. This captures data noise (heteroscedasticity).
#'   \item \code{"epistemic"}: per-term networks output a single head for the mean; epistemic
#'         uncertainty is obtained by Monte Carlo (MC) Dropout at prediction time. The line is the
#'         deterministic prediction (dropout \emph{off}); the interval comes from empirical
#'         quantiles across many stochastic forward passes (dropout \emph{on}).
#'   \item \code{"both"}: combines aleatoric and epistemic by running MC Dropout with the
#'         3-output (lower/upper/mean) head and combining them via variance decomposition.
#' }
#' **Centering for partial effects**
#' For identifiability, each smooth term \eqn{g_j(x_j)} is mean-centered after fitting. When plotting
#' partial effects (e.g., `autoplot(ngam, type = "terms", term = "x1")`), the associated PI bounds are shifted by the \emph{same}
#' centering constant so that the band and the smooth share the same baseline. (Widths/variances are
#' unaffected by this shift.) Full-model predictions on the response scale are not centered.
#'
#' **MC Dropout controls**
#' \itemize{
#'   \item \code{dropout_rate}: probability of dropping units in hidden layers. Used as a regularizer
#'         during training and \emph{reused} at prediction time to approximate epistemic uncertainty.
#'         Practical values are in \code{[0.1, 0.3]}.
#'   \item \code{forward_passes}: number of stochastic forward passes with dropout \emph{on} when
#'         \code{pi_method = "epistemic"} or \code{"both"}. Larger values yield smoother, more stable
#'         envelopes (e.g., 300-1000).
#'   \item \code{inner_samples}: only used for \code{pi_method = "both"}. For each dropout pass, the
#'         lower/upper quantiles define a local Normal approximation from which \code{inner_samples}
#'         draws are taken; final PIs are empirical quantiles of all draws across passes.
#' }
#'
#' **Losses**
#' - Aleatoric: lower/upper heads use the pinball (quantile) loss at \eqn{\alpha/2} and \eqn{1-\alpha/2};
#'   the mean head uses the user-chosen mean loss (\code{"mse"} or \code{"mae"}).
#' - Epistemic: any mean loss for the single-output head; uncertainty comes from MC Dropout only.
#' - Both: quantile + mean losses (as in aleatoric) and MC Dropout; PIs are built by variance decomposition.
#'
#' **Coverage control**
#' `alpha` sets the nominal coverage (e.g., `alpha = 0.05` for 95% PIs). If empirical coverage on a
#' validation split deviates from target, a simple global scaling of the half-width (conformal-style
#' calibration) can be applied post hoc.
#'
#' **Point prediction (`pi_method == "none"`)**
#' - Output: single value per term (no intervals).
#' - Loss: exactly as given in `loss`.
#' @references
#' Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
#' arXiv preprint arXiv:1412.6980.
#' Koenker, R., & Bassett, G. (1978). Regression quantiles.
#' *Econometrica*, 46(1), 33-50.
#' #'
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
#'   y ~ s(x1) + x2,
#'   num_units = 128,
#'   data = train,
#'   pi_method = "aleatoric",
#'   alpha = 0.05
#' )
#' # Visualize point prediction and prediction intervals using autoplot:
#' autoplot(ngam, which = "terms", term = "x1", intervals = "prediction")
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
           pi_method = c("none", "aleatoric", "epistemic", "both"),
           alpha = 0.05,
           forward_passes = 100,
           dropout_rate = 0.1,
           inner_samples = 20,
           validation_split = NULL,
           w_train = NULL,
           bf_threshold = 0.001,
           ls_threshold = 0.1,
           max_iter_backfitting = 10,
           max_iter_ls = 10,
           seed = NULL,
           verbose = 1,
           ...) {

    pi_method <- match.arg(pi_method)

    global_defaults <- list(
      num_units          = num_units,          # still required globally unless every s(...) overrides
      activation         = activation,
      learning_rate      = learning_rate,
      kernel_initializer = kernel_initializer,
      bias_initializer   = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer   = bias_regularizer,
      activity_regularizer = activity_regularizer,
      dropout_rate = dropout_rate
    )

    # --- Formula ---
    if (!inherits(formula, "formula")) {
      stop("Argument 'formula' must be a valid R formula object.")
    }

    formula <- get_formula_elements(formula)
    if (is.null(formula$np_terms)) {
      stop("No smooth terms defined in formula. Use s() to define smooth terms.")
    }

    if (!pi_method %in% c("none", "aleatoric", "epistemic", "both")) {
      stop("`pi_method` must be one of 'none', 'aleatoric', 'epistemic', or 'both'")
    }

    if (pi_method == "none") {
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
        if (!loss %in% c("mse", "mae")) {
          stop("When requesting PI/CI intervals, 'loss' must be 'mse' or 'mae' to be used in make_quantile_loss().")
        }
      } else {
        stop("When requesting PI/CI intervals, 'loss' must be a supported character string ('mse', 'mae').")
      }
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
        name = term,
        alpha = alpha,
        pi_method = pi_method,
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
                                                                 term = term, eta = eta, f = f, W = W, Z = Z,
                                                                 x_np = x_np,
                                                                 validation_split = validation_split,
                                                                 verbose = verbose,
                                                                 loss = loss, learning_rate = learning_rate,
                                                                 alpha = alpha,
                                                                 loss_weights = list(W),
                                                                 pi_method = pi_method,
                                                                 forward_passes = forward_passes, inner_samples = inner_samples,
                                                                 ...)

          model <- nonparametric_update$model
          history <- nonparametric_update$history
          fit <- nonparametric_update$fit

          # Update f with current learned function for predictor k
          f[[term]] <- fit
          mean_val <- mean(f[[term]])
          f[[term]] <- f[[term]] - mean_val
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
      mean_val <- mean(g[[term]])

      if (verbose == 1) {
        sprintf("Computing CI/PI using pi_method = %s, at α = %s", pi_method, alpha)
      }
      preds <- .compute_uncertainty(model = mdl,
                                    x = x[[term]],
                                    pi_method = pi_method, alpha = alpha,
                                    forward_passes = forward_passes,
                                    inner_samples = inner_samples)
      if(build_pi == TRUE){
        # Update prediction intervals
        lwr[[term]] <- preds$lwr - mean_val
        upr[[term]] <- preds$upr - mean_val
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
        pi_method = pi_method,
        forward_passes = forward_passes
      )
    class(res) <- "neuralGAM"
    return(res)
  }


.update_nonparametric_component <- function(model, family, term, eta, f, W, Z, x_np,
                                            validation_split, verbose, loss, learning_rate,
                                            loss_weights, alpha,
                                            pi_method,
                                            forward_passes, inner_samples, ...) {
  # Remove the term's current contribution from eta
  eta <- eta - f[[term]]
  residuals <- Z - eta
  # ---- Fit - one epoch ----
  if (family == "gaussian") {
    history <- model[[term]] %>% fit(
      x_np[[term]],
      residuals,
      validation_split = validation_split,
      epochs = 1,
      verbose = verbose
    )
  } else {
    # Compile for non-gaussian families
    model[[term]] <- set_compile(
      model[[term]], pi_method, alpha, learning_rate, loss,
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

  mu_hat <- model[[term]] %>% predict(x_np[[term]], verbose = verbose)

  if(pi_method %in% c("aleatoric", "both")){
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
  Sys.unsetenv("RETICULATE_PYTHON")
  .setupConda(.getConda())
}
