diagnosis_neuralGAM <- function(object,
                               data = NULL,          # pass if you want to appraise on newdata
                               response = NULL,      # name of response col if data!=NULL (character)
                               family = c("gaussian", "binomial"),
                               weights = NULL,       # optional per-observation weights
                               qq_envelope = TRUE,   # normal-theory envelope via simulation
                               B = 200,              # envelope simulations
                               level = 0.95,         # envelope coverage
                               point_col = "steelblue",
                               point_alpha = 0.5,
                               hist_bins = 30) {
  stopifnot(inherits(object, "neuralGAM"))
  family <- match.arg(family)

  # --- Get y, eta, mu --------------------------------------------------------
  if (is.null(data)) {
    # Predictions for the training data (predict() does this by default)
    eta <- as.numeric(predict(object, type = "link"))
    mu  <- as.numeric(predict(object, type = "response"))
    # Try to find the training response inside the object if present; otherwise ask user
    if (!is.null(object$y)) {
      y <- as.numeric(object$y)
    } else {
      stop("Supply 'data' and 'response' or fit the model with object$y stored.")
    }
  } else {
    stopifnot(!is.null(response), response %in% names(data))
    eta <- as.numeric(predict(object, newdata = data, type = "link"))
    mu  <- as.numeric(predict(object, newdata = data, type = "response"))
    y   <- as.numeric(data[[response]])
  }

  n <- length(y)
  if (!is.null(weights)) {
    stopifnot(length(weights) == n)
    w <- as.numeric(weights)
  } else {
    w <- rep(1, n)
  }

  # --- Deviance residuals (per-family) ---------------------------------------
  .eps <- 1e-12
  dev_resid <- switch(
    family,
    "gaussian" = {
      # For diagnostics, the deviance residual is proportional to ordinary residual.
      # (Dispersion scaling does not affect plots.)
      y - mu
    },
    "binomial" = {
      # Supports {0,1} outcomes (or proportions if you pass 'weights' as trials).
      mu_cl <- pmin(pmax(mu, .eps), 1 - .eps)
      y_cl  <- pmin(pmax(y,  .eps), 1 - .eps)
      # deviance contribution with weights like glm family$dev.resids
      d <- 2 * w * (y_cl * log(y_cl / mu_cl) + (1 - y_cl) * log((1 - y_cl) / (1 - mu_cl)))
      sign(y - mu) * sqrt(pmax(d, 0))
    }
  )

  df <- data.frame(
    y = y,
    mu = mu,
    eta = eta,
    dres = dev_resid
  )

  # --- QQ plot (normal theory) with optional simulation envelope -------------
  # Envelope: simulate N(0, sd(dres)) samples of size n and take pointwise quantiles.
  qq_df <- {
    ord <- order(df$dres)
    r   <- df$dres[ord]
    theo <- qnorm(ppoints(n))
    out <- data.frame(theo = theo, samp = r)

    if (qq_envelope) {
      sd_r <- sd(r, na.rm = TRUE)
      sim  <- replicate(B, sort(rnorm(n, sd = sd_r)))
      lo   <- apply(sim, 1, quantile, probs = (1 - level)/2)
      hi   <- apply(sim, 1, quantile, probs = 1 - (1 - level)/2)
      out$lo <- lo; out$hi <- hi
    }
    out
  }

  suppressPackageStartupMessages({
    library(ggplot2)
    library(patchwork)
  })

  p1 <- ggplot(qq_df, aes(x = theo, y = samp)) +
    geom_point(alpha = point_alpha) +
    { if (qq_envelope) geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, inherit.aes = FALSE) } +
    geom_abline(slope = 1, intercept = 0, linetype = 2) +
    labs(x = "Theoretical N(0,1) quantiles", y = "Deviance residuals",
         title = "QQ plot of deviance residuals")

  # --- Residuals vs linear predictor -----------------------------------------
  p2 <- ggplot(df, aes(x = eta, y = dres)) +
    geom_hline(yintercept = 0, linetype = 2) +
    geom_point(alpha = point_alpha, color = point_col) +
    geom_smooth(method = "loess", formula = y ~ x, se = FALSE) +
    labs(x = "Linear predictor (η)", y = "Deviance residuals",
         title = "Residuals vs linear predictor")

  # --- Histogram of deviance residuals ---------------------------------------
  p3 <- ggplot(df, aes(x = dres)) +
    geom_histogram(bins = hist_bins, fill = point_col, alpha = 0.7) +
    labs(x = "Deviance residuals", y = "Count",
         title = "Histogram of deviance residuals")

  # --- Observed vs fitted (response scale) -----------------------------------
  p4 <- ggplot(df, aes(x = mu, y = y)) +
    geom_point(alpha = point_alpha, color = point_col) +
    geom_abline(slope = 1, intercept = 0, linetype = 2) +
    geom_smooth(method = "loess", formula = y ~ x, se = FALSE) +
    labs(x = "Fitted values (μ)", y = "Observed (y)",
         title = "Observed vs fitted")

  # Arrange 2×2
  (p1 | p2) / (p3 | p4)
}
