#' Internal helper functions to compute MC dropout samples
#' @importFrom stats rnorm
#' @keywords internal

.mc_dropout_forward <- function(model, x, passes, output_dim) {
  if (!is.matrix(x)) x <- as.matrix(x)
  n <- nrow(x)
  out <- array(NA_real_, dim = c(passes, n, output_dim))
  x_tf <- tensorflow::tf$convert_to_tensor(x)
  for (b in seq_len(passes)) {
    y <- model(x_tf, training = TRUE)                # dropout ON
    if (length(dim(y)) == 1L) y <- tensorflow::tf$expand_dims(y, axis = -1L)
    out[b, , ] <- as.array(y)
  }
  out
}

.combine_uncertainties_sampling <- function(lwr_mat, upr_mat, mean_mat,
                                            alpha = 0.05, inner_samples = 50,
                                            centerline = NULL) {
  stopifnot(all(dim(lwr_mat) == dim(upr_mat)),
            all(dim(lwr_mat) == dim(mean_mat)))
  z <- qnorm(1 - alpha/2)
  Tpasses <- nrow(mean_mat); n <- ncol(mean_mat)

  # aleatoric sd per pass/obs
  sd_mat <- pmax((upr_mat - lwr_mat) / (2 * z), 1e-8)

  # sample from the mixture
  # total samples per obs = Tpasses * inner_samples
  # returns [n, T*inner] matrix (but we compute quantiles on the fly to save memory)
  lwr <- numeric(n); upr <- numeric(n); mean_pred <- numeric(n)

  for (i in seq_len(n)) {
    # draw eps for all passes at once
    eps <- matrix(rnorm(Tpasses * inner_samples), nrow = Tpasses)
    y_samps <- mean_mat[, i, drop = TRUE] + sd_mat[, i, drop = TRUE] * eps
    y_samps <- as.vector(y_samps)
    lwr[i]  <- as.numeric(stats::quantile(y_samps, probs = alpha/2, names = FALSE, type = 7))
    upr[i]  <- as.numeric(stats::quantile(y_samps, probs = 1 - alpha/2, names = FALSE, type = 7))
    mean_pred[i] <- if (!is.null(centerline)) centerline[i] else mean(y_samps)
  }

  var_epistemic <- matrixStats::colVars(mean_mat)
  var_aleatoric <- matrixStats::colMeans2(((upr_mat - lwr_mat)/(2*z))^2)

  data.frame(
    fit = mean_pred,
    lwr = lwr,
    upr = upr,
    var_epistemic = var_epistemic,                 # across passes
    var_aleatoric = var_aleatoric,
    var_total     = var_epistemic + var_aleatoric
  )
}
