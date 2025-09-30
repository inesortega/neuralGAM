#' Install neuralGAM python requirements
#' @description
#' Creates a conda environment (installing miniconda if required) and set ups the
#' Python requirements to run neuralGAM (Tensorflow and Keras).
#'
#' Miniconda and related environments are generated in the user's cache directory
#' given by:
#'
#' \code{tools::R_user_dir('neuralGAM', 'cache')}
#'
#' @return NULL
#' @export
#' @importFrom reticulate py_module_available conda_binary install_miniconda py_config use_condaenv conda_list conda_create
#' @importFrom tensorflow install_tensorflow
#' @importFrom keras install_keras
install_neuralGAM <- function() {

  conda <- .getConda()

  if(is.null(conda)){
    .installConda()
    conda <- .getConda()
  }

  channel <- NULL
  if(.isMac()){
    channel <- "apple"
  }

  reticulate::conda_create(envname = "neuralGAM-env",
                           conda = conda,
                           python_version = "3.10",
                           channel = channel)

  packageStartupMessage("Installing tensorflow...")
  status4 <- tryCatch(
    tensorflow::install_tensorflow(
      version = "2.15",
      method = "conda",
      conda = conda,
      envname = "neuralGAM-env",
      restart_session = FALSE,
      force = TRUE
    ),
    error = function(e) {
      packageStartupMessage(e)
      return(TRUE)
    }
  )
  if (isTRUE(status4)) {
    stop("Error during tensorflow installation.",
         call. = FALSE)
  }

  packageStartupMessage("Installing keras...")
  status3 <- tryCatch(
    keras::install_keras(
      version = "2.15",
      method = "conda",
      conda = conda,
      envname = "neuralGAM-env",
      force = TRUE
    ),
    error = function(e) {
      packageStartupMessage(e)
      return(TRUE)
    }
  )
  if (isTRUE(status3)) {
    packageStartupMessage(status3)
    stop("Error during keras installation.",
         call. = FALSE)
  }

  packageStartupMessage("Installation completed! Restarting R session...")

}

.setupConda <- function(conda) {
  .disable_tf_logs_env_only()  # keep process quiet as early as possible

  # Never call py_config(), never touch tensorflow::tf here.
  # Only point reticulate to the env WITHOUT requiring activation now.

  if (is.null(conda)) {
    packageStartupMessage(
      "NOTE: conda not found... run 'install_neuralGAM()' and load the package again..."
    )
    return(invisible(FALSE))
  }

  envs <- try(reticulate::conda_list(conda), silent = TRUE)
  if (inherits(envs, "try-error") || is.null(envs) || !"name" %in% names(envs)) {
    packageStartupMessage(
      "NOTE: unable to list conda environments... run 'install_neuralGAM()' and load the package again..."
    )
    return(invisible(FALSE))
  }

  if (!("neuralGAM-env" %in% envs$name)) {
    packageStartupMessage(
      "NOTE: conda environment 'neuralGAM-env' not found... run 'install_neuralGAM()' and load the package again..."
    )
    return(invisible(FALSE))
  }

  i <- which(envs$name == "neuralGAM-env")[1]
  py <- envs$python[i]
  if (is.na(py) || !nzchar(py)) {
    packageStartupMessage("NOTE: 'neuralGAM-env' found but Python path missing. Recreate the env.")
    return(invisible(FALSE))
  }

  # Set preference for later initialization; do NOT initialize now.
  Sys.setenv(RETICULATE_PYTHON = normalizePath(py, winslash = "/", mustWork = FALSE))

  # Suggest to reticulate but DO NOT require (avoids init at load)
  suppressMessages(try(
    reticulate::use_condaenv("neuralGAM-env", conda = conda, required = FALSE),
    silent = TRUE
  ))

  # Do NOT call py_config(); do NOT import TF/Keras here.
  invisible(TRUE)
}

.disable_tf_logs_env_only <- function() {
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")  # 0=all,1=INFO,2=WARNING,3=ERROR
  Sys.setenv(ABSL_LOGLEVEL        = "3")
  Sys.setenv(PYTHONWARNINGS       = "ignore")
  options(keras.view_metrics = FALSE)
  invisible(TRUE)
}

.quiet_python_loggers_if_initialized <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(invisible(FALSE))
  inited <- try(reticulate::py_available(initialize = FALSE), silent = TRUE)
  if (inherits(inited, "try-error") || !isTRUE(inited)) return(invisible(FALSE))

  # Run inside the already-initialized interpreter (no re-init at package load)
  suppressMessages(try(reticulate::py_run_string("
import os, logging, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGLEVEL'] = '3'
warnings.filterwarnings('ignore')
try:
    import tensorflow as tf
    try: tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception: pass
    try: tf.get_logger().setLevel('ERROR')
    except Exception: pass
    try:
        import tensorflow.autograph as ag
        try: ag.set_verbosity(0)
        except Exception: pass
    except Exception: pass
except Exception:
    pass
for name in ('tensorflow', 'absl'):
    try: logging.getLogger(name).setLevel(logging.ERROR)
    except Exception: pass
"), silent = TRUE))
  invisible(TRUE)
}
.installConda <- function() {
  packageStartupMessage("No miniconda detected, installing it using reticulate R package")
  dir <- tools::R_user_dir("neuralGAM", "cache")
  user_dir <- normalizePath(dir, winslash = "\\", mustWork = NA)

  status <- tryCatch(
    reticulate::install_miniconda(path = user_dir),
    error = function(e) {
      packageStartupMessage(e)
      return(TRUE)
    }
  )
  if (isTRUE(status)) {
    stop("Error in Miniconda Installation.", call. = FALSE)
  }

  return(.getConda())
}

.getCondaDir <- function() {
  user_dir <- tools::R_user_dir("neuralGAM", "cache")
  # set up conda_dir according to platform:
  if (.isWindows()) {
    conda_dir <- paste0(user_dir, "/condabin/conda.bat")
  }
  else {
    conda_dir <- paste0(user_dir, "/bin/conda")
  }
  return(conda_dir)
}

.getConda <- function() {

  # Try to find custom conda installation:
  conda_dir <- .getCondaDir()
  conda <- tryCatch(
    reticulate::conda_binary(conda_dir),
    error = function(e)
      NULL
  )
  if(is.null(conda)){
    # Try to obtain default conda installation
    conda <- tryCatch(
      reticulate::conda_binary("auto"),
      error = function(e)
        NULL
    )
  }
  return(conda)
}

.isTensorFlow <- function() {
  tfAvailable <- reticulate::py_module_available("tensorflow")
  if (tfAvailable) {
    tfVersion <- tensorflow::tf$`__version__`
    tfAvailable <- utils::compareVersion("2.2", tfVersion) <= 0
  }
  return(tfAvailable)
}

.isKeras <- function(){
  kerasAvailable <- reticulate::py_module_available("keras")
  return(kerasAvailable)
}

.isMac <- function() {
  sys_info <- Sys.info()
  return(sys_info[["sysname"]] == "Darwin")
}

.isWindows <- function() {
  sys_info <- Sys.info()
  return(sys_info[["sysname"]] == "Windows")
}

.isMacARM <- function() {
  sys_info <- Sys.info()
  return(sys_info[["sysname"]] == "Darwin" &&
           sys_info[["machine"]] == "arm64")
}
