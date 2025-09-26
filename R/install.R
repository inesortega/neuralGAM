#' Install neuralGAM python requirements (with virtualenv)
#' @description
#' Creates a **virtualenv** and installs the Python requirements
#' to run neuralGAM (Python 3.9, TensorFlow and Keras, version 2.15).
#'
#' The virtualenv is created inside the user's cache directory:
#' \code{file.path(tools::R_user_dir('neuralGAM', 'cache'), 'venv', envname)}.
#'
#' On Apple Silicon (arm64), installs \code{tensorflow-macos} (and \code{tensorflow-metal})
#' instead of the standard \code{tensorflow} wheel.
#'
#' @param envname character, name of the virtualenv (default "neuralGAM-venv").
#' @param python_version Optional. Python version to be used (minimum 3.9).
#'        If \code{NULL}, uses \code{Sys.which('python3')} (must be >= 3.9).
#' @param force Force installation and creation of virtualenv (default FALSE).
#' @usage install_neuralGAM(envname = "neuralGAM-venv", python_version = "3.9", force = FALSE)
#' @return NULL
#' @export
#' @importFrom reticulate py_module_available virtualenv_create virtualenv_python use_virtualenv py_config
install_neuralGAM <- function(envname = "neuralGAM-venv", python_version = "3.9", force = FALSE) {
  venv_root <- file.path(tools::R_user_dir("neuralGAM", "cache"), "venv")
  venv_path <- file.path(venv_root, envname)
  if (!dir.exists(venv_root)) dir.create(venv_root, recursive = TRUE, showWarnings = FALSE)

  # 1) Check if python3.9 is available, otherwise install it with reticulate
  python <- .ensure_python(python_version)

  # 2) create the virtualenv
  # 3) Install TensorFlow & Keras 2.15 into this virtualenv
  is_mac_arm <- .isMacARM()

  packageStartupMessage(sprintf("Creating virtualenv at: %s", venv_path))
  if (is_mac_arm) {
    # Apple Silicon: tensorflow-macos + metal plugin
    status_tf <- tryCatch(
      reticulate::virtualenv_create(venv_path, python=python, force = force, packages = c("tensorflow-macos==2.15.*",
                                                                                          "keras==2.15.*",
                                                                                          "tensorflow-metal",
                                                                                          "silence_tensorflow")),
      error = function(e) { packageStartupMessage(e$message); TRUE }
    )
  } else {
    status_tf <- tryCatch(
      reticulate::virtualenv_create(venv_path, python=python, force = TRUE, packages = c("tensorflow==2.15",
                                                                                         "keras==2.15",
                                                                                         "silence_tensorflow")),
      error = function(e) { packageStartupMessage(e$message); TRUE }
    )
  }
  packageStartupMessage("Installation completed! Restarting R session is recommended.")
  invisible(NULL)
}

.setup_virtualenv <- function(envname = "neuralGAM-venv") {
  venv_root <- file.path(tools::R_user_dir("neuralGAM", "cache"), "venv")
  venv_path <- file.path(venv_root, envname)

  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
  if (!dir.exists(venv_path)) {
    packageStartupMessage(paste("NOTE: virtualenv",envname, "not found. Run install_neuralGAM( ) - see ?install_neuralGAM for help on setting up a custom environment"))
    return(invisible(NULL))
  }

  py <- reticulate::virtualenv_python(venv_path)
  if (!nzchar(py)) {
    packageStartupMessage("Could not resolve python in the virtualenv. Reinstall the venv using install_neuralGAM(force=TRUE).")
    return(invisible(NULL))
  }
  # Fast: point directly to python and initialize
  reticulate::use_virtualenv(venv_path, required = TRUE)

  silence <- reticulate::import("silence_tensorflow")
  silence$silence_tensorflow(level="ERROR")

  reticulate::py_config()  # force initialization
  suppressWarnings(tensorflow::tf$config$list_physical_devices("CPU"))  # last check
  invisible(NULL)
}

# ---- helpers (same as your originals) ----
.isMac <- function() {
  sys_info <- Sys.info()
  sys_info[["sysname"]] == "Darwin"
}
.isWindows <- function() {
  sys_info <- Sys.info()
  sys_info[["sysname"]] == "Windows"
}
.isMacARM <- function() {
  sys_info <- Sys.info()
  sys_info[["sysname"]] == "Darwin" && sys_info[["machine"]] == "arm64"
}

# helper to ensure Python at given exists
.ensure_python <- function(version) {
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
  # look through available pythons
  reticulate::use_python_version(version)

  cfg <- reticulate::py_discover_config(required_module = NULL)

  # check if current python matches 3.9
  if (grepl(version, cfg$version)) {
    return(cfg$python)  # good, already a 3.9
  }
  message("No Python 3.9 found; install it via reticulate using reticulate::install_python(version = '3.9') and reload library")
}

.quiet_tf_logs <- function() {
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
  if (!reticulate::py_module_available("tensorflow")) return(invisible())
  tf <- reticulate::import("tensorflow", convert = FALSE)

  # TF 2.x Python logger
  # Try the public API first, fall back to compat.v1
  try(tf$get_logger()$setLevel("ERROR"), silent = TRUE)
  try(tf$compat$v1$logging$set_verbosity(tf$compat$v1$logging$ERROR), silent = TRUE)

  # Optional: mute common Keras/Future warnings
  warnings_mod <- reticulate::import("warnings", convert = FALSE)
  futurewarn   <- reticulate::import("builtins", convert = FALSE)[['FutureWarning']]
  warnings_mod$filterwarnings("ignore", category = futurewarn)
  # Keras deprecation strings
  warnings_mod$filterwarnings("ignore", message = ".*deprecated.*", module = "keras.*")
  invisible()
}
