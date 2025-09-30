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
  .disable_tf_logs()  # ensure logs are muted as early as possible

  # If reticulate isn't even available, just inform and exit quietly.
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    packageStartupMessage("NOTE: 'reticulate' not available; Python/TensorFlow will be skipped.")
    return(invisible(FALSE))
  }

  # No conda configured
  if (is.null(conda)) {
    packageStartupMessage(
      "NOTE: conda not found... run 'install_neuralGAM()' and load the package again..."
    )
    return(invisible(FALSE))
  }
  # Query conda envs defensively
  envs <- try(reticulate::conda_list(conda), silent = TRUE)
  if (inherits(envs, "try-error") || is.null(envs) || !"name" %in% names(envs)) {
    packageStartupMessage(
      "NOTE: unable to list conda environments... run 'install_neuralGAM()' and load the package again..."
    )
    return(invisible(FALSE))
  }

  if ("neuralGAM-env" %in% envs$name) {
    i <- which(envs$name == "neuralGAM-env")[1]
    # Point RETICULATE_PYTHON, but do not force-init Python
    if (!is.na(envs$python[i]) && nzchar(envs$python[i])) {
      Sys.setenv(RETICULATE_PYTHON = envs$python[i])
    }

    # Best-effort: don't fail if env can't be activated
    suppressMessages(try(
      reticulate::use_condaenv("neuralGAM-env", conda = conda, required = FALSE),
      silent = TRUE
    ))

    # Re-assert quiet logging after potential Python selection
    .disable_tf_logs()

    # Optionally probe Python config without failing package load
    suppressMessages(try(invisible(reticulate::py_config()), silent = TRUE))
  } else {
    packageStartupMessage(
      "NOTE: conda environment 'neuralGAM-env' not found... run 'install_neuralGAM()' and load the package again..."
    )
  }
  invisible(TRUE)
}

.disable_tf_logs <- function() {
  # 1) Silence logs BEFORE any TF import (works even if TF isn't installed)
  #    0=all, 1=INFO, 2=WARNING, 3=ERROR
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")   # Silence TF C++ logs
  Sys.setenv(ABSL_LOGLEVEL        = "3")   # Silence absl.logging (often used by TF)

  # If reticulate is missing, we can't set Python-side loggers — that's fine.
  if (!requireNamespace("reticulate", quietly = TRUE)) return(invisible(FALSE))

  # 2) Prepare lazy imports with on-load hooks so that when/if TF is imported,
  #    Python-side verbosity is lowered. These calls are safe even if modules
  #    don't exist yet. If Python isn't initialized yet, delay_load prevents hard failures.
  tf_obj <- try(reticulate::import(
    "tensorflow",
    delay_load = list(
      priority = 10,
      on_load = function() {
        # Guard each call; API surface may vary across TF versions
        try(tf$compat$v1$logging$set_verbosity(tf$compat$v1$logging$ERROR), silent = TRUE)
        try(tf$get_logger()$setLevel("ERROR"), silent = TRUE)
        try(tf$autograph$set_verbosity(level = 0L), silent = TRUE)
        invisible(NULL)
      }
    )
  ), silent = TRUE)

  # If TF is already initialized elsewhere, attempt to quiet logs now as well
  # (these will no-op if 'tf' above is only a delay stub)
  suppressMessages(try(tf$compat$v1$logging$set_verbosity(tf$compat$v1$logging$ERROR), silent = TRUE))
  suppressMessages(try(tf$get_logger()$setLevel("ERROR"), silent = TRUE))
  suppressMessages(try(tf$autograph$set_verbosity(level = 0L), silent = TRUE))
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
