#' Installs a custom conda environment with tensorflow and keras
#' @description Installs a custom conda environment with tensorflow and keras
#'
#' @importFrom reticulate miniconda_path install_miniconda conda_create
#' py_install conda_binary py_module_available
#' @importFrom keras install_keras
#' @keywords internal
installneuralGAMDeps <- function() {
  conda <- reticulate::conda_binary("auto")
  if (.isMac()) {
    channels <- "apple"
    packageStartupMessage("Adding APPLE channel to conda...")
  } else {
    channel <- NULL
  }
  packageStartupMessage("Creating neuralGAM-env...")
  status <- tryCatch(
    reticulate::conda_create(
      envname = "neuralGAM-env",
      packages = "python==3.9",
      channels = channels
    ),
    error = function(e) {
      return(TRUE)
    }
  )
  if (isTRUE(status)) {
    packageStartupMessage(status)
    stop("Error in Miniconda Installation.", call. = FALSE)
  }

  if (.isMacARM()) {
    # Workaround to install specific versions of tensorflow-macos and tensorflow-metal
    # https://developer.apple.com/forums/thread/721619
    packageStartupMessage("Installing tensorflow for MAC ARM")
    Sys.setenv(CONDA_SUBDIR = "osx-64")
    status2 <- tryCatch(
      reticulate::py_install(
        c(
          "tensorflow-deps==2.8.0",
          "tensorflow-macos==2.8.0",
          "tensorflow-metal==0.4.0",
          "tensorflow==2.8.0"
        ),
        method = "conda",
        conda = conda,
        envname = "neuralGAM-env"
      ),
      error = function(e) {
        return(TRUE)
      }
    )
    if (isTRUE(status2)) {
      stop("Error during tensorflow installation.",
           call. = FALSE)
    }
  }

  packageStartupMessage("Installing keras...")
  status3 <- tryCatch(
    keras::install_keras(
      version = "default",
      method = "conda",
      conda = conda,
      envname = "neuralGAM-env"
    ),
    error = function(e) {
      return(TRUE)
    }
  )
  if (isTRUE(status3)) {
    stop("Error during keras installation.",
         call. = FALSE)
  }

  packageStartupMessage("Installation complete!")
  packageStartupMessage(c("Restart R and load neuralGAM again..."))
}

.installConda <- function() {
  packageStartupMessage("=== No miniconda detected, installing it using reticulate R package")
  status <- tryCatch(
    reticulate::install_miniconda(path = reticulate::miniconda_path()),
    error = function(e) {
      packageStartupMessage(e)
      return(TRUE)
    }
  )
  if (isTRUE(status)) {
    stop("Error in Miniconda Installation.", call. = FALSE)
  }
}

.isConda <- function() {
  conda <- tryCatch(
    reticulate::conda_binary("auto"),
    error = function(e)
      NULL
  )
  ! is.null(conda)
}

.isTensorFlow <- function() {
  tfAvailable <- reticulate::py_module_available("tensorflow")
  if (tfAvailable) {
    tfVersion <- tensorflow::tf$`__version__`
    tfAvailable <- utils::compareVersion("2.2", tfVersion) <= 0
  }
  return(tfAvailable)
}


.isMac <- function() {
  sys_info <- Sys.info()
  return(sys_info[["sysname"]] == "Darwin")
}

.isMacARM <- function() {
  sys_info <- Sys.info()
  return(sys_info[["sysname"]] == "Darwin" &&
           sys_info[["machine"]] == "arm64")
}
