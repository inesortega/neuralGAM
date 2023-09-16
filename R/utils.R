#' Installs a custom conda environment with tensorflow and keras
#' @description Installs a custom conda environment with tensorflow and keras
#'
#' @importFrom reticulate install_miniconda conda_create
#' py_install conda_binary py_module_available
#' @importFrom keras install_keras
#' @importFrom tensorflow install_tensorflow
#' @keywords internal
installneuralGAMDeps <- function(conda) {
  if (.isMac()) {
    channels <- "apple"
    packageStartupMessage("Adding APPLE channel to conda...")
  } else {
    channel <- NULL
  }

  packageStartupMessage("Installing tensorflow...")
  status4 <- tryCatch(
    tensorflow::install_tensorflow(
      version = "default",
      method = "conda",
      conda = conda,
      envname = "r-reticulate"
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
      version = "default",
      method = "conda",
      conda = conda,
      envname = "r-reticulate"
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

  packageStartupMessage("Installation completed!")
}

.installConda <- function() {
  packageStartupMessage("=== No miniconda detected, installing it using reticulate R package")
  dir <- tools::R_user_dir("neuralGAM", "cache")
  dir.create(dir, recursive=TRUE)
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
  user_dir <- normalizePath(tools::R_user_dir("neuralGAM", "cache"), winslash = "\\", mustWork = NA)
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
  conda_dir <- .getCondaDir()
  conda <- tryCatch(
    reticulate::conda_binary(conda_dir),
    error = function(e)
      NULL
  )
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
