#' installNeuralGAMDeps
#' @description Installs a custom conda environment with tensorflow and keras
#'
installNeuralGAMDeps <- function() {
  if ((!.isConda())) {
    message("=== No miniconda detected, installing it using reticulate R package")
    if (is.null(miniconda.path)) {
      miniconda.path <- reticulate::miniconda_path()
    }
    status <- tryCatch(
      reticulate::install_miniconda(path = miniconda.path),
      error = function(e) {
        return(TRUE)
      }
    )
    if (isTRUE(status)) {
      stop("Error in Miniconda Installation.", call. = FALSE)
    }
  }
  conda <- reticulate::conda_binary("auto")
  message("Creating environment")

  if (.isMacARM()){
    channels = "apple"
    print("Adding APPLE channel to conda...")
  }
  else{
    channel <- NULL
  }
  status <- tryCatch(
    reticulate::conda_create(
      envname = "NeuralGAM-env",
      packages = "python==3.9",
      channels = channels
    ),
    error = function(e) {
      return(TRUE)
    }
  )
  if (isTRUE(status)) {
    print(status)
    stop("Error in Miniconda Installation.", call. = FALSE)
  }

  if(.isMacARM()){
    # Workaround to install specific versions of tensorflow-macos and tensorflow-metal
    # https://developer.apple.com/forums/thread/721619
    message("Installing tensorflow for MAC ARM")
    status2 <- tryCatch(
      reticulate::py_install(c("tensorflow-deps==2.8.0", "tensorflow-macos==2.8.0",
                               "tensorflow-metal==0.4.0", "tensorflow==2.8.0"),
                             method="conda",
                             conda = conda,
                             envname = "NeuralGAM-env"),
      error = function(e) {
        return(TRUE)
      }
    )
    if (isTRUE(status2)) {
      stop(
        "Error during tensorflow installation.",call. = FALSE)
    }
  }

  message("Installing keras...")
  status3 <- tryCatch(
    keras::install_keras(
      version = "default",
      method = "conda",
      conda = conda,
      envname = "NeuralGAM-env"
    ),
    error = function(e) {
      return(TRUE)
    }
  )
  if (isTRUE(status3)) {
    stop(
      "Error during keras installation.",call. = FALSE)
  }

  message("Installation complete!")
  message(c("Restart R and load NeuralGAM again..."))
}

.isConda <- function() {
  conda <- tryCatch(
    reticulate::conda_binary("auto"), error = function(e) NULL
  )
  !is.null(conda)
}


.isTensorFlow <- function() {
  tfAvailable <- reticulate::py_module_available("tensorflow")
  if (tfAvailable) {
    tfVersion <- tensorflow::tf$`__version__`
    tfAvailable <- utils::compareVersion("2.2", tfVersion) <= 0
  }
  return(tfAvailable)
}

.isMacARM <- function() {
  sys_info <- Sys.info()
  return (sys_info[["sysname"]] == "Darwin" && sys_info[["machine"]] == "arm64")
}

