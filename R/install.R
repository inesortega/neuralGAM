#' Install `neuralGAM` python requirements
#' @description
#' Creates a conda environment (installing miniconda if required) and set ups the
#' Python requirements to run `neuralGAM` (Tensorflow and Keras).
#' @return NULL
#' @export
#' @importFrom reticulate py_available py_module_available conda_binary install_miniconda use_condaenv conda_list conda_create
#' @importFrom tensorflow install_tensorflow tf
#' @importFrom keras install_keras
install_neuralGAM <- function() {

  conda <- .getConda()

  if(is.null(conda)){
    .installConda(force)
    conda <- .getConda()
  }

  packageStartupMessage("Installing tensorflow...")
  status4 <- tryCatch(
    tensorflow::install_tensorflow(
      version = "2.13",
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
  if (isTRUE(status4)) {
    stop("Error during tensorflow installation.",
         call. = FALSE)
  }

  packageStartupMessage("Installing keras...")
  status3 <- tryCatch(
    keras::install_keras(
      version = "2.13",
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

  packageStartupMessage("Installation completed! Load 'library(neuralGAM) again...")

}

.setupConda <- function(conda) {

  if(!is.null(conda)){
    envs <- reticulate::conda_list()
    if("neuralGAM-env" %in% envs$name){
      i <- which(envs$name == "neuralGAM-env")
      Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
      Sys.setenv(RETICULATE_PYTHON = envs$python[i])
      reticulate::use_condaenv("neuralGAM-env", required = TRUE)
    }
  }
  else{
    packageStartupMessage("Conda environment not ready... please run 'install_neuarlGAM()' and reload library")
  }
}

.installConda <- function(force) {
  status <- tryCatch(
    reticulate::install_miniconda(force = force),
    error = function(e) {
      return(TRUE)
    }
  )
  if (isTRUE(status)) {
    stop("Error in Miniconda Installation.", call. = FALSE)
  }

  return(.getConda())
}

.getConda <- function() {
  # Try to obtain default conda installation

  python <- tryCatch(
    expr = reticulate::py_available(initialize = TRUE),
    error = function(e) FALSE
  )
  if(!python){
    return(NULL)
  }

  conda <- tryCatch(
    reticulate::conda_binary("auto"),
    error = function(e){
      return(NULL)
    }
  )
  return(conda)
}

.python_available <- function() {
  tryCatch(
    expr = reticulate::py_available(initialize = TRUE),
    error = function(e) FALSE
  )
}

.isTensorFlow <- function() {
  tfAvailable <- tryCatch(
    expr = reticulate::py_module_available("tensorflow"),
    error = function(e) FALSE
  )
  if (tfAvailable) {
    tfVersion <- tensorflow::tf$`__version__`
    tfAvailable <- utils::compareVersion("2.2", tfVersion) <= 0
  }
  return(tfAvailable)
}

.isKeras <- function(){
  return(tryCatch(
    expr = reticulate::py_module_available("keras"),
    error = function(e) FALSE
  ))
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
