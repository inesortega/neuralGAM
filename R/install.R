#' Install `neuralGAM` python requirements
#' @description
#' Creates a conda environment (installing miniconda if required) and set ups the
#' Python requirements to run `neuralGAM` (Tensorflow and Keras).
#' @return NULL
#' @param force Whether to force the installation of miniconda and dependencies. Deafults to FALSE
#' @examples
#' \dontrun{
#' library(neuralGAM)
#' install_neuralGAM()
#' }
#' @export
#' @importFrom reticulate py_module_available conda_binary install_miniconda use_condaenv conda_list conda_create
#' @importFrom tensorflow install_tensorflow tf
#' @importFrom keras install_keras
install_neuralGAM <- function(force = FALSE) {

  conda <- .getConda()

  if(is.null(conda)){
    .installConda(force)
    conda <- .getConda()
  }

  channel <- NULL
  if(.isMac()){
    channel <- "apple"
  }

  reticulate::conda_create(envname = "neuralGAM-env",
                           channel = channel,
                           conda = conda,
                           python_version = "3.9",
                           force = force)

  packageStartupMessage("Installing tensorflow...")
  status4 <- tryCatch(
    tensorflow::install_tensorflow(
      version = "2.13",
      method = "conda",
      conda = conda,
      envname = "neuralGAM-env",
      restart_session = FALSE,
      force = force
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
      force = force
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

  if(is.null(conda)){
    Sys.setenv(RETICULATE_PYTHON = Sys.which("python"))
    Sys.setenv(RETICULATE_OK = "FALSE")
    packageStartupMessage("NOTE: conda installation not found... run 'install_neuralGAM()' and load library again...")
  }
  else{
    envs <- reticulate::conda_list()
    if("neuralGAM-env" %in% envs$name){
      i <- which(envs$name == "neuralGAM-env")
      Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
      Sys.setenv(RETICULATE_OK = "TRUE")
      Sys.setenv(RETICULATE_PYTHON = envs$python[i])
      reticulate::use_condaenv("neuralGAM-env", required = TRUE)
    }
    else{
      packageStartupMessage("NOTE: conda environment not found... run 'install_neuralGAM()' and load library again...")
    }
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
  conda <- tryCatch(
    reticulate::conda_binary("auto"),
    error = function(e){
      return(NULL)
    }
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
