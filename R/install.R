#' Install `neuralGAM` python requirements
#' @description
#' Creates a conda environment (installing miniconda if required) and set ups the
#' Python requirements to run `neuralGAM` (Tensorflow and Keras).
#'
#' Miniconda and related environments are generated in the user's cache directory
#' given by `tools::R_user_dir('neuralGAM', 'cache')`.
#' @return NULL
#' @examples
#' \dontrun{
#' library(neuralGAM)
#' install_neuralGAM()
#' }
#' @export
#' @importFrom reticulate py_module_available conda_binary install_miniconda use_condaenv conda_list conda_create
#' @importFrom tensorflow install_tensorflow tf
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
                           channel = channel,
                           conda = conda,
                           python_version = "3.9")

  packageStartupMessage("Installing tensorflow...")
  status4 <- tryCatch(
    tensorflow::install_tensorflow(
      version = "2.13",
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
      version = "2.13",
      method = "conda",
      conda = conda,
      envname = "neuralGAM-env",
      force = TRUE,

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
    Sys.setenv(RETICULATE_OK = FALSE)
    packageStartupMessage("NOTE: conda installation not found... run 'install_neuralGAM()' and load library again...")
  }
  else{
    envs <- reticulate::conda_list(conda)
    if("neuralGAM-env" %in% envs$name){
      i <- which(envs$name == "neuralGAM-env")
      Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
      Sys.setenv(RETICULATE_PYTHON = envs$python[i])
      reticulate::use_condaenv("neuralGAM-env", conda = conda, required = TRUE)
      library(reticulate)
    }
    else{
      packageStartupMessage("NOTE: conda environment not found... run 'install_neuralGAM()' and load library again...")
    }
  }

}
.installConda <- function() {
  packageStartupMessage("Installing miniconda using reticulate R package")
  dir <- tools::R_user_dir("neuralGAM", "cache")
  user_dir <- normalizePath(dir, winslash = "\\", mustWork = NA)

  status <- tryCatch(
    reticulate::install_miniconda(path = user_dir),
    error = function(e) {
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
    error = function(e){
      return(NULL)
    }

  )
  if(is.null(conda)){
    # Try to obtain default conda installation
    conda <- tryCatch(
      reticulate::conda_binary("auto"),
      error = function(e){
        return(NULL)
      }
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
