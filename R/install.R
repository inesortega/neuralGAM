#' Install neuralGAM python requirements
#' @param envname name of the environment to be used
#' @return NULL
#' @export
#' @importFrom reticulate virtualenv_create install_python
#' @importFrom tensorflow install_tensorflow
#' @importFrom keras install_keras
install_neuralGAM <-
  function(envname = "r-tensorflow") {
    if(!reticulate::py_available()){
      python_path <- reticulate::install_python()
    }
    reticulate::virtualenv_create(envname, python = "3.9", force = TRUE)

    tensorflow::install_tensorflow(
      version = "default",
      envname = envname,
      force = TRUE
    )

    keras::install_keras(
      version = "default",
      envname = envname,
      force = TRUE
    )

    python <- reticulate::virtualenv_python(envname)
    Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
    Sys.setenv(RETICULATE_PYTHON = python)
    reticulate::use_virtualenv(envname, required = TRUE)
  }
