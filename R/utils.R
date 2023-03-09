

#' installPythonDeps
#'
#' @param conda
#' @param miniconda.path
#'
#' @return
#' @export
#'
#' @examples
installPythonDeps <- function(
    conda = "auto",
    miniconda.path = NULL
) {
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
  status <- tryCatch(
    reticulate::conda_create(
      envname = "env",
      packages = "python==3.9.6"
    ),
    error = function(e) {
      return(TRUE)
    }
  )
  if (isTRUE(status)) {
    print(status)
    stop("Error in Miniconda Installation.", call. = FALSE)
  }

  message("Installing tensorflow in environment")
  status2 <- tryCatch(
    tensorflow::install_tensorflow(
      version = "2.9.1",
      method = "conda",
      conda = conda,
      envname = "env"
    ),
    error = function(e) {
      return(TRUE)
    }
  )
  if (isTRUE(status2)) {
    stop(
      "Error during Tensorflow installation.",call. = FALSE)
  }

  message("Installing keras in environment")
  status3 <- tryCatch(
    keras::install_keras(
      version = "2.9.0",
      method = "conda",
      conda = conda,
      envname = "env"
    ),
    error = function(e) {
      return(TRUE)
    }
  )
  if (isTRUE(status3)) {
    stop(
      "Error during Tensorflow installation.",call. = FALSE)
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

