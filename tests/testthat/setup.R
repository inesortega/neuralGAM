clean_environment <- function() {
  library(withr)
  library(fs)

  # Clean up tensorflow-generated files and caches
  python_temp_dir <- dirname(reticulate::py_run_string(
    "import tempfile; x=tempfile.NamedTemporaryFile().name",
    local = TRUE
  )$x)

  tmp <- tempdir()
  rm_files(python_temp_dir)
  rm_files(tmp)

  unlink(file.path(fs::path_home(), ".cache", "conda"), recursive = TRUE)
  unlink(file.path(fs::path_home(), ".cache", "pip"), recursive = TRUE)
  unlink(file.path(fs::path_home(), ".conda"), recursive = TRUE)
  unlink(file.path(fs::path_home(), ".config", "calibre"), recursive = TRUE)
  unlink(file.path(fs::path_home(), ".keras"), recursive = TRUE)
  unlink(file.path(fs::path_home(), ".dbus"), recursive = TRUE)

}

rm_files <- function(location){

  if (dir.exists(file.path(location, "__pycache__"))) {
    unlink(file.path(location, "__pycache__"), recursive = TRUE, force = TRUE)
  }
  tmp_py_files <- list.files(location, pattern = "^(tmp|__autograph_generated_file).*\\.py$", full.names = TRUE)
  fs::file_delete(tmp_py_files)

}
withr::defer(clean_environment(), testthat::teardown_env())
