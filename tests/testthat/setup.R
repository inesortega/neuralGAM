clean_environment <- function() {
  # Clean up tensorflow-generated files and caches
  python_temp_dir <- dirname(reticulate::py_run_string(
    "import tempfile; x=tempfile.NamedTemporaryFile().name",
    local = TRUE
  )$x)

  rm_files(python_temp_dir)
  rm_files(tempdir())

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

  files <- fs::dir_ls(
    location,
    regexp = "__autograph_generated_file|__pycache__"
  )

  cat(files)

  fs::file_delete(files)
}

library(withr)
library(fs)
withr::defer(clean_environment())
