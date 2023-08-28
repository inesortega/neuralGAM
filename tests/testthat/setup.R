
clean_environment <- function() {
  library(fs)
  # Clean up tensorflow-generated files and caches
  python_temp_dir <- dirname(reticulate::py_run_string(
    "import tempfile; x=tempfile.NamedTemporaryFile().name",
    local = TRUE
  )$x)

  if (dir.exists(file.path(python_temp_dir, "__pycache__"))) {
    unlink(file.path(python_temp_dir, "__pycache__"), recursive = TRUE, force = TRUE)
  }

  files <- dir_ls(
    python_temp_dir,
    regexp = "__autograph_generated_file|__pycache__"
  )

  file_delete(files)

  unlink(file.path(path_home(), ".cache", "conda"), recursive = TRUE)
  unlink(file.path(path_home(), ".cache", "pip"), recursive = TRUE)
  unlink(file.path(path_home(), ".conda"), recursive = TRUE)
  unlink(file.path(path_home(), ".config", "calibre"), recursive = TRUE)
  unlink(file.path(path_home(), ".keras"), recursive = TRUE)
  unlink(file.path(path_home(), ".dbus"), recursive = TRUE)
}

library(withr)
withr::defer(clean_environment())
