library(testthat)
library(neuralGAM)

## ----------------------------
## Base cases (yours, kept)
## ----------------------------

test_that("get_formula_elements returns the correct output for a linear formula", {
  formula <- y ~ x1 + x2 + x3
  result <- get_formula_elements(formula)
  expect_equal(as.character(result$y), "y")
  expect_equal(result$terms, c("x1", "x2", "x3"))
  expect_null(result$np_terms)
  expect_equal(result$p_terms, c("x1", "x2", "x3"))
  expect_null(result$np_formula)
  expect_equal(as.character(result$p_formula), "y ~ x1 + x2 + x3")
  expect_equal(as.character(result$formula), "y ~ x1 + x2 + x3")
  expect_type(result$np_architecture, "list")
  expect_length(result$np_architecture, 0)
})

test_that("get_formula_elements returns the correct output for a formula with smooth terms", {
  formula <- y ~ x1 + s(x2) + s(x3)
  result <- get_formula_elements(formula)
  expect_equal(as.character(result$y), "y")
  expect_equal(result$terms, c("x1", "x2", "x3"))
  expect_equal(result$np_terms, c("x2", "x3"))
  expect_equal(result$p_terms, "x1")
  expect_equal(as.character(result$np_formula), "y ~ s(x2) + s(x3)")
  expect_equal(as.character(result$p_formula), "y ~ x1")
  expect_equal(as.character(result$formula), "y ~ x1 + s(x2) + s(x3)")
  expect_true(all(c("x2","x3") %in% names(result$np_architecture)))
})

test_that("get_formula_elements returns the correct output for a smooth formula", {
  formula <- y ~ s(x1) + s(x2)
  result <- get_formula_elements(formula)
  expect_equal(as.character(result$y), "y")
  expect_equal(result$terms, c("x1", "x2"))
  expect_equal(result$np_terms, c("x1", "x2"))
  expect_equal(result$p_terms, character(0))
  expect_equal(as.character(result$np_formula), "y ~ s(x1) + s(x2)")
  expect_null(result$p_formula)
  expect_equal(as.character(result$formula), "y ~ s(x1) + s(x2)")
})

test_that("get_formula_elements handles variable names containing 's' (not smooths)", {
  formula <- y ~ s(x1) + s(x2) + sample_var
  result <- get_formula_elements(formula)
  expect_equal(as.character(result$y), "y")
  expect_equal(result$terms, c("x1", "x2", "sample_var"))
  expect_equal(result$np_terms, c("x1", "x2"))
  expect_equal(result$p_terms, "sample_var")
  expect_equal(as.character(result$np_formula), "y ~ s(x1) + s(x2)")
  expect_equal(as.character(result$p_formula), "y ~ sample_var")
  expect_equal(as.character(result$formula), "y ~ s(x1) + s(x2) + sample_var")
})

test_that("get_formula_elements with a single smooth term", {
  formula <- y ~ s(sample_var)
  result <- get_formula_elements(formula)
  expect_equal(as.character(result$y), "y")
  expect_equal(result$terms, "sample_var")
  expect_equal(result$np_terms, "sample_var")
  expect_equal(result$p_terms, character(0))
  expect_equal(as.character(result$np_formula), "y ~ s(sample_var)")
  expect_null(result$p_formula)
  expect_equal(as.character(result$formula), "y ~ s(sample_var)")
})

## ----------------------------
## New: per-term architecture extraction
## ----------------------------

test_that("get_formula_elements parses per-term num_units and activation", {
  formula <- y ~ s(x1, num_units = c(128, 64), activation = "tanh") + s(x2, num_units = 32)
  result <- get_formula_elements(formula)

  expect_equal(result$np_terms, c("x1", "x2"))
  expect_true(all(c("x1","x2") %in% names(result$np_architecture)))

  cfg1 <- result$np_architecture$x1
  cfg2 <- result$np_architecture$x2

  expect_equal(cfg1$num_units, c(128, 64))
  expect_equal(cfg1$activation, "tanh")
  expect_null(cfg1$learning_rate)

  expect_equal(cfg2$num_units, 32)
  expect_null(cfg2$activation)
})

## ----------------------------
## New: spacing / multiline robustness
## ----------------------------

test_that("get_formula_elements handles whitespace and multiline s(...) calls", {
  formula <- y ~
    s( x1 ,
       num_units = c(64,32) ,
       activation = "relu"
    ) + x2 +
    s(
      x3, num_units = 16
    )

  result <- get_formula_elements(formula)
  expect_equal(result$np_terms, c("x1","x3"))
  expect_equal(sort(result$terms), c("x1","x2","x3"))
  expect_equal(result$p_terms, "x2")

  cfg1 <- result$np_architecture$x1
  cfg3 <- result$np_architecture$x3
  expect_equal(cfg1$num_units, c(64,32))
  expect_equal(cfg1$activation, "relu")
  expect_equal(cfg3$num_units, 16)
})

## ----------------------------
## New: do not misparse other calls (e.g., sin())
## ----------------------------

test_that("get_formula_elements does not treat sin(x) as a smooth term", {
  formula <- y ~ sin(x1) + s(x2)
  result <- get_formula_elements(formula)
  expect_equal(result$terms, c("x1","x2"))
  expect_equal(result$np_terms, "x2")
  expect_equal(result$p_terms, "x1")
  expect_equal(as.character(result$np_formula), "y ~ s(x2)")
  expect_equal(as.character(result$p_formula), "y ~ x1")
})

## ----------------------------
## New: Keras object arguments (initializers/regularizers)
## ----------------------------

skip_if_no_keras <- function() {
  if (!requireNamespace("keras", quietly = TRUE)) skip("keras not available")
}

test_that("get_formula_elements captures Keras initializers and regularizers as objects", {
  skip_if_no_keras()

  formula <- y ~ s(
    x,
    num_units = 5,
    kernel_initializer = keras::initializer_he_normal(),
    bias_initializer   = keras::initializer_zeros(),
    kernel_regularizer = keras::regularizer_l2(1e-4)
  )

  result <- get_formula_elements(formula)
  cfg <- result$np_architecture$x

  # presence
  expect_true("kernel_initializer" %in% names(cfg))
  expect_true("bias_initializer" %in% names(cfg))
  expect_true("kernel_regularizer" %in% names(cfg))

  # basic type checks via class/inherits
  expect_true(inherits(cfg$kernel_initializer, "python.builtin.object"))
  expect_true(inherits(cfg$bias_initializer, "python.builtin.object"))
  expect_true(
    inherits(cfg$kernel_regularizer, "keras.regularizers.Regularizer") ||
      inherits(cfg$kernel_regularizer, "tensorflow.python.keras.regularizers.Regularizer") ||
      inherits(cfg$kernel_regularizer, "keras.src.regularizers.Regularizer")
  )
})

## ----------------------------
## New: no-smooth all-parametric returns np_terms = NULL and empty np_architecture
## ----------------------------

test_that("get_formula_elements returns NULL np_terms and empty np_architecture when there are no s(...) terms", {
  formula <- y ~ x + z + w
  result <- get_formula_elements(formula)
  expect_null(result$np_terms)
  expect_type(result$np_architecture, "list")
  expect_length(result$np_architecture, 0)
})
