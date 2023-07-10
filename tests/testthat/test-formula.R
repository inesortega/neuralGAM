library(testthat)
library(neuralGAM)

# Write tests using testthat
test_that("get_formula_elements returns the correct output for a linear formula", {
  formula <- y ~ x1 + x2 + x3
  result <- get_formula_elements(formula)
  expect_equal(as.character(result$y), "y")
  expect_equal(result$terms, c("x1", "x2", "x3"))
  expect_equal(result$np_terms, NULL)
  expect_equal(result$p_terms, c("x1", "x2", "x3"))
  expect_equal(result$np_formula, NULL)
  expect_equal(as.character(result$p_formula), "y ~ x1 + x2 + x3")
  expect_equal(as.character(result$formula), "y ~ x1 + x2 + x3")
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
})

test_that("get_formula_elements returns the correct output for a smooth formula", {
  formula <- y ~ s(x1) + s(x2)
  result <- get_formula_elements(formula)
  expect_equal(as.character(result$y), "y")
  expect_equal(result$terms, c("x1", "x2"))
  expect_equal(result$np_terms, c("x1", "x2"))
  expect_equal(result$p_terms, NULL)
  expect_equal(as.character(result$np_formula), "y ~ s(x1) + s(x2)")
  expect_equal(result$p_formula, NULL)
  expect_equal(as.character(result$formula), "y ~ s(x1) + s(x2)")
})

test_that("get_formula_elements returns the correct output for a formula which contains s in a variable", {
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

test_that("get_formula_elements returns the correct output for a formula which contains s in a variable", {
  formula <- y ~ s(sample_var)
  result <- get_formula_elements(formula)
  expect_equal(as.character(result$y), "y")
  expect_equal(result$terms, c("sample_var"))
  expect_equal(result$np_terms, c("sample_var"))
  expect_equal(result$p_terms, NULL)
  expect_equal(as.character(result$np_formula), "y ~ s(sample_var)")
  expect_equal(result$p_formula, NULL)
  expect_equal(as.character(result$formula), "y ~ s(sample_var)")
})
