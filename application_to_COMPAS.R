#install.packages("fairness")

dataset <- read.csv("./compas-scores-two-years.csv")

data <- subset(dataset, select=c("age", "priors_count", "c_charge_degree", "race", "sex", "two_year_recid"))

data$c_charge_degree <- factor(data$c_charge_degree)
data$sex <- factor(data$sex)
data$race <- factor(data$race)

seed <- 42
set.seed(seed)

sample <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.8,0.2))
train <- data[sample, ]
test <- data[!sample, ]

library(neuralGAM)
ngam <- neuralGAM(two_year_recid ~ s(priors_count) + s(age) + c_charge_degree  + race + sex,
                  data = train, num_units = 512, family = "binomial",
                  learning_rate = 0.01, ls_threshold = 0.01, seed = seed, bf_threshold=0.01)

ngam

plot(ngam, select = c("priors_count", "age"), xlab = c("Priors Count", "Age"))

plot(ngam, select = c("c_charge_degree", "race", "sex"), xlab = c("Charge Degree", "Race", "Gender"))

predictions <- predict(ngam, newdata = test, type = "response")

library(pROC)

rocobj = pROC::roc(test$two_year_recid, predictions)
threshold <- coords(rocobj, "best")

# Threshold the predictions to obtain binary class labels (e.g., using 0.5 as the threshold)
predicted_classes <- ifelse(predictions > threshold$threshold, 1, 0)

# Compute metrics
TP <- sum(test$two_year_recid == 1 & predicted_classes == 1)
TN <- sum(test$two_year_recid == 0 & predicted_classes == 0)
FP <- sum(test$two_year_recid == 0 & predicted_classes == 1)
FN <- sum(test$two_year_recid == 1 & predicted_classes == 0)

accuracy <- (TP + TN) / (TP + TN + FP + FN)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)
f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
AUC <- pROC::auc(rocobj)

# Print the metrics
print(paste("Accuracy ", accuracy))
print(paste("Sensitivity ", sensitivity))
print(paste("Specificity ", specificity))
print(paste("Precision ", precision))
print(paste("F1Score ", f1_score))
print(AUC)

