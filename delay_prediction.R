library(neuralGAM)
library(magrittr)
library(dplyr)
library(ggplot2)
library(gridExtra)


# filter(flights, month %in% c(12)) %>%

data(flights, package="nycflights13")
data(weather, package="nycflights13")
data(airlines, package="nycflights13")

dat <- filter(flights, origin == "EWR" & month %in% c(12,11,10)) %>%
  left_join(weather, by = c("origin", "time_hour"))  %>%
  left_join(airlines, by = c("carrier")) %>%
  select(arr_delay, dep_delay, dep_time, origin, carrier, name, visib, distance, air_time, temp, humid) %>%
  data.frame


dat$delay_group <- cut(dat$arr_delay,
                      breaks = c(-Inf, 0, 5, 45, Inf),
                      labels = c("No delay", "Small Delay", "Medium Delay", "Large Delay"),
                      right = FALSE)


dat$delay = ifelse(dat$arr_delay > 0, 1, 0)

dat <- dat[!rowSums(is.na(dat)), ]

# ggplot(dat, aes(x = arr_delay, y = name, fill = delay_group)) +
#   geom_bar(stat = "identity", position = "dodge") +
#   labs(y = "Carrier", x = "Arr Delay", fill = "Delay Group") +
#   ggtitle("Arr Delay per Carrier Grouped by Delay") +
#   theme_minimal()

dim(dat)

seed <- 42
set.seed(seed) # setting seed to reproduce results of random sampling

trainingRows <- sample(nrow(dat), 0.8 * nrow(dat))# row indices for training data
training <- dat[trainingRows,] # model training data
test <- dat[-trainingRows,]   # test data

formula <- delay ~ distance + s(dep_delay) + s(temp) + s(humid)

ngam <- neuralGAM(formula,
                  data = training, num_units = c(1024), family = "binomial",
                  seed = 1234, bf_threshold = 1e-2, ls_threshold = 0.1,
                  loss = "mean_squared_error")
plot(ngam)
# Make predictions using the binomial GAM model
predictions <- predict(ngam, newdata = test, type = "response")

library(pROC)
roc_data <- roc(test$delay, predictions)
fpr <- roc_data$specificities
tpr <- roc_data$sensitivities
best <- coords(roc_data, "best")

# Threshold the predictions to obtain binary class labels (e.g., using the youden index as thershold)
predicted_classes <- ifelse(predictions > best$threshold, 1, 0)

# Compute metrics
TP <- sum(test$delay == 1 & predicted_classes == 1)
TN <- sum(test$delay == 0 & predicted_classes == 0)
FP <- sum(test$delay == 0 & predicted_classes == 1)
FN <- sum(test$delay == 1 & predicted_classes == 0)

accuracy <- (TP + TN) / (TP + TN + FP + FN)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)
f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)

# Print the metrics
print(accuracy)
print(sensitivity)
print(specificity)
print(precision)
print(f1_score)
print(roc_data$auc)


##### Compute the same for GAM MGCV:


library(mgcv)

gam <- mgcv::gam(delay ~ distance + air_time + s(temp) + s(humid),
                 data = training,
                 family = binomial)

plot(gam, all.terms = TRUE)

# Make predictions using the binomial GAM model
predictions <- predict(gam, newdata = test, type = "response")

library(pROC)
roc_data <- roc(test$delay, predictions)
fpr <- roc_data$specificities
tpr <- roc_data$sensitivities
best <- coords(roc_data, "best")

# Threshold the predictions to obtain binary class labels (e.g., using the youden index as thershold)
predicted_classes <- ifelse(predictions > best$threshold, 1, 0)

# Compute metrics
TP <- sum(test$delay == 1 & predicted_classes == 1)
TN <- sum(test$delay == 0 & predicted_classes == 0)
FP <- sum(test$delay == 0 & predicted_classes == 1)
FN <- sum(test$delay == 1 & predicted_classes == 0)

accuracy <- (TP + TN) / (TP + TN + FP + FN)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)
f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)

# Print the metrics
print(accuracy)
print(sensitivity)
print(specificity)
print(precision)
print(f1_score)
print(roc_data$auc)
