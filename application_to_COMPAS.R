dataset <- read.csv("./compas-scores-two-years.csv")

data <- subset(dataset, select=c("age", "priors_count", "c_charge_degree", "race", "sex", "two_year_recid"))
head(data)

data$c_charge_degree <- factor(data$c_charge_degree)
levels(data$c_charge_degree) <- c("Felony", "Misdemeanor")
data$sex <- factor(data$sex)
data$race <- factor(data$race)

seed <- 4321412
set.seed(seed)

library(NeuralGAM)
ngam <- NeuralGAM(two_year_recid ~ s(priors_count) + s(age) + c_charge_degree  + race + sex,
                  data = data, num_units = list(1024), family = "binomial",
                  learning_rate = 0.001, ls_threshold = 0.01, seed = seed,
                  bf_threshold = 0.01)

ngam

plot(ngam, select = c("priors_count", "age"), xlab = c("Priors Count", "Age"))

plot(ngam, select = c("c_charge_degree", "race", "sex"), xlab = c("Charge Degree", "Race", "Gender"))
