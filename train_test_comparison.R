library(mlr)
library(readr)
source("aucpr_measure.R")

train_df <- read_csv("data/PCA_Insurance_Frauds_Train.csv")
test_df <- read_csv("data/PCA_Insurance_Frauds_Test.csv")
train_df$TARGET = as.factor(train_df$TARGET)
test_df$TARGET = as.factor(test_df$TARGET)

task <- makeClassifTask(id="TestTrainComparison", data = train_df, target = "TARGET", positive = 1)
learner <- makeLearner("classif.xgboost", predict.type = "prob", par.vals = list(nrounds=283, max_depth=6, eta=0.0073, lambda=0.695))

model <- train(learner, task)

pred <- predict(model, newdata = test_df)

# test data aucpr
compute_aucpr(pred=pred)

#train data aucpr
compute_aucpr(pred=predict(model, task))
