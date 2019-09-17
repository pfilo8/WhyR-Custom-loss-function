library(mlr)
library(readr)
library(dplyr)
source("aucpr_measure.R")
source("loss_functions.R")
source('RLearner_classif_xgboost_custom_loss_function.R')

train_df <- read_csv("data/PCA_Insurance_Frauds_Train.csv")
test_df <- read_csv("data/PCA_Insurance_Frauds_Test.csv")
train_size <- dim(train_df)[1]


df <- bind_rows(train_df, test_df)
df$TARGET = as.factor(df$TARGET)

total_size <- dim(df)[1]

task <-
  makeClassifTask(
    id = "TestTrainComparison",
    data = df,
    target = "TARGET",
    positive = 1
  )
learner_standard <-
  makeLearner(
    "classif.xgboost",
    predict.type = "prob",
    par.vals = list(
      nrounds = 283,
      max_depth = 6,
      eta = 0.0073,
      lambda = 0.695
    )
  )
learners_focal <- lapply(list(1, 2, 4, 6, 8), function(gamma) {
  makeLearner(
    "classif.xgboost.custom.loss.function",
    predict.type = "prob",
    par.vals = list(
      nrounds = 283,
      max_depth = 6,
      eta = 0.0073,
      lambda = 0.695,
      objective = focal_loss_with_params,
      focal_loss_gamma = gamma
    )
  )
})


resampling <-
  makeFixedHoldoutInstance(
    train.inds = seq(1, train_size),
    test.inds = seq(train_size + 1, total_size),
    size = total_size
  )
bmr <- benchmark(
  learners = learners_focal,
  tasks = task,
  measures = list(aucpr, aucpr_train),
  resamplings = resampling
)

bmr
