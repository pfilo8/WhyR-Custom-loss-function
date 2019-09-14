library(xgboost)
library(readr)
library(dplyr)
library(mlr)
library(parallelMap)
parallelStartSocket(7)

source('RLearner_classif_xgboost_custom_loss_function.R')
source('loss_functions.r')

df <- read_csv("data/PCA_Insurance_Frauds.csv")
df <- df %>% select(-EVENT_DATE)
df$TARGET <- factor(df$TARGET)

task <- makeClassifTask(id="CE-xgboost", data = df, target = "TARGET", positive = 1)
learner <- makeLearner("classif.xgboost.custom.loss.function", 
                       predict.type = "prob",
                       max_depth = 4,
                       eta = 0.01,
                       nthread = 7,
                       nrounds = 50,
                       objective = focal_loss)

measures <- list(mlr::auc, mmce)
sampling_startegy <- makeResampleDesc("CV", iters=2, stratify = TRUE)

bmr <- benchmark(list(learner), tasks = task, resamplings = sampling_startegy, measures = measures)
bmr$results
perf_data_pr <- generateThreshVsPerfData(bmr, measures = list(tpr, ppv), gridsize = 100)
perf_data_auc <- generateThreshVsPerfData(bmr, measures = list(fpr, tpr), gridsize = 100)
plotROCCurves(perf_data_auc)

parallelStop()
