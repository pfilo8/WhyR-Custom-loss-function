library(xgboost)
library(readr)
library(dplyr)
library(mlr)
library(parallelMap)
parallelStartMulticore(3)
source("aucpr_measure.R")

df <- read_csv("data/PCA_Insurance_Frauds.csv")
df$TARGET <- factor(df$TARGET)

task <- makeClassifTask(id="CE-xgboost", data = df, target = "TARGET", positive = 1)
learner <- makeLearner("classif.xgboost", predict.type = "prob",
            par.vals = list(
              max_depth = 4,
              eta = 0.01,
              nthread = 4,
              nrounds = 50,
              objective = "binary:logistic"))

measures <- list(mlr::auc, aucpr)
sampling_startegy <- makeResampleDesc("CV", iters=2, stratify = TRUE)

bmr <- benchmark(list(learner), tasks = task, resamplings = sampling_startegy, measures = measures)

perf_data <- generateThreshVsPerfData(bmr, measures = list(tpr, ppv), gridsize = 1000)
perf_data_auc <- generateThreshVsPerfData(bmr, measures = list(fpr, tpr), gridsize = 1000)
plotROCCurves(perf_data_auc)

parallelStop()
