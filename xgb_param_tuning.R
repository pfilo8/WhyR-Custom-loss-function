library(xgboost)
library(readr)
library(dplyr)
library(mlr)
library(parallelMap)

source("aucpr_measure.R")
parallelStartMulticore(3)

task <- makeClassifTask(id="CE-xgboost", data = df, target = "TARGET", positive = 1)
learner <- makeLearner("classif.xgboost", predict.type = "prob")

xgb_params <- makeParamSet(
  # The number of trees in the model (each one built sequentially)
  makeIntegerParam("nrounds", lower = 100, upper = 500),
  # number of splits in each tree
  makeIntegerParam("max_depth", lower = 1, upper = 10),
  # "shrinkage" - prevents overfitting
  makeNumericParam("eta", lower = .1, upper = .5),
  # L2 regularization - prevents overfitting
  makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x)
)
control <- makeTuneControlRandom(maxit = 1)
resample_desc <- makeResampleDesc("CV", iters = 2)

tunedParams <- tuneParams(learner = learner,
                          task = task,
                          resampling = resample_desc,
                          measures = aucpr,
                          par.set = xgb_params,
                          control=control)

parallelStop()