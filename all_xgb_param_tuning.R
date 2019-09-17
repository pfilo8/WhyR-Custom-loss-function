library(xgboost)
library(readr)
library(dplyr)
library(mlr)
library(parallelMap)

source("loss_functions.R")
source('RLearner_classif_xgboost_custom_loss_function.R')
source("aucpr_measure.R")
parallelStartMulticore(32)
set.seed(42)

df <- read_csv("data/PCA_Insurance_Frauds_Train.csv")
df$TARGET <- factor(df$TARGET)

task <-
  makeClassifTask(
    id = "CE-xgboost",
    data = df,
    target = "TARGET",
    positive = 1
  )

# Learners
standard_xgb_learner <- makeLearner("classif.xgboost", predict.type = "prob")
focal_loss_xgb_learner <- makeLearner("classif.xgboost.custom.loss.function", predict.type = "prob", 
                                  objective = focal_loss_with_params)
weighted_cross_entropy_xgb_learner <- makeLearner("classif.xgboost.custom.loss.function", predict.type = "prob", 
                                      objective = weighted_cross_entropy_loss_with_params)
bilinear_loss_xgb_learner <- makeLearner("classif.xgboost.custom.loss.function", predict.type = "prob", 
                                      objective = bilinear_loss_with_params)

# Params
standard_xgb_params <- makeParamSet(
  makeIntegerParam("nrounds", lower = 100, upper = 500),
  makeIntegerParam("max_depth", lower = 1, upper = 10),
  makeNumericParam("eta", lower = .1, upper = .5),
  makeNumericParam(
    "lambda",
    lower = -1,
    upper = 0,
    trafo = function(x)
      10 ^ x
  ),
  makeNumericParam("colsample_bytree", lower = 0.4, upper = 1),
  makeNumericParam("subsample", lower = 0.4, upper = 1))

focal_loss_xgb_params <- c(
  standard_xgb_params, 
  makeParamSet(
    makeNumericParam("focal_loss_gamma", lower = -1, upper = 1, trafo = function(x) 10^x)
    )
  )

weighted_cross_entropy_xgb_params <- c(
  standard_xgb_params,
  makeParamSet(
    makeNumericParam("weighted_cross_entropy_loss_weight", lower = 1, upper = 200)
    )
  )

bilinear_loss_xgb_params <- c(
  standard_xgb_params,
  makeParamSet(
    makeNumericParam("biliniear_loss_weight", lower = 1, upper = 200),
    makeNumericParam("biliniear_loss_alpha", lower = 0, upper = 1)
    )
  )


get.tuned.params <- function(df, task, learner, par.set) {

  control <- makeTuneControlRandom(maxit = 64L)
  resample_desc <- makeResampleDesc("CV", iters = 5, stratify = TRUE)
  
  tunedParams <- tuneParams(
    learner = learner,
    task = task,
    resampling = resample_desc,
    measures = list(aucpr, aucpr_train),
    par.set = par.set,
    control = control
  )
  return(tunedParams)
}

learners <- list(
#  standard_xgb_learner, 
  focal_loss_xgb_learner, 
  weighted_cross_entropy_xgb_learner, 
  bilinear_loss_xgb_learner
  )

params <- list(
#  standard_xgb_params,
  focal_loss_xgb_params,
  weighted_cross_entropy_xgb_params,
  bilinear_loss_xgb_params
)

tune_results <-
  mapply(
    function(learner, param) {
    get.tuned.params(
      df = df,
      task = task,
      learner = learner,
      par.set = param)
      }, 
    learners, 
    params
  )

save(tune_results, file="tune_results.Rdata")
parallelStop()
