library(PRROC)
library(dplyr)
library(purrr)
library(mlr)

compute_aucpr <- function(task, model, pred, feats, extra.args) {
  PRROC::pr.curve(
    scores.class0 = pred$data$prob.1,
    weights.class0 = as.numeric(as.character(pred$data$truth))
  )$auc.integral
}

aucpr <- makeMeasure(
  id = "aucpr",
  minimize = FALSE,
  properties = c("classif", "req.pred", "req.truth"),
  fun = compute_aucpr,
  best = 1,
  worst = 0
)

compute_aucpr_train <-
  function(task, model, pred, feats, extra.args) {
    compute_aucpr(pred = predict(model, task))
  }

aucpr_train <- makeMeasure(
  id = "aucpr_train",
  minimize = FALSE,
  properties = c("classif", "req.pred", "req.truth", "req.model", "req.task"),
  fun = compute_aucpr_train,
  best = 1,
  worst = 0
)
