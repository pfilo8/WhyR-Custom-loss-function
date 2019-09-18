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

compute_aucpr_perc <- function(task, model, pred, feats, extra.args, perc) {
  top_n_percent <-
    arrange(pred$data, desc(prob.1)) %>% head(floor(nrow(pred$data) * perc))
  pr.curve(scores.class0 = top_n_percent$prob.1,
           weights.class0 = as.integer(as.character(top_n_percent$truth)))$auc.integral
}

aucpr_perc <- function(perc) {
  makeMeasure(
    id = "aucpr_perc",
    minimize = FALSE,
    properties = c("classif", "req.pred", "req.truth"),
    fun = partial(compute_aucpr_perc, perc=perc),
    best = 1,
    worst = 0
  )
}
