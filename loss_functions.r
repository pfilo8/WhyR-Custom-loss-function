

focal_loss <- function(preds, dtrain) {
  
  robust_pow <- function(num_base, num_pow){
    return (sign(num_base) * (abs(num_base)) ** (num_pow))
  }
  
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  gamma_indct <- 2
  
  g1 <- preds * (1 - preds)
  g2 <- labels + ((-1) ** labels) * preds
  g3 <- preds + labels - 1
  g4 <- 1 - labels - ((-1) ** labels) * preds
  g5 <- labels + ((-1) ** labels) * preds
  
  
  grad <- gamma_indct * g3 * robust_pow(g2, gamma_indct) * log(g4 + 1e-9) + ((-1) ** labels) * robust_pow(g5, (gamma_indct + 1))

  hess_1 <- robust_pow(g2, gamma_indct) + gamma_indct * ((-1) ** labels) * g3 * robust_pow(g2, (gamma_indct - 1))
  hess_2 <- ((-1) ** labels) * g3 * robust_pow(g2, gamma_indct) / g4
  hess <- ((hess_1 * log(g4 + 1e-9) - hess_2) * gamma_indct + (gamma_indct + 1) * robust_pow(g5, gamma_indct)) * g1
  
  return(list(grad = grad, hess = hess))
}


weighted_loss <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  alpha <- 200
  
  grad <- -(alpha ** labels) * (labels - preds)
  hess <- (alpha ** labels) * preds * (1 - preds)
  
  return(list(grad = grad, hess = hess))
}

bilinear_loss <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  alpha <- 0.25
  cost <- 5
  
  grad_ce <- preds - labels
  grad_l <- preds*(1 - preds)*(1 - labels*(1 + cost))
  grad <- (1 - alpha)*grad_ce + alpha*grad_l
  
  hess_ce <- preds*(1 - preds)
  hess_l <- grad_l*(1 - 2*preds)
  hess <- (1 - alpha)*hess_ce + alpha*hess_l
  
  return(list(grad = grad, hess = hess))
}

