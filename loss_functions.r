
robust_power <- function(num_base, num_pow){
  return (sign(num_base) * (abs(num_base)) ** (num_pow))
}


focal_loss <- function(preds, dtrain, gamma_indct_indct) {
  labels <- getinfo(dtrain, 'label')
  preds <- 1/(1 + exp(-preds))
  
  g1 <- preds * (1 - preds)
  g2 <- label + ((-1) ** label) * preds
  g3 <- preds + label - 1
  g4 <- 1 - label - ((-1) ** label) * preds
  g5 <- label + ((-1) ** label) * preds
  
  
  grad <- gamma_indct * g3 * robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) + ((-1) ** label) * robust_pow(g5, (gamma_indct + 1))

  hess_1 <- robust_pow(g2, gamma_indct) + gamma_indct * ((-1) ** label) * g3 * robust_pow(g2, (gamma_indct - 1))
  hess_2 <- ((-1) ** label) * g3 * robust_pow(g2, gamma_indct) / g4
  hess <- ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct + (gamma_indct + 1) * robust_pow(g5, gamma_indct)) * g1
  
  return(list(grad <- grad, hess <- hess))
}


weighted_loss <- function(preds, dtrain, alpha) {
  labels <- getinfo(dtrain, 'label')
  preds <- 1/(1 + exp(-preds))
  
  grad <- -(alpha ** label) * (label - preds)
  hess <- (alpha ** label) * preds * (1.0 - preds)
  
  return(list(grad <- grad, hess <- hess))
}


bilinear_loss <- function(preds, dtrain, alpha, cost) {
  labels <- getinfo(dtrain, 'label')
  preds <- 1/(1 + exp(-preds))
  
  grad_ce <- (preds-labels) + 
  grad_l <- preds*(1-preds)*(1-labels*(1+cost))
  grad <- (1-alpha)*grad_ce + alpha*grad_l
  
  hess_ce <- preds*(1-preds)
  hess_l <- grad_l*(1-2*preds)
  hess <- (1-alpha)*hess_ce + alpha*hess_l
  
  return(list(grad <- grad, hess <- hess))
}
