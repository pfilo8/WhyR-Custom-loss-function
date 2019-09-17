
focal_loss_with_params <- function(focal_loss_gamma = 1) {
  
  force(focal_loss_gamma)
  
  focal_loss <- function(preds, dtrain) {
    
    robust_pow <- function(num_base, num_pow){
      return (sign(num_base) * (abs(num_base)) ** (num_pow))
    }

    labels <- getinfo(dtrain, "label")
    preds <- 1/(1 + exp(-preds))
    
    g1 <- preds * (1 - preds)
    g2 <- labels + ((-1) ** labels) * preds
    g3 <- preds + labels - 1
    g4 <- 1 - labels - ((-1) ** labels) * preds
    g5 <- labels + ((-1) ** labels) * preds
    
    
    grad <- focal_loss_gamma * g3 * robust_pow(g2, focal_loss_gamma) * log(g4 + 1e-9) + ((-1) ** labels) * robust_pow(g5, (focal_loss_gamma + 1))
    
    hess_1 <- robust_pow(g2, focal_loss_gamma) + focal_loss_gamma * ((-1) ** labels) * g3 * robust_pow(g2, (focal_loss_gamma - 1))
    hess_2 <- ((-1) ** labels) * g3 * robust_pow(g2, focal_loss_gamma) / g4
    hess <- ((hess_1 * log(g4 + 1e-9) - hess_2) * focal_loss_gamma + (focal_loss_gamma + 1) * robust_pow(g5, focal_loss_gamma)) * g1
    
    return(list(grad = grad, hess = hess))
  }
  
  return(focal_loss)
}


weighted_cross_entropy_loss_with_params <- function(weighted_cross_entropy_loss_weight = 1) {
  force(weighted_cross_entropy_loss_weight)
  
  weighted_cross_entropy_loss <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    preds <- 1/(1 + exp(-preds))
    
    grad <- -(weighted_cross_entropy_loss_weight ** labels) * (labels - preds)
    hess <- (weighted_cross_entropy_loss_weight ** labels) * preds * (1 - preds)
    
    return(list(grad = grad, hess = hess))
  }
  
  return(weighted_cross_entropy_loss)
}


bilinear_loss_with_params <- function(biliniear_loss_weight = 1, biliniear_loss_alpha = 0){
  force(biliniear_loss_weight)
  force(biliniear_loss_alpha)
  
  bilinear_loss <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    preds <- 1/(1 + exp(-preds))
    
    grad_ce <- preds - labels
    grad_l <- preds*(1 - preds)*(1 - labels*(1 + biliniear_loss_weight))
    grad <- (1 - biliniear_loss_alpha)*grad_ce + biliniear_loss_alpha*grad_l
    
    hess_ce <- preds*(1 - preds)
    hess_l <- grad_l*(1 - 2*preds)
    hess <- (1 - biliniear_loss_alpha)*hess_ce + biliniear_loss_alpha*hess_l
    
    return(list(grad = grad, hess = hess))
  }
  
  return(bilinear_loss)
}




