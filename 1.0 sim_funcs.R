# 
# Project code for Master Thesis for MSc Data Science and Business Analytics
# Bocconi University
#
# Name:           Sean Goedgeluk 
# Student number: 3190974
# Supervised by:  Prof.dr. Giacomo Zanella
#
# Title: Estimating Loss Reserves using a Mixture Density Network in a Deep Learning Recurrent Framework
#
# File Description:
# Defines functions to simulate loss data using the SynthETIC simulation model.

# Function to simulate individual loss data using SynthETIC simulation model
GenData = function(seed){
  
  set.seed(as.integer(seed))
  
  # Module 1: Occurance times
  n_vector <- claim_frequency(I, E, lambda)
  occurrence_times <- claim_occurrence(n_vector)
  
  # Module 2: Claim size
  claim_sizes <- claim_size(n_vector, S_df, type = "p", range = c(0, 1e24))
  
  # Module 3: Claim notification
  notidel <- claim_notification(n_vector, claim_sizes, paramfun = notidel_param)
  
  # Module 4: Claim settlement
  setldel <- claim_closure(n_vector, claim_sizes, paramfun = setldel_param)
  
  # Module 5: Claim payment count
  no_payments <- claim_payment_no(n_vector, claim_sizes, rfun = rmixed_payment_no,
                                  claim_size_benchmark_1 = 0.0375 * ref_claim,
                                  claim_size_benchmark_2 = 0.075 * ref_claim)
  
  # Module 6: Claim payment size
  payment_sizes <- claim_payment_size(n_vector, claim_sizes, no_payments,
                                      rfun = rmixed_payment_size)
  
  # Module 7: Claim payment time
  payment_delays <- claim_payment_delay(n_vector, claim_sizes, no_payments, setldel,
                                        rfun = r_pmtdel, paramfun = param_pmtdel,
                                        occurrence_period = rep(1:I, times = n_vector))
  payment_times <- claim_payment_time(n_vector, occurrence_times, notidel, payment_delays)
  
  # Module 8: Claim inflation
  payment_inflated <- claim_payment_inflation(
    n_vector, payment_sizes, payment_times, occurrence_times,
    claim_sizes, base_inflation_vector, SI_occurrence, SI_payment)
  
  transaction_dataset <- generate_transaction_dataset(
    claims(
      frequency_vector = n_vector,
      occurrence_list = occurrence_times,
      claim_size_list = claim_sizes,
      notification_list = notidel,
      settlement_list = setldel,
      no_payments_list = no_payments,
      payment_size_list = payment_sizes,
      payment_delay_list = payment_delays,
      payment_time_list = payment_times,
      payment_inflated_list = payment_inflated),
    adjust = FALSE)  # adjust = FALSE to retain the original simulated times
  
  return(transaction_dataset)
}

# # Construct triangle using loss_data
# GenLossTriangle = function(loss_data, runoff) {
#   triangle <- matrix(0, nrow = 10, ncol = 10)
#   
#   # For loop construction to calculate the losses for each of the cells in the
#   # 10 by 10 loss triangle
#   for (n in 1:10){
#     for (m in 1:10){
#       tr1 <- copy(loss_data[(loss_data$occurrence_period > (n-1)*4) & (loss_data$occurrence_period <= n*4),])
#       if (m == 10 && runoff == 1){
#         tr2 <- tr1[(tr1$payment_period > (m-1)*4 + (n-1)*4),]
#       } else {
#         tr2 <- tr1[(tr1$payment_period > (m-1)*4 + (n-1)*4) & (tr1$payment_period <= (m-1)*4 + (n-1)*4 + 7),]
#       }
#       
#       triangle[n,m] = sum(tr2$payment_inflated)
#     }
#     
#   }
#   
#   triangle <- as.data.table(triangle)
#   
#   Data = as.data.frame(copy(triangle))
#   AY <- c(1:10)
#   Data <- cbind(AY, Data)
#   colnames(Data)<- c("AY",1:10)
#   return(Data)
# }

# Converts quarterly triangles into yearly triangles.
trq_to_try <- function(q_triangle) {
  y_triangle <- data.frame(matrix(ncol = 10, nrow = 10))
  q_dat <- melt(as.data.table(q_triangle),id = c("AP")) %>%
    arrange(AP)
  q_dat$CALP <- as.integer(q_dat$AP)-1+as.integer(q_dat$variable)
  colnames(q_dat) <- c("AP","DP","value","CALP")
  
  for(i in 1:10) {
    AP_min <- 4*(i-1)
    AP_max <- 4*i
    for(j in 1:10) {
      CALP_min <- AP_min+4*(j-1)
      CALP_max <- AP_min+4*j
      tmp <- q_dat[(AP>AP_min & AP<=AP_max) & (CALP>CALP_min & CALP<=CALP_max)]
      if(j==10) {
        tmp <- q_dat[(AP>AP_min & AP<=AP_max) & (CALP>CALP_min)]
      }
      tmp <- sum(tmp$value)
      y_triangle[i,j] <- tmp
    }
  }
  y_triangle <- as.data.frame(cbind(1:10,y_triangle))
  colnames(y_triangle) <- c("AY",1:10)
  return(y_triangle)
}

# Determines the loss triangles based on the individual losses simulated by 
# the SynthETIC simulation package.
gen_loss_triangle_q <- function(loss_data, runoff) {
  triangle <- matrix(0, nrow = 40, ncol = 40)
  
  # For loop construction to calculate the losses for each of the cells in the
  # 10 by 10 loss triangle
  for (n in 1:40){
    for (m in 1:40){
      tr1 <- copy(loss_data[loss_data$occurrence_period == n,])
      if (m == 40 && runoff == 1){
        tr2 <- tr1[(tr1$payment_period >= m + n - 1),]
      } else {
        tr2 <- tr1[(tr1$payment_period == m + n - 1),]
      }
      
      triangle[n,m] = sum(tr2$payment_inflated)
    }
    
  }
  
  triangle <- as.data.table(triangle)
  
  Data = as.data.frame(copy(triangle))
  AY <- c(1:40)
  Data <- cbind(AY, Data)
  colnames(Data)<- c("AP",1:40)
  return(Data)
}
