# 
# Project code for Master Thesis for MSc Data Science and Business Analytics
# Bocconi University
#
# Name:           Sean Goedgeluk 
# Student number: 3190974
# Supervised by:  Prof.dr. Giacomo Zanella
#
# Title: 
#
# File Description:
# Final training of the DeepTriangle-based models for the real data scenarios

# Import necessary packages, models and support functions
source("models/model_dt_no_cr.R")
source("models/model_dt_single.R")

source("0.1 support_funcs.R")
source("2.1 rd_keras.R")

## Output directory
data_used <- "rd_wc"

## Maximum number of epochs ##
set_epochs <- 500

## Best params
params_list_DT <- fread(paste0("output_rd/",data_used,"/y_DT_test_scores_neurons.csv"))

best_params_dt  <- params_list_DT[(params_list_DT$test_score == min(params_list_DT[params_list_DT$model=="DT",]$test_score) 
                                        & params_list_DT$model=="DT"),]
best_params_dt2  <- params_list_DT[(params_list_DT$test_score == min(params_list_DT[params_list_DT$model=="DT2",]$test_score) 
                                     & params_list_DT$model=="DT2"),]

best_params_dt <- as.data.frame(best_params_dt)
best_params_dt2 <- as.data.frame(best_params_dt2)

best_params_dt[,2:3] <- as.numeric(c(0.1,128))
best_params_dt2[,2:3] <- as.numeric(c(0.1,128))


## Data ##
data_lob <- data_with_features_sd[data_with_features_sd$lob==sel_lob,]
keras_data_dt <- get_keras_data_lob(data_with_features_sd, sel_lob, company_index_recipe_sd, keep_case_reserves = 0, single_output = 0)
training_data_dt <- keras_data_dt$training_data
val_data_dt <- keras_data_dt$val_data
all_data_dt <- keras_data_dt$all_data

x_dt <- list(ay_seq_input=abind(training_data_dt$x$ay_seq_input, val_data_dt$x$ay_seq_input, along=1),
             company_input=abind(training_data_dt$x$company_input, val_data_dt$x$company_input, along=1))
y_dt <- abind(training_data_dt$y$paid_output,val_data_dt$y$paid_output,along=1)

keras_data_dt2 <- get_keras_data_lob(data_with_features_sd, sel_lob, company_index_recipe_sd, keep_case_reserves = 0, single_output = 1)
training_data_dt2 <- keras_data_dt2$training_data
val_data_dt2 <- keras_data_dt2$val_data
all_data_dt2 <- keras_data_dt2$all_data

x_dt2 <- list(ay_seq_input=abind(training_data_dt2$x$ay_seq_input, val_data_dt2$x$ay_seq_input, along=1),
             company_input=abind(training_data_dt2$x$company_input, val_data_dt2$x$company_input, along=1))
y_dt2 <- abind(training_data_dt2$y$paid_output,val_data_dt2$y$paid_output,along=1)
y_dt2 <- array_reshape(y_dt2, c(dim(y_dt2)[1],1))

preds_dt <- matrix(data=0,nrow=dim(all_data_dt$y$paid_output)[1],ncol=9)
preds_dt2 <- matrix(data=0,nrow=length(all_data_dt2$y$paid_output),ncol=1)

for(i in 1:5) {

  implement_dt <- model_dt_no_cr(best_params_dt$dropout, best_params_dt$neurons)
  
  print(paste0("Run: ",i,". Model: DT."))
  implement_dt %>%
    compile(
      optimizer = optimizer_adam(lr = 0.0005, amsgrad = TRUE),
      loss = list(masked_mse(-99))
    )
  
  fit_dt <- implement_dt %>% 
    fit(
      x = x_dt,
      y = y_dt,
      epochs = set_epochs,
      batch_size = dim(x_dt[[1]])[1],
      callbacks = list(callback_terminate_on_naan()),
      verbose = 1
    )
  
  preds_dt <- preds_dt + (implement_dt %>% predict(all_data_dt$x, verbose = 0))[,1,1]
  
  
  print(paste0("Run: ",i,". Model: DT2."))
  ## DT2 Model
  implement_dt2 <- model_dt_single(best_params_dt2$dropout, best_params_dt2$neurons)
  implement_dt2 %>%
    compile(
      optimizer = optimizer_adam(lr = 0.0005, amsgrad = TRUE),
      loss = loss_mean_squared_error
    )
  
  fit_dt2 <- implement_dt2 %>% 
    fit(
      x = list(x_dt2$ay_seq_input,x_dt2$company_input),
      y = y_dt2,
      epochs = set_epochs,
      batch_size = dim(x_dt2[[1]])[1],
      callbacks = list(callback_terminate_on_naan()),
      verbose = 1
    )
  
  preds_dt2 <- preds_dt2 + (implement_dt2 %>% predict(all_data_dt2$x, verbose = 0))
  
  
  
}

preds_dt <-PERM_preds_dt
preds_dt2 <-PERM_preds_dt2

preds_dt <- preds_dt[,1]
preds_dt <- cbind(data_lob$group_code,data_lob$accident_year-1987,data_lob$development_lag,(preds_dt*1/5*data_lob$inc_sd+data_lob$inc_mean),(data_lob$incremental_paid_actual*data_lob$inc_sd+data_lob$inc_mean))
preds_dt2 <- cbind(data_lob$group_code,data_lob$accident_year-1987,data_lob$development_lag,(preds_dt2*1/5*data_lob$inc_sd+data_lob$inc_mean),(data_lob$incremental_paid_actual*data_lob$inc_sd+data_lob$inc_mean))

colnames(preds_dt) <- c("index_company","AP","DP","DT_mean","loss")
colnames(preds_dt2) <-c("index_company","AP","DP","DT2_mean","loss")

fwrite(preds_dt,paste0("output_rd/",data_used,"/",data_used,"_DT.csv"))
fwrite(preds_dt2,paste0("output_rd/",data_used,"/",data_used,"_DT2.csv"))
fwrite(preds_dtf,paste0("output_rd/",data_used,"/",data_used,"_DTF.csv"))
