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
# Final training of the DT-based models for the simulated data environments

# Import necessary packages, models and support functions
source("models/model_dt_no_cr.R")
source("models/model_dt_single.R")

source("0.1 support_funcs.R")

sim_env <- 1

#### Best params ####
params_list_DT <- fread(paste0("output_sim_full/env_",sim_env,"/y_DT_test_scores_neurons.csv"))

best_params_dt  <- params_list_DT[(params_list_DT$test_score == min(params_list_DT[params_list_DT$model=="DT",]$test_score) 
                                                              & params_list_DT$model=="DT"),]
best_params_dt2  <- params_list_DT[(params_list_DT$test_score == min(params_list_DT[params_list_DT$model=="DT2",]$test_score) 
                                                           & params_list_DT$model=="DT2"),]


#### Data General ####
num_triangles <- 50
set_epochs <- 500
num_periods <- 10

output_dir <- paste0("output_sim_full/env_",sim_env)
raw_data_dir <- paste0("sim/env ",sim_env)
triangles <- load_sim_triangles(raw_data_dir,num_triangles,num_periods)
list_data_sim <- construct_list_data_multi(triangles)

#### Data for DT ####
norm_data_dt <- normalize_sd(list_data_sim)
mean_dt <- norm_data_dt$mean
sd_dt <- norm_data_dt$sd
norm_data_dt <- norm_data_dt$normalized_data

keras_data_dt <- construct_keras_data(norm_data_dt,include_target = TRUE)
keras_data_dt <- split_keras_data(keras_data_dt,target_multi = TRUE)
train_data_dt <- keras_data_dt$train
val_data_dt <- keras_data_dt$validation
test_data_dt <- keras_data_dt$test
all_data_dt <- keras_data_dt$all


x_dt <- list(ay_seq_input=abind(train_data_dt$x$ay_seq_input, val_data_dt$x$ay_seq_input, along=1),
            company_input=abind(train_data_dt$x$company_input, val_data_dt$x$company_input, along=1))
y_dt <- abind(train_data_dt$y,val_data_dt$y,along=1)


#### Data for DT2 ####
keras_data_dt2 <- construct_keras_data(norm_data_dt)
keras_data_dt2 <- split_keras_data(keras_data_dt2)
train_data_dt2 <- keras_data_dt2$train
val_data_dt2 <- keras_data_dt2$validation
test_data_dt2 <- keras_data_dt2$test
all_data_dt2 <- keras_data_dt2$all

x_dt2 <- list(ay_seq_input=abind(train_data_dt2$x$ay_seq_input, val_data_dt2$x$ay_seq_input, along=1),
             company_input=abind(train_data_dt2$x$company_input, val_data_dt2$x$company_input, along=1))
y_dt2 <- abind(train_data_dt2$y,val_data_dt2$y,along=1)

preds_dt <- matrix(data=0,nrow=length(all_data_dt$y[,1,1]),ncol=1)
preds_dt2 <- matrix(data=0,nrow=length(all_data_dt2$y),ncol=1)

for(i in 1:1) {
  ## DT- Model
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
      x = x_dt2,
      y = y_dt2,
      epochs = set_epochs,
      batch_size = dim(x_dt2[[1]])[1],
      callbacks = list(callback_terminate_on_naan()),
      verbose = 1
    )
  
  preds_dt2 <- preds_dt2 + (implement_dt2 %>% predict(all_data_dt2$x, verbose = 0))
}

preds_dt <- cbind(list_data_sim$index_company,list_data_sim$AP,list_data_sim$DP,preds_dt*1/5*sd_dt+mean_dt,list_data_sim$value)
preds_dt2 <- cbind(list_data_sim$index_company,list_data_sim$AP,list_data_sim$DP,preds_dt2*1/5*sd_dt+mean_dt,list_data_sim$value)

colnames(preds_dt) <- c("index_company","AP","DP","DT_pred","loss")
colnames(preds_dt2) <- c("index_company","AP","DP","DT_pred","loss")

fwrite(preds_dt,paste0(output_dir,"/","sim_env",sim_env,"_DT.csv"))
fwrite(preds_dt2,paste0(output_dir,"/","sim_env",sim_env,"_DT2.csv"))
