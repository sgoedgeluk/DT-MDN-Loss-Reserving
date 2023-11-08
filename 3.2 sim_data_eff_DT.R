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
# Training and forecasting of the DeepTriangle-based models for the experiment focusing on 
# the effect of the amount of data on forecasting performance

# Import necessary packages, models and support functions
source("models/model_dt_no_cr.R")
source("models/model_dt_single.R")

source("0.1 support_funcs.R")

## Output directory
output_dir <- "output_sim_data"
data_used <- "sim_env1"
tot_triangles <- 200
num_triangles <- 1#c(5,10,25,50,100,160,200)
set_epochs <- 500

## Data
sel_env <- 1
raw_data_dir_y <- paste0("sim/env ",sel_env)
raw_data_dir_q <- paste0("sim/env ",sel_env," q")
triangles_y <- load_sim_triangles(raw_data_dir_y,200,10)
triangles_q <- load_sim_triangles(raw_data_dir_q,200,40)

list_data_sim_y <- construct_list_data_multi(triangles_y)
list_data_sim_q <- construct_list_data_multi(triangles_q)

early_stop = callback_early_stopping(
  min_delta = 0.001, 
  patience = 200, 
  mode = "min", 
  restore_best_weights = TRUE
)

#### EXPERIMENT: Quarters ####
num_periods <- 40
test_errors_list_q <- list()
preds_list_q <- list()

for(curr_num in num_triangles) {
  sel_data <- list_data_sim_q[list_data_sim_q$index_company <= curr_num,]
  norm_data <- normalize_sd(sel_data)
  mean <- norm_data$mean
  sd <- norm_data$sd
  norm_data <- norm_data$normalized_data
  
  keras_data <- construct_keras_data(norm_data,include_target = TRUE)
  keras_data <- split_keras_data(keras_data,target_multi = TRUE)
  train_data <- keras_data$train
  val_data <- keras_data$validation
  test_data <- keras_data$test
  all_data <- keras_data$all
  
  x_tmp <- list(ay_seq_input=train_data$x$ay_seq_input, company_input=train_data$x$company_input)
  val_tmp <- list(val_data$x,val_data$y)
  
  print(paste0("Current number of triangles: ",curr_num,"; Current model: DT."))
  start_time <- Sys.time()

  if(curr_num > 50) {
    company_num <- curr_num
  } else {
    company_num <- 50
  }
  
  ## DT Model
  curr_model <- model_dt_no_cr(num_companies = company_num)
  curr_model %>%
    compile(
      optimizer = optimizer_adam(lr = 0.0005, amsgrad = TRUE),
      loss = list(masked_mse(-99))
    )
  
  model_fit <- curr_model %>% 
    fit(
      x = x_tmp,
      y = train_data$y,
      epochs = set_epochs,
      batch_size = dim(x_tmp[[1]])[1],
      validation_data = val_tmp,
      callbacks = list(early_stop, callback_terminate_on_naan()),
      verbose = 1
    )
  
  test_error <- curr_model %>% evaluate(test_data$x, test_data$y, verbose = 0)
  test_errors_list_q <- append(test_errors_list_q,list(list(num_triangles=curr_num,model="DT",error=test_error)))
  
  preds <- curr_model %>% predict(all_data$x)

  preds_conv <- preds[,1,1]*sd+mean
  
  preds_conv <- cbind(sel_data$index_company,
                      sel_data$AP,
                      sel_data$DP,
                      preds_conv,
                      sel_data$value)
  colnames(preds_conv) <- c("company_input","AP","DP","DT_pred","loss")
  
  preds_list_q <- append(preds_list_q,list(list(num_triangles=curr_num,model="DT",preds=preds_conv)))
  
  fwrite(as.data.table(preds_conv), paste0(output_dir,"/",data_used,"_",curr_num,"_q_DT.csv"))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes."))

  
  ## DT2
  print(paste0("Current number of triangles: ",curr_num,"; Current model: DT2."))
  start_time <- Sys.time()
  
  sel_data <- list_data_sim_q[list_data_sim_q$index_company <= curr_num,]
  norm_data <- normalize_sd(sel_data)
  mean <- norm_data$mean
  sd <- norm_data$sd
  norm_data <- norm_data$normalized_data
  
  keras_data <- construct_keras_data(norm_data)
  keras_data <- split_keras_data(keras_data)
  train_data <- keras_data$train
  val_data <- keras_data$validation
  test_data <- keras_data$test
  all_data <- keras_data$all
  
  x_tmp <- list(ay_seq_input=train_data$x$ay_seq_input, company_input=train_data$x$company_input)
  val_tmp <- list(val_data$x,val_data$y)
  
  
  ## DT single output
  curr_model <- model_dt_single(num_companies = company_num)
  curr_model %>%
    compile(
      optimizer = optimizer_adam(),
      loss = loss_mean_squared_error
    )
  
  model_fit <- curr_model %>% 
    fit(
      x = x_tmp,
      y = train_data$y,
      epochs = set_epochs,
      batch_size = dim(x_tmp[[1]])[1],
      validation_data = val_tmp,
      callbacks = list(early_stop, callback_terminate_on_naan()),
      verbose = 1
    )
  
  test_error <- curr_model %>% evaluate(test_data$x, test_data$y, verbose = 0)
  test_errors_list_y <- append(test_errors_list_y,list(list(num_triangles=curr_num,model="DT2",error=test_error)))
  
  preds <- curr_model %>% predict(all_data$x)
  
  preds_conv <- preds*sd+mean
  
  preds_conv <- cbind(sel_data$index_company,
                      sel_data$AP,
                      sel_data$DP,
                      preds_conv,
                      sel_data$value)
  colnames(preds_conv) <- c("company_input","AP","DP","DT_pred","loss")
  
  preds_list_q <- append(preds_list_q,list(list(num_triangles=curr_num,model="DT2",preds=preds_conv)))
  
  fwrite(as.data.table(preds_conv), paste0(output_dir,"/",data_used,"_",curr_num,"_q_DT2.csv"))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes."))
  
}


#### EXPERIMENT: Years ####
num_periods <- 10
test_errors_list_y <- list()
preds_list_y <- list()
for(curr_num in num_triangles) {
  sel_data <- list_data_sim_y[list_data_sim_y$index_company <= curr_num,]
  norm_data <- normalize_sd(sel_data)
  mean <- norm_data$mean
  sd <- norm_data$sd
  norm_data <- norm_data$normalized_data
  
  keras_data <- construct_keras_data(norm_data,include_target = TRUE)
  keras_data <- split_keras_data(keras_data,target_multi = TRUE)
  train_data <- keras_data$train
  val_data <- keras_data$validation
  test_data <- keras_data$test
  all_data <- keras_data$all
  
  x_tmp <- list(ay_seq_input=train_data$x$ay_seq_input, company_input=train_data$x$company_input)
  val_tmp <- list(val_data$x,val_data$y)
  
  print(paste0("Current number of triangles: ",curr_num,"; Current model: DT."))
  start_time <- Sys.time()
  
  if(curr_num > 50) {
    company_num <- curr_num
  } else {
    company_num <- 50
  }
  
  ## DT Model
  curr_model <- model_dt_no_cr(num_companies = company_num)
  curr_model %>%
    compile(
      optimizer = optimizer_adam(lr = 0.0005, amsgrad = TRUE),
      loss = list(masked_mse(-99))
    )
  
  model_fit <- curr_model %>% 
    fit(
      x = x_tmp,
      y = train_data$y,
      epochs = set_epochs,
      batch_size = dim(x_tmp[[1]])[1],
      validation_data = val_tmp,
      callbacks = list(early_stop, callback_terminate_on_naan()),
      verbose = 1
    )
  
  test_error <- curr_model %>% evaluate(test_data$x, test_data$y, verbose = 0)
  test_errors_list_y <- append(test_errors_list_y,list(list(num_triangles=curr_num,model="DT",error=test_error)))
  
  preds <- curr_model %>% predict(all_data$x)
  
  preds_conv <- preds[,1,1]*sd+mean
  
  preds_conv <- cbind(sel_data$index_company,
                      sel_data$AP,
                      sel_data$DP,
                      preds_conv,
                      sel_data$value)
  colnames(preds_conv) <- c("company_input","AP","DP","DT_pred","loss")
  
  preds_list_y <- append(preds_list_y,list(list(num_triangles=curr_num,model="DT",preds=preds_conv)))
  
  fwrite(as.data.table(preds_conv), paste0(output_dir,"/",data_used,"_",curr_num,"_y_DT.csv"))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes."))
  
  
  
  ## DT2
  sel_data <- list_data_sim_y[list_data_sim_y$index_company <= curr_num,]
  norm_data <- normalize_sd(sel_data)
  mean <- norm_data$mean
  sd <- norm_data$sd
  norm_data <- norm_data$normalized_data
  
  keras_data <- construct_keras_data(norm_data)
  keras_data <- split_keras_data(keras_data)
  train_data <- keras_data$train
  val_data <- keras_data$validation
  test_data <- keras_data$test
  all_data <- keras_data$all
  
  x_tmp <- list(ay_seq_input=train_data$x$ay_seq_input, company_input=train_data$x$company_input)
  val_tmp <- list(val_data$x,val_data$y)
  
  print(paste0("Current number of triangles: ",curr_num,"; Current model: DT-MDN."))
  start_time <- Sys.time()
  # set_random_seed(curr_num)
  
  ## DT single output
  curr_model <- model_dt_single(num_companies = company_num)
  curr_model %>%
    compile(
      optimizer = optimizer_adam(),
      loss = loss_mean_squared_error
    )
  
  model_fit <- curr_model %>% 
    fit(
      x = x_tmp,
      y = train_data$y,
      epochs = set_epochs,
      batch_size = dim(x_tmp[[1]])[1],
      validation_data = val_tmp,
      callbacks = list(early_stop, callback_terminate_on_naan()),
      verbose = 1
    )
  
  test_error <- curr_model %>% evaluate(test_data$x, test_data$y, verbose = 0)
  test_errors_list_y <- append(test_errors_list_y,list(list(num_triangles=curr_num,model="DT2",error=test_error)))
  
  preds <- curr_model %>% predict(all_data$x)
  
  preds_conv <- preds*sd+mean
  
  preds_conv <- cbind(sel_data$index_company,
                      sel_data$AP,
                      sel_data$DP,
                      preds_conv,
                      sel_data$value)
  colnames(preds_conv) <- c("company_input","AP","DP","DT_pred","loss")
  
  preds_list_y <- append(preds_list_y,list(list(num_triangles=curr_num,model="DT2",preds=preds_conv)))
  
  fwrite(as.data.table(preds_conv), paste0(output_dir,"/",data_used,"_",curr_num,"_y_DT2.csv"))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes."))
}
