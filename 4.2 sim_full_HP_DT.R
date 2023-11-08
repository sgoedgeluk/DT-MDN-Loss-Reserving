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
# Hyper-parameter selection for the DeepTriangle-based models for the simulated environments

source("models/model_dt_no_cr.R")
source("models/model_dt_single.R")

source("0.1 support_funcs.R")

## Input
num_triangles <- 50
set_epochs <- 500

initial_dropout <- 0
initial_neurons <- 16

dropout_range <- c(0,0.1,0.2)
neurons_range <- c(32,64,128,256)

num_periods <- 10
for(sim_env in 1:4) {
  print(paste0("Current env: ",sim_env))
  ## Output directory ##
  output_dir <- paste0("output_sim_full/env_",sim_env)
  data_used <- paste0("sim_env",sim_env)
  
  
  ## Data general 
  raw_data_dir <- paste0("sim/env ",sim_env)
  triangles <- load_sim_triangles(raw_data_dir,num_triangles,num_periods)
  list_data_sim <- construct_list_data_multi(triangles)
  
  
  ## Data for DT 
  norm_data <- normalize_sd(list_data_sim)
  mean <- norm_data$mean
  sd <- norm_data$sd
  norm_data <- norm_data$normalized_data
  
  keras_data_dt <- construct_keras_data(norm_data,include_target = TRUE)
  keras_data_dt <- split_keras_data(keras_data_dt,target_multi = TRUE)
  train_data_dt <- keras_data_dt$train
  val_data_dt <- keras_data_dt$validation
  test_data_dt <- keras_data_dt$test
  all_data_dt <- keras_data_dt$all
  
  x_dt <- list(ay_seq_input=train_data_dt$x$ay_seq_input, company_input=train_data_dt$x$company_input)
  val_dt <- list(val_data_dt$x,val_data_dt$y)
  
  ## Data for DT2
  keras_data_dt2 <- construct_keras_data(norm_data)
  keras_data_dt2 <- split_keras_data(keras_data_dt2)
  train_data_dt2 <- keras_data_dt2$train
  val_data_dt2 <- keras_data_dt2$validation
  test_data_dt2 <- keras_data_dt2$test
  all_data_dt2 <- keras_data_dt2$all
  
  x_dt2 <- list(ay_seq_input=train_data_dt2$x$ay_seq_input, company_input=train_data_dt2$x$company_input)
  val_dt2 <- list(val_data_dt2$x,val_data_dt2$y)
  
  ## Additional reqs
  early_stop = callback_early_stopping(
    min_delta = 0.001, 
    patience = 200, 
    mode = "min", 
    restore_best_weights = TRUE
  )
  
  test_scores_dropout <- data.frame(model=c(),dropout=c(),neurons=c(),test_score=c())
  test_scores_neurons <- data.frame(model=c(),dropout=c(),neurons=c(),test_score=c())
  test_scores_all <- data.frame(model=c(),dropout=c(),neurons=c(),test_score=c())
  
  #### Dropout ####
  for (i in 1:length(dropout_range)){
    dropout <- dropout_range[i]
    print(paste0("Current dropout rate: ",dropout))
    
    ## DT Model
    print(paste0("Current env: ",sim_env,". Current Dropout: ", dropout, ". Current model: DT."))
    start_time <- Sys.time()

    implement_dt <- model_dt_no_cr(dropout, initial_neurons)
    implement_dt %>%
      compile(
        optimizer = optimizer_adam(lr = 0.0005, amsgrad = TRUE),
        loss = list(masked_mse(-99))
      )
    
    fit_dt <- implement_dt %>% 
      fit(
        x = x_dt,
        y = train_data_dt$y,
        epochs = set_epochs,
        batch_size = dim(x_dt[[1]])[1],
        validation_data = val_dt,
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- implement_dt %>% evaluate(test_data_dt$x, test_data_dt$y, verbose = 0)
    
    test_scores_dropout <- rbind(test_scores_dropout,c("DT",dropout,initial_neurons,test_error))
    test_scores_all <- rbind(test_scores_all,c("DT",dropout,initial_neurons,test_error))
    
    end_time <- Sys.time()
    print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
    
    
    
    ## DT2 Model
    print(paste0("Current env: ",sim_env,". Current Dropout: ", dropout, ". Current model: DT2."))
    start_time <- Sys.time()
    
    implement_dt2 <- model_dt_single(dropout, initial_neurons)
    implement_dt2 %>%
      compile(
        optimizer = optimizer_adam(lr = 0.0005, amsgrad = TRUE),
        loss = loss_mean_squared_error
      )
    
    fit_dt2 <- implement_dt2 %>% 
      fit(
        x = x_dt2,
        y = train_data_dt2$y,
        epochs = set_epochs,
        batch_size = dim(x_dt2[[1]])[1],
        validation_data = val_dt2,
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- implement_dt2 %>% evaluate(test_data_dt2$x, test_data_dt2$y, verbose = 0)
    
    test_scores_dropout <- rbind(test_scores_dropout,c("DT2",dropout,initial_neurons,test_error))
    test_scores_all <- rbind(test_scores_all,c("DT2",dropout,initial_neurons,test_error))
    
    end_time <- Sys.time()
    print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
    
  }
  
  colnames(test_scores_dropout) <- c("model","dropout","neurons","test_score")
  colnames(test_scores_all) <- c("model","dropout","neurons","test_score")
  
  best_dropout_dt  <- as.numeric(test_scores_dropout[(test_scores_dropout$test_score == min(test_scores_dropout[test_scores_dropout$model=="DT",]$test_score) 
                                                          & test_scores_dropout$model=="DT"),
                                                         "dropout"])
  best_dropout_dt2  <- as.numeric(test_scores_dropout[(test_scores_dropout$test_score == min(test_scores_dropout[test_scores_dropout$model=="DT2",]$test_score) 
                                                       & test_scores_dropout$model=="DT2"),
                                                      "dropout"])

  
  
  
  
  #### Neurons ####
  for (i in 1:length(neurons_range)){
    num_neurons <- neurons_range[i]
    print(paste0("Current neurons: ",num_neurons))
    
    ## DT Model
    print(paste0("Current env: ",sim_env,". Current Neurons: ", num_neurons, ". Current model: DT."))
    start_time <- Sys.time()

    implement_dt <- model_dt_no_cr(best_dropout_dt, num_neurons)
    implement_dt %>%
      compile(
        optimizer = optimizer_adam(lr = 0.0005, amsgrad = TRUE),
        loss = list(masked_mse(-99))
      )
    
    fit_dt <- implement_dt %>% 
      fit(
        x = x_dt,
        y = train_data_dt$y,
        epochs = set_epochs,
        batch_size = dim(x_dt[[1]])[1],
        validation_data = val_dt,
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- implement_dt %>% evaluate(test_data_dt$x, test_data_dt$y, verbose = 0)
    
    test_scores_neurons <- rbind(test_scores_neurons,c("DT",best_dropout_dt,num_neurons,test_error))
    test_scores_all <- rbind(test_scores_all,c("DT",best_dropout_dt,num_neurons,test_error))
    
    end_time <- Sys.time()
    print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
    
    
    
    ## DT2 Model
    print(paste0("Current env: ",sim_env,". Current Neurons: ", num_neurons, ". Current model: DT2."))
    start_time <- Sys.time()
    
    implement_dt2 <- model_dt_single(best_dropout_dt2, num_neurons)
    implement_dt2 %>%
      compile(
        optimizer = optimizer_adam(lr = 0.0005, amsgrad = TRUE),
        loss = loss_mean_squared_error
      )
    
    fit_dt2 <- implement_dt2 %>% 
      fit(
        x = x_dt2,
        y = train_data_dt2$y,
        epochs = set_epochs,
        batch_size = dim(x_dt2[[1]])[1],
        validation_data = val_dt2,
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- implement_dt2 %>% evaluate(test_data_dt2$x, test_data_dt2$y, verbose = 0)
    
    test_scores_neurons <- rbind(test_scores_neurons,c("DT2",best_dropout_dt2,num_neurons,test_error))
    test_scores_all <- rbind(test_scores_all,c("DT2",best_dropout_dt2,num_neurons,test_error))
    
    end_time <- Sys.time()
    print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
  }
  
  colnames(test_scores_neurons) <- c("model","dropout","neurons","test_score")
  colnames(test_scores_all) <- c("model","dropout","neurons","test_score")
  
  fwrite(test_scores_dropout,paste0(output_dir,"/y_DT_test_scores_dropout.csv"))
  fwrite(test_scores_neurons,paste0(output_dir,"/y_DT_test_scores_neurons.csv"))
  fwrite(test_scores_all,paste0(output_dir,"/y_DT_test_scores_all.csv"))
}
