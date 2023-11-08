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
# Hyper-parameter selection for the DeepTriangle-based models for the real data scenarios

# Import necessary packages, models and support functions
source("models/model_dt_no_cr.R")
source("models/model_dt_single.R")

source("0.1 support_funcs.R")

#### Input and data preparation ####
## Output directory ##
output_dir <- "output_hp"
data_used <- "rd_ca"

## Model ##
sel_model <- model_dt_mdn
mse_weight <- 4
nll_weight <- 1

## Maximum number of epochs ##
epochs <- 100

## Initial parameters ##
initial_dropout <- 0
initial_neurons <- 16
initial_components <- 1

## Parameter grid search ##
#Specify range of values to test for each hyper-parameter
#The number of components will increase so long as the error decreases
sigmal2_range <- c(0,0.0001,0.001,0.01,0.1)
dropout_range <- c(0,0.1,0.2)
neurons_range <- c(16,24,32,48,64)
components_range <- c(1,2,3)

## Data ##
sel_lob <- "commercial_auto"
keras_data <- get_keras_data_lob(data_with_features_sd, sel_lob, company_index_recipe_sd, keep_case_reserves = 0, single_output = 1)
training_data <- keras_data$training_data
val_data <- keras_data$val_data
test_data <- keras_data$test_data


early_stop = callback_early_stopping(
  monitor = "val_loss",
  min_delta = 0,
  patience = 1000,
  verbose = 1,
  mode = c("auto", "min", "max"),
  baseline = NULL,
  restore_best_weights = TRUE
)

data_lob <- data_with_features_sd[data_with_features_sd$lob==sel_lob,]
keras_data_dt <- get_keras_data_lob(data_with_features_sd, sel_lob, company_index_recipe_sd, keep_case_reserves = 0, single_output = 0)
training_data_dt <- keras_data_dt$training_data
val_data_dt <- keras_data_dt$val_data
test_data_dt <- keras_data_dt$test_data
all_data_dt <- keras_data_dt$all_data

x_dt <- list(ay_seq_input=training_data_dt$x$ay_seq_input,
             company_input=training_data_dt$x$company_input)

keras_data_dt2 <- get_keras_data_lob(data_with_features_sd, sel_lob, company_index_recipe_sd, keep_case_reserves = 0, single_output = 1)
training_data_dt2 <- keras_data_dt2$training_data
val_data_dt2 <- keras_data_dt2$val_data
test_data_dt2 <- keras_data_dt2$test_data
all_data_dt2 <- keras_data_dt2$all_data

x_dt2 <- list(ay_seq_input=training_data_dt2$x$ay_seq_input,
              company_input=training_data_dt2$x$company_input)

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
  print(paste0("Current Dropout: ", dropout, ". Current model: DT."))
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
  print(paste0("Current Dropout: ", dropout, ". Current model: DT2."))
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
  print(paste0("Current Neurons: ", num_neurons, ". Current model: DT."))
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
  print(paste0("Current Neurons: ", num_neurons, ". Current model: DT2."))
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