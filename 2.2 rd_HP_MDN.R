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
# Hyper-parameter selection for the MDN-based models for the real data scenarios

# Import necessary packages, models and support functions
source("models/model_mdn_org.R")
source("models/model_dt_mdn.R")

source("0.1 support_funcs.R")
source("2.1 rd_keras.R")

#### Input and data preparation ####
## Output directory ##
data_used <- "rd_ol"
out_dir <- paste0("output_rd/",data_used)

## Model loss weights ##
mse_weight <- 4
nll_weight <- 1

## Maximum number of epochs ##
set_epochs <- 500

## Initial parameters
initial_sigmal2 <- 0
initial_dropout <- 0
initial_hidden <- 1
initial_components <- 1
initial_neurons <- 16

## Parameter grid search
#Specify range of values to test for each hyper-parameter
#The number of components will increase so long as the error decreases
netl2_range <- c(0,0.001,0.01)
sigmal2_range <- c(0,0.001,0.01,0.1)
dropout_range <- c(0,0.1,0.2)
num_hidden_range <- c(1,2)
neurons_range <- c(24,32,48,64)
components_range <- c(2,3)

## Prepare data
sel_lob <- "other_liability"
data_lob <- data_with_features_sd[data_with_features_sd$lob==sel_lob,]
keras_data <- get_keras_data_lob(data_with_features_sd, sel_lob, company_index_recipe_sd, keep_case_reserves = 0, single_output = 1)
training_data <- keras_data$training_data
val_data <- keras_data$val_data
test_data <- keras_data$test_data

# Split the data in different buckets for train, validation and test, for both the 
# DT-MDN and MDN model
train_x_dt <- training_data$x
train_y_dt <- training_data$y$paid_output

val_x_dt <- val_data$x
val_y_dt <- val_data$y$paid_output

test_x_dt <- test_data$x
test_y_dt <- test_data$y$paid_output

indices_all <- cbind(data_with_features_sd[(data_with_features_sd$lob==sel_lob),]$accident_year-1987,
                     data_with_features_sd[(data_with_features_sd$lob==sel_lob),]$development_lag,
                     data_with_features_sd[(data_with_features_sd$lob==sel_lob),]$bucket)

indices_all <- as.data.frame(indices_all)
indices_all[,1:2] <- lapply(indices_all[,1:2],as.numeric)
colnames(indices_all) <- c("AP","DP","bucket")

indices_mean <- colMeans(indices_all[,1:2])
indices_sd <- unlist(lapply(indices_all[,1:2],sd))

indices_all[,1:2] <- (indices_all[,1:2]-indices_mean)/indices_sd

train_x_mdn <- as.matrix(indices_all[indices_all$bucket %in% c("train"),1:2])
train_y_mdn <- training_data$y$paid_output

val_x_mdn <- as.matrix(indices_all[indices_all$bucket %in% c("validation"),1:2])
val_y_mdn <- val_data$y$paid_output

test_x_mdn <- as.matrix(indices_all[indices_all$bucket %in% c("test"),1:2])
test_y_mdn <- test_data$y$paid_output

# Early stopping criteria based on the validation loss
early_stop = callback_early_stopping(
  monitor = "val_loss",
  min_delta = 0,
  patience = 1000,
  verbose = 1,
  mode = c("auto", "min", "max"),
  baseline = NULL,
  restore_best_weights = TRUE
)

# Start hyperparameter selection
test_scores_net = data.frame(model=c(),net=c(),sigma=c(),dropout=c(),num_hidden=c(),neurons=c(),components=c(),test_score=c())
test_scores_sigma = data.frame(model=c(),net=c(),sigma=c(),dropout=c(),num_hidden=c(),neurons=c(),components=c(),test_score=c())
test_scores_dropout = data.frame(model=c(),net=c(),sigma=c(),dropout=c(),num_hidden=c(),neurons=c(),components=c(),test_score=c())
test_scores_hidden = data.frame(model=c(),net=c(),sigma=c(),dropout=c(),num_hidden=c(),neurons=c(),components=c(),test_score=c())
test_scores_neurons = data.frame(model=c(),net=c(),sigma=c(),dropout=c(),num_hidden=c(),neurons=c(),components=c(),test_score=c())
test_scores_components = data.frame(model=c(),net=c(),sigma=c(),dropout=c(),num_hidden=c(),neurons=c(),components=c(),test_score=c())
test_scores_all = data.frame(model=c(),sigma=c(),net=c(),dropout=c(),num_hidden=c(),neurons=c(),components=c(),test_score=c())

#### NetL2 ####
for (i in 1:length(netl2_range)){
  netl2 <- netl2_range[i]
  print(paste0("Current netl2: ",netl2))
  
  ## MDN Model
  print(paste0("Current model: MDN"))
  start_time <- Sys.time()
  
  test_error <- 0
  for(company in unique(data_lob$group_code)) {
    tmp_train_x_mdn <- train_x_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company,]
    
    tmp_train_y_mdn <- train_y_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company]
    
    tmp_val_x_mdn <- val_x_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company,]
    tmp_val_y_mdn <- val_y_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company]
  
    tmp_test_x_mdn <- test_x_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company,]
    tmp_test_y_mdn <- test_y_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company]
    
    print(paste0("Current model: MDN"))
    start_time <- Sys.time()
    set_random_seed(0)
    
    implement_mdn <- model_mdn_org(netl2, initial_sigmal2, initial_dropout, initial_neurons, initial_hidden, initial_components)
    implement_mdn %>%
      compile(
        optimizer = 'adam',
        loss = NLLcustom
      )
    
    fit_mdn <- implement_mdn %>% 
      fit(
        x = tmp_train_x_mdn,
        y = tmp_train_y_mdn,
        epochs = set_epochs,
        batch_size = dim(tmp_train_x_mdn)[1],
        validation_data = list(tmp_val_x_mdn,tmp_val_y_mdn),
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_y_mdn, verbose = 0)
  }
  
  test_scores_net <- rbind(test_scores_net,c("MDN",netl2,initial_sigmal2,initial_dropout,initial_hidden,initial_neurons,initial_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("MDN",netl2,initial_sigmal2,initial_dropout,initial_hidden,initial_neurons,initial_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
}

colnames(test_scores_net) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")
colnames(test_scores_all) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")

best_netl2_mdn  <- as.numeric(test_scores_net[(test_scores_net$test_score == min(test_scores_net[test_scores_net$model=="MDN",]$test_score) 
                                               & test_scores_net$model=="MDN"),
                                              "netl2"])

#### SigmaL2 ####
for (i in 1:length(sigmal2_range)){
  sigmal2 <- sigmal2_range[i]
  print(paste0("Current sigma: ",sigmal2))
  
  ## DT-MDN Model
  print(paste0("Current sigma: ",sigmal2, ". Current model: DT-MDN"))
  start_time <- Sys.time()
  set_random_seed(0)
  
  implement_dt_mdn <- model_dt_mdn(sigmal2, initial_dropout, initial_neurons, initial_components)
  implement_dt_mdn %>%
    compile(
      optimizer = 'adam',
      loss = NLLcustom
    )
  
  fit_dt_mdn <- implement_dt_mdn %>% 
    fit(
      x = train_x_dt,
      y = train_y_dt,
      epochs = set_epochs,
      batch_size = dim(train_x_dt[[1]])[1],
      validation_data = list(val_x_dt,val_y_dt),
      callbacks = list(early_stop, callback_terminate_on_naan()),
      verbose = 1
    )
  
  test_error <- implement_dt_mdn %>% evaluate(test_x_dt, test_y_dt, verbose = 0)
  
  test_scores_sigma <- rbind(test_scores_sigma,c("DT_MDN",NA, sigmal2,initial_dropout,initial_hidden,initial_neurons,initial_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("DT_MDN",NA, sigmal2,initial_dropout,initial_hidden,initial_neurons,initial_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
  
  ## MDN Model
  test_error <- 0
  for(company in unique(data_lob$group_code)) {
    tmp_train_x_mdn <- train_x_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company,]
    
    tmp_train_y_mdn <- train_y_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company]
    
    tmp_val_x_mdn <- val_x_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company,]
    tmp_val_y_mdn <- val_y_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company]
    
    tmp_test_x_mdn <- test_x_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company,]
    tmp_test_y_mdn <- test_y_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company]
    
    print(paste0("Current model: MDN"))
    start_time <- Sys.time()
    set_random_seed(0)
    
    implement_mdn <- model_mdn_org(best_netl2_mdn, sigmal2, initial_dropout, initial_neurons, initial_hidden, initial_components)
    implement_mdn %>%
      compile(
        optimizer = 'adam',
        loss = NLLcustom
      )
    
    fit_mdn <- implement_mdn %>% 
      fit(
        x = tmp_train_x_mdn,
        y = tmp_train_y_mdn,
        epochs = set_epochs,
        batch_size = dim(tmp_train_x_mdn)[1],
        validation_data = list(tmp_val_x_mdn,tmp_val_y_mdn),
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_y_mdn, verbose = 0)
  }
  test_scores_sigma <- rbind(test_scores_sigma,c("MDN",best_netl2_mdn, sigmal2,initial_dropout,initial_hidden,initial_neurons,initial_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("MDN",best_netl2_mdn, sigmal2,initial_dropout,initial_hidden,initial_neurons,initial_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
}

colnames(test_scores_sigma) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")
colnames(test_scores_all) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")

best_sigma_dt_mdn  <- as.numeric(test_scores_sigma[(test_scores_sigma$test_score == min(test_scores_sigma[test_scores_sigma$model=="DT_MDN",]$test_score) 
                                                    & test_scores_sigma$model=="DT_MDN"),
                                                   "sigma"])
best_sigma_mdn  <- as.numeric(test_scores_sigma[(test_scores_sigma$test_score == min(test_scores_sigma[test_scores_sigma$model=="MDN",]$test_score) 
                                                 & test_scores_sigma$model=="MDN"),
                                                "sigma"])


#### Dropout ####
for (i in 1:length(dropout_range)){
  dropout <- dropout_range[i]
  print(paste0("Current dropout rate: ",dropout))
  
  ## DT-MDN Model
  print(paste0("Current model: DT-MDN"))
  start_time <- Sys.time()
  set_random_seed(0)
  
  implement_dt_mdn <- model_dt_mdn(best_sigma_dt_mdn, dropout, initial_neurons, initial_components)
  implement_dt_mdn %>%
    compile(
      optimizer = 'adam',
      loss = NLLcustom
    )
  
  fit_dt_mdn <- implement_dt_mdn %>% 
    fit(
      x = train_x_dt,
      y = train_y_dt,
      epochs = set_epochs,
      batch_size = dim(train_x_dt[[1]])[1],
      validation_data = list(val_x_dt,val_y_dt),
      callbacks = list(early_stop, callback_terminate_on_naan()),
      verbose = 1
    )
  
  test_error <- implement_dt_mdn %>% evaluate(test_x_dt, test_y_dt, verbose = 0)
  
  test_scores_dropout <- rbind(test_scores_dropout,c("DT_MDN",NA, best_sigma_dt_mdn,dropout,initial_hidden,initial_neurons,initial_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("DT_MDN",NA, best_sigma_dt_mdn,dropout,initial_hidden,initial_neurons,initial_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
  
  
  ## MDN Model
  test_error <- 0
  for(company in unique(data_lob$group_code)) {
    tmp_train_x_mdn <- train_x_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company,]
    
    tmp_train_y_mdn <- train_y_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company]
    
    tmp_val_x_mdn <- val_x_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company,]
    tmp_val_y_mdn <- val_y_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company]
    
    tmp_test_x_mdn <- test_x_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company,]
    tmp_test_y_mdn <- test_y_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company]
    
    print(paste0("Current model: MDN, company ",company,", index ",i))
    start_time <- Sys.time()
    set_random_seed(0)
    
    implement_mdn <- model_mdn_org(best_netl2_mdn, best_sigma_mdn, dropout, initial_neurons, initial_hidden, initial_components)
    implement_mdn %>%
      compile(
        optimizer = 'adam',
        loss = NLLcustom
      )
    
    fit_mdn <- implement_mdn %>% 
      fit(
        x = tmp_train_x_mdn,
        y = tmp_train_y_mdn,
        epochs = set_epochs,
        batch_size = dim(tmp_train_x_mdn)[1],
        validation_data = list(tmp_val_x_mdn,tmp_val_y_mdn),
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_y_mdn, verbose = 0)
  }
  
  test_scores_dropout <- rbind(test_scores_dropout,c("MDN",best_netl2_mdn, best_sigma_mdn,dropout,initial_hidden,initial_neurons,initial_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("MDN",best_netl2_mdn, best_sigma_mdn,dropout,initial_hidden,initial_neurons,initial_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
}

colnames(test_scores_dropout) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")
colnames(test_scores_all) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")


best_dropout_dt_mdn  <- as.numeric(test_scores_dropout[(test_scores_dropout$test_score == min(test_scores_dropout[test_scores_dropout$model=="DT_MDN",]$test_score) 
                                                        & test_scores_dropout$model=="DT_MDN"),
                                                       "dropout"])
best_dropout_mdn  <- as.numeric(test_scores_dropout[(test_scores_dropout$test_score == min(test_scores_dropout[test_scores_dropout$model=="MDN",]$test_score) 
                                                     & test_scores_dropout$model=="MDN"),
                                                    "dropout"])

#### Hidden Layers ####
for (i in 1:length(num_hidden_range)){
  num_hidden <- num_hidden_range[i]
  print(paste0("Current number of hidden layers: ",num_hidden))
  
  ## MDN Model
  print(paste0("Current model: MDN"))
  start_time <- Sys.time()
  test_error <- 0
  for(company in unique(data_lob$group_code)) {
    tmp_train_x_mdn <- train_x_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company,]
    
    tmp_train_y_mdn <- train_y_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company]
    
    tmp_val_x_mdn <- val_x_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company,]
    tmp_val_y_mdn <- val_y_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company]
    
    tmp_test_x_mdn <- test_x_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company,]
    tmp_test_y_mdn <- test_y_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company]
    
    print(paste0("Current model: MDN, company ",company,", index ",i))
    start_time <- Sys.time()
    set_random_seed(0)
    
    implement_mdn <- model_mdn_org(best_netl2_mdn, best_sigma_mdn, best_dropout_mdn, initial_neurons, num_hidden, initial_components)
    implement_mdn %>%
      compile(
        optimizer = 'adam',
        loss = NLLcustom
      )
    
    fit_mdn <- implement_mdn %>% 
      fit(
        x = tmp_train_x_mdn,
        y = tmp_train_y_mdn,
        epochs = set_epochs,
        batch_size = dim(tmp_train_x_mdn)[1],
        validation_data = list(tmp_val_x_mdn,tmp_val_y_mdn),
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_y_mdn, verbose = 0)
  }
  
  test_scores_hidden <- rbind(test_scores_hidden,c("MDN",best_netl2_mdn, best_sigma_mdn,best_dropout_mdn,num_hidden,initial_neurons,initial_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("MDN",best_netl2_mdn, best_sigma_mdn,best_dropout_mdn,num_hidden,initial_neurons,initial_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
}

colnames(test_scores_hidden) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")
colnames(test_scores_all) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")

best_hidden_mdn  <- as.numeric(test_scores_hidden[(test_scores_hidden$test_score == min(test_scores_hidden[test_scores_hidden$model=="MDN",]$test_score) 
                                                   & test_scores_hidden$model=="MDN"),
                                                  "num_hidden"])

#### Neurons ####
for (i in 1:length(neurons_range)){
  num_neurons <- neurons_range[i]
  print(paste0("Current number of neurons: ",num_neurons))
  
  ## DT-MDN Model
  print(paste0("Current model: DT-MDN"))
  start_time <- Sys.time()
  set_random_seed(0)
  
  implement_dt_mdn <- model_dt_mdn(best_sigma_dt_mdn, best_dropout_dt_mdn, num_neurons, initial_components)
  implement_dt_mdn %>%
    compile(
      optimizer = 'adam',
      loss = NLLcustom
    )
  
  fit_dt_mdn <- implement_dt_mdn %>% 
    fit(
      x = train_x_dt,
      y = train_y_dt,
      epochs = set_epochs,
      batch_size = dim(train_x_dt[[1]])[1],
      validation_data = list(val_x_dt,val_y_dt),
      callbacks = list(early_stop, callback_terminate_on_naan()),
      verbose = 1
    )
  
  test_error <- implement_dt_mdn %>% evaluate(test_x_dt, test_y_dt, verbose = 0)
  
  test_scores_neurons <- rbind(test_scores_neurons,c("DT_MDN",NA, best_sigma_dt_mdn,best_dropout_dt_mdn,NA,num_neurons,initial_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("DT_MDN",NA, best_sigma_dt_mdn,best_dropout_dt_mdn,NA,num_neurons,initial_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
  
  ## MDN Model
  test_error <- 0
  for(company in unique(data_lob$group_code)) {
    tmp_train_x_mdn <- train_x_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company,]
    
    tmp_train_y_mdn <- train_y_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company]
    
    tmp_val_x_mdn <- val_x_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company,]
    tmp_val_y_mdn <- val_y_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company]
    
    tmp_test_x_mdn <- test_x_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company,]
    tmp_test_y_mdn <- test_y_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company]
    
    print(paste0("Current model: MDN"))
    start_time <- Sys.time()
    set_random_seed(0)
    
    implement_mdn <- model_mdn_org(best_netl2_mdn, best_sigma_mdn, best_dropout_mdn, num_neurons, best_hidden_mdn, initial_components)
    implement_mdn %>%
      compile(
        optimizer = 'adam',
        loss = NLLcustom
      )
    
    fit_mdn <- implement_mdn %>% 
      fit(
        x = tmp_train_x_mdn,
        y = tmp_train_y_mdn,
        epochs = set_epochs,
        batch_size = dim(tmp_train_x_mdn)[1],
        validation_data = list(tmp_val_x_mdn,tmp_val_y_mdn),
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_y_mdn, verbose = 0)
  }
  test_scores_neurons <- rbind(test_scores_neurons,c("MDN",best_netl2_mdn, best_sigma_mdn,best_dropout_mdn,best_hidden_mdn,num_neurons,initial_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("MDN",best_netl2_mdn, best_sigma_mdn,best_dropout_mdn,best_hidden_mdn,num_neurons,initial_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
}

colnames(test_scores_neurons) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")
colnames(test_scores_all) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")


best_neurons_dt_mdn  <- as.numeric(test_scores_neurons[(test_scores_neurons$test_score == min(test_scores_neurons[test_scores_neurons$model=="DT_MDN",]$test_score) 
                                                        & test_scores_neurons$model=="DT_MDN"),
                                                       "neurons"])
best_neurons_mdn  <- as.numeric(test_scores_neurons[(test_scores_neurons$test_score == min(test_scores_neurons[test_scores_neurons$model=="MDN",]$test_score) 
                                                     & test_scores_neurons$model=="MDN"),
                                                    "neurons"])

#### Components ####
for (i in 1:length(components_range)){
  num_components <- components_range[i]
  print(paste0("Current number of components: ",num_components))
  
  ## DT-MDN Model
  print(paste0("Current model: DT-MDN"))
  start_time <- Sys.time()
  set_random_seed(0)
  
  implement_dt_mdn <- model_dt_mdn(best_sigma_dt_mdn, best_dropout_dt_mdn, best_neurons_dt_mdn, num_components)
  implement_dt_mdn %>%
    compile(
      optimizer = 'adam',
      loss = NLLcustom
    )
  
  fit_dt_mdn <- implement_dt_mdn %>% 
    fit(
      x = train_x_dt,
      y = train_y_dt,
      epochs = set_epochs,
      batch_size = dim(train_x_dt[[1]])[1],
      validation_data = list(val_x_dt,val_y_dt),
      callbacks = list(early_stop, callback_terminate_on_naan()),
      verbose = 1
    )
  
  test_error <- implement_dt_mdn %>% evaluate(test_x_dt, test_y_dt, verbose = 0)
  
  test_scores_components <- rbind(test_scores_components,c("DT_MDN",NA, best_sigma_dt_mdn,best_dropout_dt_mdn,NA,best_neurons_dt_mdn,num_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("DT_MDN",NA, best_sigma_dt_mdn,best_dropout_dt_mdn,NA,best_neurons_dt_mdn,num_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
  

  ## MDN Model
  test_error <- 0
  for(company in unique(data_lob$group_code)) {
    tmp_train_x_mdn <- train_x_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company,]
    
    tmp_train_y_mdn <- train_y_mdn[data_lob[data_lob$bucket %in% c("train"),]$group_code==company]
    
    tmp_val_x_mdn <- val_x_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company,]
    tmp_val_y_mdn <- val_y_mdn[data_lob[data_lob$bucket %in% c("validation"),]$group_code==company]
    
    tmp_test_x_mdn <- test_x_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company,]
    tmp_test_y_mdn <- test_y_mdn[data_lob[data_lob$bucket %in% c("test"),]$group_code==company]
    
    print(paste0("Current model: MDN"))
    start_time <- Sys.time()
    set_random_seed(0)

    implement_mdn <- model_mdn_org(best_netl2_mdn, best_sigma_mdn, best_dropout_mdn, best_neurons_mdn, best_hidden_mdn, num_components)
    implement_mdn %>%
      compile(
        optimizer = 'adam',
        loss = NLLcustom
      )
    
    fit_mdn <- implement_mdn %>% 
      fit(
        x = tmp_train_x_mdn,
        y = tmp_train_y_mdn,
        epochs = set_epochs,
        batch_size = dim(tmp_train_x_mdn)[1],
        validation_data = list(tmp_val_x_mdn,tmp_val_y_mdn),
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_y_mdn, verbose = 0)
  }
  
  test_scores_components <- rbind(test_scores_components,c("MDN",best_netl2_mdn, best_sigma_mdn,best_dropout_mdn,best_hidden_mdn,best_neurons_mdn,num_components,test_error))
  test_scores_all <- rbind(test_scores_all,c("MDN",best_netl2_mdn, best_sigma_mdn,best_dropout_mdn,best_hidden_mdn,best_neurons_mdn,num_components,test_error))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
}

colnames(test_scores_components) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")
colnames(test_scores_all) <- c("model","netl2","sigma","dropout","num_hidden","neurons","components","test_score")


best_components_dt_mdn  <- as.numeric(test_scores_components[(test_scores_components$test_score == min(test_scores_components[test_scores_components$model=="DT_MDN",]$test_score) 
                                                              & test_scores_components$model=="DT_MDN"),
                                                             "components"])
best_components_mdn  <- as.numeric(test_scores_components[(test_scores_components$test_score == min(test_scores_components[test_scores_components$model=="MDN",]$test_score) 
                                                           & test_scores_components$model=="MDN"),
                                                          "components"])

fwrite(test_scores_net,paste0(out_dir,"/q_test_scores_net.csv"))
fwrite(test_scores_sigma,paste0(out_dir,"/q_test_scores_sigma.csv"))
fwrite(test_scores_dropout,paste0(out_dir,"/q_test_scores_dropout.csv"))
fwrite(test_scores_hidden,paste0(out_dir,"/q_test_scores_hidden.csv"))
fwrite(test_scores_neurons,paste0(out_dir,"/q_test_scores_neurons.csv"))
fwrite(test_scores_components,paste0(out_dir,"/q_test_scores_components.csv"))
fwrite(test_scores_all,paste0(out_dir,"/q_test_scores_all.csv"))
