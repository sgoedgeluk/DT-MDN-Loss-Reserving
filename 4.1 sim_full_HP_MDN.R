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
# Hyper-parameter selection for the MDN-based models for the simulated environments

# Import necessary packages, models and support functions
source("models/model_dt_mdn.R")
source("models/model_mdn_org.R")

source("0.1 support_funcs.R")

## Input 
num_triangles <- 50
set_epochs <- 100
sim_env <- 1

mse_weight <- 4
nll_weight <- 1

initial_sigmal2 <- 0
initial_dropout <- 0
initial_hidden <- 2
initial_components <- 1
initial_neurons <- 16

netl2_range <- c(0,0.0001,0.001,0.01)
sigmal2_range <- c(0,0.0001,0.001,0.01,0.1)
dropout_range <- c(0,0.1,0.2)
num_hidden_range <- c(1,2,3,4)
neurons_range <- c(16,24,32,48,64)
components_range <- c(1,2,3)

#### EXPERIMENT: Years ####
num_periods <- 10
for(sim_env in 1:4) {
  print(paste0("Current env: ",sim_env))
  ## Output dir
  output_dir <- paste0("output_sim_full/env_",sim_env)
  data_used <- paste0("sim_env",sim_env)
  
  
  #### Data general
  raw_data_dir <- paste0("sim/env ",sim_env)
  triangles <- load_sim_triangles(raw_data_dir,num_triangles,num_periods)
  list_data_sim <- construct_list_data_multi(triangles)
  
  
  #### Data for DT-MDN 
  norm_data_dt <- normalize_sd(list_data_sim)
  mean_dt <- norm_data_dt$mean
  sd_dt <- norm_data_dt$sd
  norm_data_dt <- norm_data_dt$normalized_data
  
  keras_data_dt <- construct_keras_data(norm_data_dt)
  keras_data_dt <- split_keras_data(keras_data_dt)
  train_data_dt <- keras_data_dt$train
  val_data_dt <- keras_data_dt$validation
  test_data_dt <- keras_data_dt$test
  all_data_dt <- keras_data_dt$all
  
  x_dt <- list(ay_seq_input=train_data_dt$x$ay_seq_input, company_input=train_data_dt$x$company_input)
  val_dt <- list(val_data_dt$x,val_data_dt$y)
  
  ## Data for MDN
  norm_data_mdn <- normalize_sd_all(list_data_sim)
  mean <- norm_data_mdn$mean
  sd <- norm_data_mdn$sd
  norm_data_mdn <- norm_data_mdn$normalized_data
  
  keras_data_mdn <- construct_keras_data(norm_data_mdn)
  keras_data_mdn <- split_keras_data(keras_data_mdn)
  train_data_mdn <- keras_data_mdn$train
  val_data_mdn <- keras_data_mdn$validation
  test_data_mdn <- keras_data_mdn$test
  all_data_mdn <- keras_data_mdn$all
  
  x_mdn <- as.array(cbind(norm_data_mdn$AP[norm_data_mdn$bucket %in% c("train")],
                          norm_data_mdn$DP[norm_data_mdn$bucket %in% c("train")]))
  val_mdn <- list(list(ay_seq_input=as.array(cbind(norm_data_mdn$AP[norm_data_mdn$bucket %in% c("validation")],
                                                   norm_data_mdn$DP[norm_data_mdn$bucket %in% c("validation")]))),
                  val_data_mdn$y)
  test_x_mdn <- as.array(cbind(norm_data_mdn$AP[norm_data_mdn$bucket %in% c("test")],
                               norm_data_mdn$DP[norm_data_mdn$bucket %in% c("test")]))
  all_data_x_mdn <- as.array(cbind(norm_data_mdn$AP,
                                   norm_data_mdn$DP))
  
  
  ## Additional reqs
  early_stop = callback_early_stopping(
    monitor = "val_loss",
    min_delta = 0,
    patience = 1000,
    verbose = 1,
    mode = c("auto", "min", "max"),
    baseline = NULL,
    restore_best_weights = TRUE
  )
  
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
    for(company in unique(norm_data_dt$index_company)) {
      tmp_x_mdn <- x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company,]

      tmp_train_data_mdn_y <- train_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company]
      
      tmp_val_mdn <- list(val_mdn[[1]]$ay_seq_input[norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company,],
                          val_mdn[[2]][norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company])
      
      tmp_test_x_mdn <- test_x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company,]
      tmp_test_data_mdn_y <- test_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company]
      
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
          x = tmp_x_mdn,
          y = tmp_train_data_mdn_y,
          epochs = set_epochs,
          batch_size = dim(tmp_x_mdn)[1],
          validation_data = tmp_val_mdn,
          callbacks = list(early_stop, callback_terminate_on_naan()),
          verbose = 1
        )
      
      test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_data_mdn_y, verbose = 0)
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
    print(paste0("Current model: DT-MDN"))
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
        x = x_dt,
        y = train_data_dt$y,
        epochs = set_epochs,
        batch_size = dim(x_dt[[1]])[1],
        validation_data = val_dt,
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- implement_dt_mdn %>% evaluate(test_data_dt$x, test_data_dt$y, verbose = 0)
    
    test_scores_sigma <- rbind(test_scores_sigma,c("DT_MDN",NA, sigmal2,initial_dropout,initial_hidden,initial_neurons,initial_components,test_error))
    test_scores_all <- rbind(test_scores_all,c("DT_MDN",NA, sigmal2,initial_dropout,initial_hidden,initial_neurons,initial_components,test_error))
    
    end_time <- Sys.time()
    print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
    
    
    ## MDN Model
    print(paste0("Current model: MDN"))
    start_time <- Sys.time()
    test_error <- 0
    for(company in unique(norm_data_dt$index_company)) {
      tmp_x_mdn <- x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company,]
      
      tmp_train_data_mdn_y <- train_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company]
      
      tmp_val_mdn <- list(val_mdn[[1]]$ay_seq_input[norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company,],
                          val_mdn[[2]][norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company])
      
      tmp_test_x_mdn <- test_x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company,]
      tmp_test_data_mdn_y <- test_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company]
      
      implement_mdn <- model_mdn_org(best_netl2_mdn, sigmal2, initial_dropout, initial_neurons, initial_hidden, initial_components)
      implement_mdn %>%
        compile(
          optimizer = 'adam',
          loss = NLLcustom
        )
      
      fit_mdn <- implement_mdn %>% 
        fit(
          x = tmp_x_mdn,
          y = tmp_train_data_mdn_y,
          epochs = set_epochs,
          batch_size = dim(tmp_x_mdn)[1],
          validation_data = tmp_val_mdn,
          callbacks = list(early_stop, callback_terminate_on_naan()),
          verbose = 1
        )
      
      test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_data_mdn_y, verbose = 0)
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
        x = x_dt,
        y = train_data_dt$y,
        epochs = set_epochs,
        batch_size = dim(x_dt[[1]])[1],
        validation_data = val_dt,
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- implement_dt_mdn %>% evaluate(test_data_dt$x, test_data_dt$y, verbose = 0)
    
    test_scores_dropout <- rbind(test_scores_dropout,c("DT_MDN",NA, best_sigma_dt_mdn,dropout,initial_hidden,initial_neurons,initial_components,test_error))
    test_scores_all <- rbind(test_scores_all,c("DT_MDN",NA, best_sigma_dt_mdn,dropout,initial_hidden,initial_neurons,initial_components,test_error))
    
    end_time <- Sys.time()
    print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
    
    
    
    ## MDN Model
    print(paste0("Current model: MDN"))
    start_time <- Sys.time()
    test_error <- 0
    for(company in unique(norm_data_dt$index_company)) {
      tmp_x_mdn <- x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company,]
      
      tmp_train_data_mdn_y <- train_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company]
      
      tmp_val_mdn <- list(val_mdn[[1]]$ay_seq_input[norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company,],
                          val_mdn[[2]][norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company])
      
      tmp_test_x_mdn <- test_x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company,]
      tmp_test_data_mdn_y <- test_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company]
      
      implement_mdn <- model_mdn_org(best_netl2_mdn, best_sigma_mdn, dropout, initial_neurons, initial_hidden, initial_components)
      implement_mdn %>%
        compile(
          optimizer = 'adam',
          loss = NLLcustom
        )
      
      fit_mdn <- implement_mdn %>% 
        fit(
          x = tmp_x_mdn,
          y = tmp_train_data_mdn_y,
          epochs = set_epochs,
          batch_size = dim(tmp_x_mdn)[1],
          validation_data = tmp_val_mdn,
          callbacks = list(early_stop, callback_terminate_on_naan()),
          verbose = 1
        )
    
      test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_data_mdn_y, verbose = 0)
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
    for(company in unique(norm_data_dt$index_company)) {
      tmp_x_mdn <- x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company,]
      
      tmp_train_data_mdn_y <- train_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company]
      
      tmp_val_mdn <- list(val_mdn[[1]]$ay_seq_input[norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company,],
                          val_mdn[[2]][norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company])
      
      tmp_test_x_mdn <- test_x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company,]
      tmp_test_data_mdn_y <- test_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company]
      
      implement_mdn <- model_mdn_org(best_netl2_mdn, best_sigma_mdn, best_dropout_mdn, initial_neurons, num_hidden, initial_components)
      implement_mdn %>%
        compile(
          optimizer = 'adam',
          loss = NLLcustom
        )
      
      fit_mdn <- implement_mdn %>% 
        fit(
          x = tmp_x_mdn,
          y = tmp_train_data_mdn_y,
          epochs = set_epochs,
          batch_size = dim(tmp_x_mdn)[1],
          validation_data = tmp_val_mdn,
          callbacks = list(early_stop, callback_terminate_on_naan()),
          verbose = 1
        )
      
      test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_data_mdn_y, verbose = 0)
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
        x = x_dt,
        y = train_data_dt$y,
        epochs = set_epochs,
        batch_size = dim(x_dt[[1]])[1],
        validation_data = val_dt,
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- implement_dt_mdn %>% evaluate(test_data_dt$x, test_data_dt$y, verbose = 0)
    
    test_scores_neurons <- rbind(test_scores_neurons,c("DT_MDN",NA, best_sigma_dt_mdn,best_dropout_dt_mdn,NA,num_neurons,initial_components,test_error))
    test_scores_all <- rbind(test_scores_all,c("DT_MDN",NA, best_sigma_dt_mdn,best_dropout_dt_mdn,NA,num_neurons,initial_components,test_error))
    
    end_time <- Sys.time()
    print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
    
    
    ## MDN Model
    print(paste0("Current model: MDN"))
    start_time <- Sys.time()
    for(company in unique(norm_data_dt$index_company)) {
      tmp_x_mdn <- x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company,]
      
      tmp_train_data_mdn_y <- train_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company]
      
      tmp_val_mdn <- list(val_mdn[[1]]$ay_seq_input[norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company,],
                          val_mdn[[2]][norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company])
      
      tmp_test_x_mdn <- test_x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company,]
      tmp_test_data_mdn_y <- test_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company]
      
      implement_mdn <- model_mdn_org(best_netl2_mdn, best_sigma_mdn, best_dropout_mdn, num_neurons, best_hidden_mdn, initial_components)
      implement_mdn %>%
        compile(
          optimizer = 'adam',
          loss = NLLcustom
        )
      
      fit_mdn <- implement_mdn %>% 
        fit(
          x = tmp_x_mdn,
          y = tmp_train_data_mdn_y,
          epochs = set_epochs,
          batch_size = dim(tmp_x_mdn)[1],
          validation_data = tmp_val_mdn,
          callbacks = list(early_stop, callback_terminate_on_naan()),
          verbose = 1
        )
      
      test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_data_mdn_y, verbose = 0)
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

  
  #### Neurons ####
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
        x = x_dt,
        y = train_data_dt$y,
        epochs = set_epochs,
        batch_size = dim(x_dt[[1]])[1],
        validation_data = val_dt,
        callbacks = list(early_stop, callback_terminate_on_naan()),
        verbose = 1
      )
    
    test_error <- implement_dt_mdn %>% evaluate(test_data_dt$x, test_data_dt$y, verbose = 0)
    
    test_scores_components <- rbind(test_scores_components,c("DT_MDN",NA, best_sigma_dt_mdn,best_dropout_dt_mdn,NA,best_neurons_dt_mdn,num_components,test_error))
    test_scores_all <- rbind(test_scores_all,c("DT_MDN",NA, best_sigma_dt_mdn,best_dropout_dt_mdn,NA,best_neurons_dt_mdn,num_components,test_error))
    
    end_time <- Sys.time()
    print(paste0("Time taken: ", end_time - start_time," minutes; Test error: ", round(test_error, 3)))
    
    
    
    
    
    
    ## MDN Model
    print(paste0("Current model: MDN"))
    start_time <- Sys.time()
    for(company in unique(norm_data_dt$index_company)) {
      tmp_x_mdn <- x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company,]
      
      tmp_train_data_mdn_y <- train_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("train"),]$index_company==company]
      
      tmp_val_mdn <- list(val_mdn[[1]]$ay_seq_input[norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company,],
                          val_mdn[[2]][norm_data_mdn[norm_data_mdn$bucket %in% c("validation"),]$index_company==company])
      
      tmp_test_x_mdn <- test_x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company,]
      tmp_test_data_mdn_y <- test_data_mdn$y[norm_data_mdn[norm_data_mdn$bucket %in% c("test"),]$index_company==company]
      
      implement_mdn <- model_mdn_org(best_netl2_mdn, best_sigma_mdn, best_dropout_mdn, best_neurons_mdn, best_hidden_mdn, num_components)
      implement_mdn %>%
        compile(
          optimizer = 'adam',
          loss = NLLcustom
        )
      
      fit_mdn <- implement_mdn %>% 
        fit(
          x = tmp_x_mdn,
          y = tmp_train_data_mdn_y,
          epochs = set_epochs,
          batch_size = dim(tmp_x_mdn)[1],
          validation_data = val_mdn,
          callbacks = list(early_stop, callback_terminate_on_naan()),
          verbose = 1
        )
      
      test_error <- test_error + implement_mdn %>% evaluate(tmp_test_x_mdn, tmp_test_data_mdn_y, verbose = 0)
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
  
  
  fwrite(test_scores_net,paste0(output_dir,"/q_test_scores_net.csv"))
  fwrite(test_scores_sigma,paste0(output_dir,"/q_test_scores_sigma.csv"))
  fwrite(test_scores_dropout,paste0(output_dir,"/q_test_scores_dropout.csv"))
  fwrite(test_scores_hidden,paste0(output_dir,"/q_test_scores_hidden.csv"))
  fwrite(test_scores_neurons,paste0(output_dir,"/q_test_scores_neurons.csv"))
  fwrite(test_scores_components,paste0(output_dir,"/q_test_scores_components.csv"))
  fwrite(test_scores_all,paste0(output_dir,"/q_test_scores_all.csv"))
}

