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
# Final training of the MDN-based models for the simulated data environments

# Import necessary packages, models and support functions
source("models/model_dt_mdn.R")
source("models/model_mdn_org.R")

source("0.1 support_funcs.R")

sim_env <- 4
data_dir <- paste0("output_sim_full/env_",sim_env)

#### Best params ####
params_list_MDN <- fread(paste0("output_sim_full/env_",sim_env,"/y_test_scores_components.csv"))

best_params_dt_mdn  <- params_list_MDN[(params_list_MDN$test_score == min(params_list_MDN[params_list_MDN$model=="DT_MDN",]$test_score) 
                                                              & params_list_MDN$model=="DT_MDN"),]
best_params_mdn  <- params_list_MDN[(params_list_MDN$test_score == min(params_list_MDN[params_list_MDN$model=="MDN",]$test_score) 
                                                           & params_list_MDN$model=="MDN"),]

best_params_dt_mdn <- as.data.frame(best_params_dt_mdn)
best_params_mdn <- as.data.frame(best_params_mdn)


#### Data General ####
num_triangles <- 50
set_epochs <- 500

mse_weight <- 4
nll_weight <- 1
num_periods <- 10

output_dir <- paste0("output_sim_full/env_",sim_env)
raw_data_dir <- paste0("sim/env ",sim_env)
triangles <- load_sim_triangles(raw_data_dir,num_triangles,num_periods)
list_data_sim <- construct_list_data_multi(triangles)

#### Data for DT-MDN ####
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


x_dt <- list(ay_seq_input=abind(train_data_dt$x$ay_seq_input, val_data_dt$x$ay_seq_input, along=1),
            company_input=abind(train_data_dt$x$company_input, val_data_dt$x$company_input, along=1))
y_dt <- abind(train_data_dt$y,val_data_dt$y,along=1)


#### Data for MDN ####
norm_data_mdn <- normalize_sd_all(list_data_sim)
mean_mdn <- norm_data_mdn$mean
sd_mdn <- norm_data_mdn$sd
norm_data_mdn <- norm_data_mdn$normalized_data

keras_data_mdn <- construct_keras_data(norm_data_mdn)
keras_data_mdn <- split_keras_data(keras_data_mdn)
train_data_mdn <- keras_data_mdn$train
val_data_mdn <- keras_data_mdn$validation
test_data_mdn <- keras_data_mdn$test
all_data_mdn <- keras_data_mdn$all

x_mdn <- as.array(cbind(norm_data_mdn$AP[norm_data_mdn$bucket %in% c("train","validation")],
                        norm_data_mdn$DP[norm_data_mdn$bucket %in% c("train","validation")]))
y_mdn <- abind(train_data_mdn$y,val_data_mdn$y,along=1)

all_data_x_mdn <- as.array(cbind(norm_data_mdn$AP,
                                 norm_data_mdn$DP))


preds_dt_mdn <- matrix(data=0,nrow=length(all_data_dt$y),ncol=3*best_params_dt_mdn$components)
preds_mdn <- matrix(data=0,nrow=length(all_data_mdn$y),ncol=3*best_params_mdn$components)

for(i in 1:5) {
  set_random_seed(i)
  
  ## DT-MDN Model
  implement_dt_mdn <- model_dt_mdn(as.numeric(best_params_dt_mdn$sigma), as.numeric(best_params_dt_mdn$dropout),
                                   as.numeric(best_params_dt_mdn$neurons), as.numeric(best_params_dt_mdn$components))
  implement_dt_mdn %>%
    compile(
      optimizer = 'adam',
      loss = NLLcustom
    )

  fit_dt_mdn <- implement_dt_mdn %>%
    fit(
      x = x_dt,
      y = y_dt,
      epochs = set_epochs,
      batch_size = dim(x_dt[[1]])[1],
      callbacks = list(callback_terminate_on_naan()),
      verbose = 1
    )

  preds_dt_mdn <- preds_dt_mdn + (implement_dt_mdn %>% predict(all_data_dt$x, verbose = 0))
  
 
  ## MDN Model
  for(company in unique(norm_data_dt$index_company)) {
    print(paste0("Index: ",i," company: ",company))
    tmp_x_mdn <- x_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("train","validation"),]$index_company==company,]
    
    tmp_train_data_mdn_y <- y_mdn[norm_data_mdn[norm_data_mdn$bucket %in% c("train","validation"),]$index_company==company]
    
    tmp_all_data_x_mdn <- all_data_x_mdn[norm_data_mdn$index_company==company,]

    print(paste0("Current model: MDN"))
    start_time <- Sys.time()
    set_random_seed(0)
    
    implement_mdn <- model_mdn_org(as.numeric(best_params_mdn$netl2), as.numeric(best_params_mdn$sigma), 
                                   as.numeric(best_params_mdn$dropout), as.numeric(best_params_mdn$neurons),
                                   as.numeric(best_params_mdn$num_hidden), as.numeric(best_params_mdn$components))
    
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
        callbacks = list(callback_terminate_on_naan()),
        verbose = 1
      )
    
    preds_mdn[norm_data_mdn$index_company==company,] <- preds_mdn[norm_data_mdn$index_company==company,] + 
      (implement_mdn %>% predict(tmp_all_data_x_mdn, verbose = 0))
  }
  
  
}

preds_dt_mdn <- cbind(list_data_sim$index_company,list_data_sim$AP,list_data_sim$DP,inv_normalize_sd(preds_dt_mdn*1/5,mean_dt,sd_dt),list_data_sim$value)
preds_mdn <- cbind(list_data_sim$index_company,list_data_sim$AP,list_data_sim$DP,inv_normalize_sd(preds_mdn*1/5,mean_mdn[3],sd_mdn[3]),list_data_sim$value)

colnames(preds_dt_mdn) <- c("index_company","AP","DP",paste0("alpha_",1:best_params_dt_mdn$components),
                            paste0("mu_",1:best_params_dt_mdn$components),paste0("sigma_",1:best_params_dt_mdn$components),
                            "mdn_mean","mdn_sigma","loss")
colnames(preds_mdn) <-c("index_company","AP","DP",paste0("alpha_",1:best_params_mdn$components),
                          paste0("mu_",1:best_params_mdn$components),paste0("sigma_",1:best_params_mdn$components),
                          "mdn_mean","mdn_sigma","loss")

fwrite(preds_dt_mdn,paste0(output_dir,"/","sim_env4_DT-MDN.csv"))
fwrite(preds_mdn,paste0(output_dir,"/","sim_env4_MDN.csv"))

