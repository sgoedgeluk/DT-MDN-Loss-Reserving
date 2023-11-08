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
# Final training of the MDN-based models for the real data scenarios

# Import necessary packages, models and support functions
source("models/model_mdn_org.R")
source("models/model_dt_mdn.R")

source("0.1 support_funcs.R")
source("2.1 rd_keras.R")

## Output directory ##
data_used <- "rd_ol"

## Loss weights
mse_weight <- 4
nll_weight <- 1

## Maximum number of epochs ##
set_epochs <- 500

## Best params
params_list_MDN <- fread(paste0("output_rd/",data_used,"/y_test_scores_components.csv"))

best_params_dt_mdn  <- params_list_MDN[(params_list_MDN$test_score == min(params_list_MDN[params_list_MDN$model=="DT_MDN",]$test_score) 
                                        & params_list_MDN$model=="DT_MDN"),]
best_params_mdn  <- params_list_MDN[(params_list_MDN$test_score == min(params_list_MDN[params_list_MDN$model=="MDN",]$test_score) 
                                     & params_list_MDN$model=="MDN"),]

best_params_dt_mdn <- as.data.frame(best_params_dt_mdn)
best_params_mdn <- as.data.frame(best_params_mdn)

## Data
sel_lob <- "other_liability"
data_lob <- data_with_features_sd[data_with_features_sd$lob==sel_lob,]
keras_data <- get_keras_data_lob(data_with_features_sd, sel_lob, company_index_recipe_sd, keep_case_reserves = 0, single_output = 1)
training_data <- keras_data$training_data
val_data <- keras_data$val_data
all_data <- keras_data$all_data

x_dt <- list(ay_seq_input=abind(training_data$x$ay_seq_input, val_data$x$ay_seq_input, along=1),
             company_input=abind(training_data$x$company_input, val_data$x$company_input, along=1))
y_dt <- abind(training_data$y$paid_output,val_data$y$paid_output,along=1)

indices_all <- cbind(data_with_features_sd[(data_with_features_sd$lob==sel_lob),]$accident_year-1987,
                     data_with_features_sd[(data_with_features_sd$lob==sel_lob),]$development_lag,
                     data_with_features_sd[(data_with_features_sd$lob==sel_lob),]$bucket)

indices_all <- as.data.frame(indices_all)
indices_all[,1:2] <- lapply(indices_all[,1:2],as.numeric)
colnames(indices_all) <- c("AP","DP","bucket")

indices_mean <- colMeans(indices_all[,1:2])
indices_sd <- unlist(lapply(indices_all[,1:2],sd))

indices_all[,1:2] <- (indices_all[,1:2]-indices_mean)/indices_sd

x_mdn <- as.matrix(indices_all[indices_all$bucket %in% c("train","validation"),1:2])
y_mdn <- abind(training_data$y$paid_output,val_data$y$paid_output,along=1)

all_x_mdn <- as.matrix(as.data.frame(indices_all[,1:2]))
all_y_mdn <- data_lob$incremental_paid_actual

preds_dt_mdn <- matrix(data=0,nrow=length(all_data$y$paid_output),ncol=3*best_params_dt_mdn$components)
preds_mdn <- matrix(data=0,nrow=length(all_y_mdn),ncol=3*best_params_mdn$components)

# Final evaluation of the models
for(i in 1:5) {
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
  
  tmp <- preds_dt_mdn
  preds_dt_mdn <- preds_dt_mdn + (implement_dt_mdn %>% predict(all_data$x, verbose = 0))
  
  
  ## MDN Model
  for(company in unique(data_lob$group_code)) {
    print(paste0("Index: ",i," company: ",company))
    tmp_x_mdn <- x_mdn[data_lob[data_lob$bucket %in% c("train","validation"),]$group_code==company,]

    tmp_y_mdn <- y_mdn[data_lob[data_lob$bucket %in% c("train","validation"),]$group_code==company]

    tmp_all_x_mdn <- all_x_mdn[data_lob$group_code==company,]

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
        y = tmp_y_mdn,
        epochs = set_epochs,
        batch_size = dim(tmp_x_mdn)[1],
        callbacks = list(callback_terminate_on_naan()),
        verbose = 1
      )

    preds_mdn[data_lob$group_code==company,] <- preds_mdn[data_lob$group_code==company,] +
      (implement_mdn %>% predict(tmp_all_x_mdn, verbose = 0))
  }
  
  
}

preds_dt_mdn <- cbind(data_lob$group_code,data_lob$accident_year-1987,data_lob$development_lag,inv_normalize_sd(preds_dt_mdn * (1/5),data_lob$inc_mean,data_lob$inc_sd),(data_lob$incremental_paid_actual*data_lob$inc_sd+data_lob$inc_mean))
preds_mdn <- cbind(data_lob$group_code,data_lob$accident_year-1987,data_lob$development_lag,inv_normalize_sd(preds_mdn * (1/5),data_lob$inc_mean,data_lob$inc_sd),(data_lob$incremental_paid_actual*data_lob$inc_sd+data_lob$inc_mean))

colnames(preds_dt_mdn) <- c("index_company","AP","DP",paste0("alpha_",1:best_params_dt_mdn$components),
                            paste0("mu_",1:best_params_dt_mdn$components),paste0("sigma_",1:best_params_dt_mdn$components),
                            "mdn_mean","mdn_sigma","loss")
colnames(preds_mdn) <-c("index_company","AP","DP",paste0("alpha_",1:best_params_mdn$components),
                        paste0("mu_",1:best_params_mdn$components),paste0("sigma_",1:best_params_mdn$components),
                        "mdn_mean","mdn_sigma","loss")

fwrite(preds_dt_mdn,paste0("output_rd/",data_used,"/",data_used,"_DT-MDN.csv"))
fwrite(preds_mdn,paste0("output_rd/",data_used,"/",data_used,"_MDN.csv"))
