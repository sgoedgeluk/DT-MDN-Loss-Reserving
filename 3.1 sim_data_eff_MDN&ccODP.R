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
# Training and forecasting of the MDN-based models for the experiment focusing on 
# the effect of the amount of data on forecasting performance

# Import necessary packages, models and support functions
source("models/model_dt_mdn.R")
source("models/model_mdn_org.R")

source("0.1 support_funcs.R")

## Input
output_dir <- "output_sim_data"
data_used <- "sim_env1"
tot_triangles <- 200
num_triangles <- 1#c(5,10,25,50,100,160,200)
set_epochs <- 500

mse_weight <- 4
nll_weight <- 1
netl2 <- 0.01
sigmal2 <- 0.01
dropout <- 0.1
neurons <- 32
no_hidden <- 3
components <- 3

## Data
sel_env <- 1
raw_data_dir_y <- paste0("sim/env ",sel_env)
raw_data_dir_q <- paste0("sim/env ",sel_env," q")
triangles_y <- load_sim_triangles(raw_data_dir_y,200,10)
triangles_q <- load_sim_triangles(raw_data_dir_q,200,40)

list_data_sim_y <- construct_list_data_multi(triangles_y)
list_data_sim_q <- construct_list_data_multi(triangles_q)

early_stop = callback_early_stopping(
  monitor = "val_loss",
  min_delta = 0,
  patience = 1000,
  verbose = 1,
  mode = c("auto", "min", "max"),
  baseline = NULL,
  restore_best_weights = TRUE
)

#### Experiment: Quarters ####
num_periods <- 40
test_errors_list_q <- list()
preds_list_q <- list()

for(curr_num in num_triangles) {
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
  
  print(paste0("Current number of triangles: ",curr_num,"; Current model: DT-MDN."))
  start_time <- Sys.time()
  set_random_seed(curr_num)
  
  if(curr_num > 50) {
    company_num <- curr_num
  } else {
    company_num <- 50
  }
  
  ## DT-MDN Model
  curr_model <- model_dt_mdn(sigmal2, dropout, neurons, components, company_num)
  curr_model %>%
    compile(
      optimizer = 'adam',
      loss = NLLcustom
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
  test_errors_list_q <- append(test_errors_list_q,list(list(num_triangles=curr_num,model="DT-MDN",error=test_error)))
  
  preds <- curr_model %>% predict(all_data$x)
  
  preds_conv <- inv_normalize_sd(preds,mean,sd)
  
  preds_conv <- cbind(sel_data$index_company,
                      sel_data$AP,
                      sel_data$DP,
                      preds_conv,
                      sel_data$value)
  colnames(preds_conv) <- c("company_input","AP","DP",
                            paste0("alpha_",1:components),paste0("mu_",1:components),paste0("sigma_",1:components),
                            "mdn_mean","mdn_sigma","loss")
  
  preds_list_q <- append(preds_list_q,list(list(num_triangles=curr_num,model="DT-MDN",preds=preds_conv)))
  
  fwrite(as.data.table(preds_conv), paste0(output_dir,"/",data_used,"_",curr_num,"_q_DT-MDN.csv"))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes."))
}


#ccODP model
data_odp <- list_data_sim_q %>% select(index_company,AP,DP,value)

data_odp <- cbind(data_odp,rep(0,nrow(data_odp)),rep(0,nrow(data_odp)))
colnames(data_odp) <- c("company_input","AP","DP","loss","odp_mean","odp_dispersion")
data_odp$loss[data_odp$loss<0] <- 0

train_data <- data_odp[(data_odp$AP+data_odp$DP)<=(num_periods+1),]
test_data <- data_odp[data_odp$AP+data_odp$DP>(num_periods+1),]
glm_models <- list()
for (company in unique(data_odp$company_input)) {
  tmp_train <- train_data[train_data$company_input==company,]
  glm_model <- glm(loss ~ as.factor(AP) + as.factor(DP) - 1, data = tmp_train, family = quasipoisson(link = "log"))
  
  #Store alpha, beta and dispersion parameters
  dispersion <- summary(glm_model)$dispersion
  
  log_a <- as.numeric(glm_model$coefficients[1:num_periods])
  log_b <- as.numeric(c(0, glm_model$coefficients[(num_periods+1):(num_periods*2-1)]))

  a_vec <- exp(log_a)
  b_vec <- exp(log_b)
  
  a_vec <- a_vec * sum(b_vec)
  b_vec <- b_vec/sum(b_vec)
  
  data_odp$odp_mean[data_odp$company_input==company] <- a_vec[data_odp$AP[data_odp$company_input==company]]*
    b_vec[data_odp$DP[data_odp$company_input==company]]
  data_odp$odp_dispersion[data_odp$company_input==company] <- dispersion
  
  tmp_list <- list(list(company=company,model=glm_model,dispersion=dispersion,a_vec=a_vec,b_vec=b_vec))
  glm_models <- append(glm_models,tmp_list)
}

fwrite(data_odp, paste0(output_dir,"/",data_used, "_odp_q.csv"))


## MDN model
norm_data_mdn <- normalize_sd_all(list_data_sim_q)
mean_mdn <- norm_data_mdn$mean
sd_mdn <- norm_data_mdn$sd
norm_data_mdn <- norm_data_mdn$normalized_data

preds_mdn <- matrix(data=0,nrow=length(norm_data_mdn$index_company),ncol=3*components)

for(company in 1:200) {
  print(paste0("Index: ",i," company: ",company))
  
  norm_data_tmp <- norm_data_mdn[norm_data_mdn$index_company==company,]
  
  x_mdn <- as.array(cbind(norm_data_tmp$AP[norm_data_tmp$bucket %in% c("train","validation")],
                          norm_data_tmp$DP[norm_data_tmp$bucket %in% c("train","validation")]))
  y_mdn <- norm_data_tmp$value[norm_data_tmp$bucket%in% c("train","validation")]
  
  all_data_x_mdn <- as.array(cbind(norm_data_tmp$AP,
                                   norm_data_tmp$DP))
  
  print(paste0("Current model: MDN"))
  start_time <- Sys.time()
  set_random_seed(0)
  
  implement_mdn <- model_mdn_org(netl2, sigmal2, dropout, neurons, no_hidden, components)
  
  implement_mdn %>%
    compile(
      optimizer = 'adam',
      loss = NLLcustom
    )
  
  fit_mdn <- implement_mdn %>% 
    fit(
      x = x_mdn,
      y = y_mdn,
      epochs = set_epochs,
      batch_size = dim(x_mdn)[1],
      callbacks = list(callback_terminate_on_naan()),
      verbose = 1
    )
  
  preds_mdn[norm_data_mdn$index_company==company,] <- preds_mdn[norm_data_mdn$index_company==company,] + 
    (implement_mdn %>% predict(all_data_x_mdn, verbose = 0))
}

num_triangles <- c(5,10,25,50,100,160,200)
for(n in num_triangles) {
  fwrite(preds_mdn[preds_mdn$index_company<=n,],paste0("output_sim_data/","sim_env1_",n,"_q_MDN.csv"))
}



#### Experiment: Years ####
num_periods <- 10
test_errors_list_y <- list()
preds_list_y <- list()

for(curr_num in num_triangles) {
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
  set_random_seed(curr_num)
  
  if(curr_num > 50) {
    company_num <- curr_num
  } else {
    company_num <- 50
  }
  
  ## DT-MDN Model
  curr_model <- model_dt_mdn(sigmal2, dropout, neurons, components, company_num)
  curr_model %>%
    compile(
      optimizer = 'adam',
      loss = NLLcustom
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
  test_errors_list_y <- append(test_errors_list_y,list(list(num_triangles=curr_num,model="DT-MDN",error=test_error)))
  
  preds <- curr_model %>% predict(all_data$x)
  
  preds_conv <- inv_normalize_sd(preds,mean,sd)
  
  preds_conv <- cbind(sel_data$index_company,
                      sel_data$AP,
                      sel_data$DP,
                      preds_conv,
                      sel_data$value)
  colnames(preds_conv) <- c("company_input","AP","DP",
                            paste0("alpha_",1:components),paste0("mu_",1:components),paste0("sigma_",1:components),
                            "mdn_mean","mdn_sigma","loss")
  
  preds_list_y <- append(preds_list_y,list(list(num_triangles=curr_num,model="DT-MDN",preds=preds_conv)))
  
  fwrite(as.data.table(preds_conv), paste0(output_dir,"/",data_used,"_",curr_num,"_y_DT-MDN.csv"))
  
  end_time <- Sys.time()
  print(paste0("Time taken: ", end_time - start_time," minutes."))
}

#ccODP model
data_odp <- list_data_sim_y %>% select(index_company,AP,DP,value)

data_odp <- cbind(data_odp,rep(0,nrow(data_odp)),rep(0,nrow(data_odp)))
colnames(data_odp) <- c("company_input","AP","DP","loss","odp_mean","odp_dispersion")
data_odp$loss[data_odp$loss<0] <- 0

train_data <- data_odp[(data_odp$AP+data_odp$DP)<=(num_periods+1),]
test_data <- data_odp[data_odp$AP+data_odp$DP>(num_periods+1),]
glm_models <- list()
for (company in unique(data_odp$company_input)) {
  tmp_train <- train_data[train_data$company_input==company,]
  glm_model <- glm(loss ~ as.factor(AP) + as.factor(DP) - 1, data = tmp_train, family = quasipoisson(link = "log"))
  
  #Store alpha, beta and dispersion parameters
  dispersion <- summary(glm_model)$dispersion
  
  log_a <- as.numeric(glm_model$coefficients[1:num_periods])
  log_b <- as.numeric(c(0, glm_model$coefficients[(num_periods+1):(num_periods*2-1)]))

  a_vec <- exp(log_a)
  b_vec <- exp(log_b)
  
  a_vec <- a_vec * sum(b_vec)
  b_vec <- b_vec/sum(b_vec)
  
  data_odp$odp_mean[data_odp$company_input==company] <- a_vec[data_odp$AP[data_odp$company_input==company]]*
    b_vec[data_odp$DP[data_odp$company_input==company]]
  data_odp$odp_dispersion[data_odp$company_input==company] <- dispersion
  
  tmp_list <- list(list(company=company,model=glm_model,dispersion=dispersion,a_vec=a_vec,b_vec=b_vec))
  glm_models <- append(glm_models,tmp_list)
}

fwrite(data_odp, paste0(output_dir,"/",data_used, "_odp_y.csv"))

## MDN model
norm_data_mdn <- normalize_sd_all(list_data_sim_y)
mean_mdn <- norm_data_mdn$mean
sd_mdn <- norm_data_mdn$sd
norm_data_mdn <- norm_data_mdn$normalized_data

preds_mdn <- matrix(data=0,nrow=length(norm_data_mdn$index_company),ncol=3*components)

for(company in 1:200) {
  print(paste0("Index: ",i," company: ",company))
  
  norm_data_tmp <- norm_data_mdn[norm_data_mdn$index_company==company,]
  
  x_mdn <- as.array(cbind(norm_data_tmp$AP[norm_data_tmp$bucket %in% c("train","validation")],
                          norm_data_tmp$DP[norm_data_tmp$bucket %in% c("train","validation")]))
  y_mdn <- norm_data_tmp$value[norm_data_tmp$bucket%in% c("train","validation")]
  
  all_data_x_mdn <- as.array(cbind(norm_data_tmp$AP,
                                   norm_data_tmp$DP))
  
  print(paste0("Current model: MDN"))
  start_time <- Sys.time()
  set_random_seed(0)
  
  implement_mdn <- model_mdn_org(netl2, sigmal2, dropout, neurons, no_hidden, components)
  
  implement_mdn %>%
    compile(
      optimizer = 'adam',
      loss = NLLcustom
    )
  
  fit_mdn <- implement_mdn %>% 
    fit(
      x = x_mdn,
      y = y_mdn,
      epochs = set_epochs,
      batch_size = dim(x_mdn)[1],
      callbacks = list(callback_terminate_on_naan()),
      verbose = 1
    )
  
  preds_mdn[norm_data_mdn$index_company==company,] <- preds_mdn[norm_data_mdn$index_company==company,] + 
    (implement_mdn %>% predict(all_data_x_mdn, verbose = 0))
}

num_triangles <- c(5,10,25,50,100,160,200)
for(n in num_triangles) {
  fwrite(preds_mdn[preds_mdn$index_company<=n,],paste0("output_sim_data/","sim_env1_",n,"_y_MDN.csv"))
}
