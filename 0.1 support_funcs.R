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
# Defines support functions for the structuring of data, normalization processes, 
# performance metrics and evaluation of the results.

# Given a time series, return a list where each element is a vector representing a window
# of the time series determined by the offsets.
make_series <- function(v, start_offset, end_offset, na_pad = -99) {
  prepad_mask <- function(v, l = 9) {
    length_diff <- l - length(v)
    if (length_diff > 0) {
      c(rep(na_pad, length_diff), v)
    } else {
      v
    }
  }
  
  purrr::map(
    seq_along(v),
    function(x) {
      start <- max(0, x + start_offset)
      end <- max(0, x + end_offset)
      out <- v[start:end]
      ifelse(is.na(out), na_pad, out)
    } %>%
      prepad_mask()
  )
}

# Creates historical input series for use as input in the machine learning models
mutate_series <- function(data, timesteps = 9) {
  data %>%
    dplyr::group_by(.data$lob, .data$group_code, .data$accident_year) %>%
    dplyr::arrange(.data$lob, .data$group_code, .data$accident_year, .data$development_lag) %>%
    mutate(
      paid_lags = make_series(incremental_paid_actual, -timesteps, -1),
      case_lags = make_series(case_reserves, -timesteps, -1),
      paid_target = make_series(incremental_paid_actual, 0, timesteps - 1),
      case_target = make_series(case_reserves, 0, timesteps - 1),
    ) %>%
    ungroup()
}

# Averages multiple triangles into a single triangle cell-by-cell.
average_triangle = function(triangles) {
  res = lm2a(copy(triangles[[2]]), dim.order = c(3, 1, 2))
  res = colMeans(res)
  return(res)
}

# Constructs a list of data points from an individual loss triangle.
construct_list_data <- function(triangle) {
  n_periods <- ncol(triangle)-1
  res <- pivot_longer(triangle,2:(n_periods+1),names_to=c("DP"), names_prefix="[a-zA-Z]")
  res$DP <- as.integer(res$DP)
  CALP <- c()
  for(i in 1:n_periods) {
    CALP <- c(CALP,(i:(i+n_periods-1)))
  }
  res <- cbind(CALP,res)
  return(res)
}

# Constructs a list of data points from multiple loss triangles, concatenating each of 
# the triangles into one list.
construct_list_data_multi <- function(triangles,incl_bucket=TRUE) {
  res <- data.frame(index_company=integer(),CALP=integer(),AP=integer(), DP=integer(),value=double())
  n_periods <- ncol(triangles$triangle[[1]])
  for(i in 1:nrow(triangles)) {
    curr <- data.frame(AP=1:n_periods,triangles$triangle[[i]])
    colnames(curr) <- c("AP",1:n_periods)
    curr_list <- construct_list_data(curr)
    curr_list <- cbind(rep(i,nrow(curr_list)),curr_list)
    res <- rbind(res,curr_list)
  }
  colnames(res) <- c("index_company","CALP","AP","DP","value")
  res <- res %>%
    mutate(
      bucket = case_when(
        CALP <= (4/5*n_periods) & DP > 1 ~ "train",
        CALP > (4/5*n_periods) & CALP <= n_periods &
          DP > 1 ~ "validation",
        CALP > n_periods ~ "test"
      )
    )
  return(res)
}

# Formats the simulated, historical data into a format that can be used by the DeepTriangle-based 
# models. The original model uses nine historical data points as input and nine future 
# data points as output, whereas the DT-MDN model uses one output to predict densities.
construct_keras_data = function(list_data, timesteps=9, include_target=FALSE) {
  if(include_target) {
    res <- copy(list_data) %>%
      dplyr::group_by(.data$index_company, .data$AP) %>%
      dplyr::arrange(.data$index_company, .data$AP, .data$DP) %>%
      mutate(
        paid_lags = make_series(value, -timesteps, -1),
        paid_target = make_series(value, 0, timesteps - 1)
      ) %>%
      ungroup()
  } else {
    res <- copy(list_data) %>%
      dplyr::group_by(.data$index_company, .data$AP) %>%
      dplyr::arrange(.data$index_company, .data$AP, .data$DP) %>%
      mutate(
        paid_lags = make_series(value, -timesteps, -1)
      ) %>%
      ungroup()
  }
  return(data.frame(res))
}

# Loads all the triangles from a folder and stores these in a list.
load_sim_triangles <- function(folder, num_triangles, num_periods=10) {
  triangles <- data.frame(num=integer(), triangle=I(list()))
  
  i = 1
  for(file in list.files(path=folder, pattern=".csv", all.files=TRUE, full.names=TRUE)) {
    print(file)
    triangles[i,"num"] <- i
    triangles[[i,"triangle"]] <- data.matrix(read.csv(file)[,2:(num_periods+1)])
    i=i+1
    if(i > num_triangles) {
      break
    }
  }
  return(triangles)
}

# Normalizes the losses using a z-score approach, please refer to the paper for 
# details on this approach.
normalize_sd <- function(list_data) {
  res <- copy(list_data)
  data_mean <- mean(na.omit(res[res$bucket %in% c("train","validation"),"value"]))
  data_sd <- sd(na.omit(res[res$bucket %in% c("train","validation"),"value"]))
  res$value <- (res$value-data_mean)/data_sd
  
  return(list(mean=data_mean,sd=data_sd,normalized_data=res))
}

# Normalizes the losses, accident period and development period using a z-score 
# approach, please refer to the paper for details on this approach.
normalize_sd_all <- function(list_data) {
  res <- copy(list_data)
  
  AP_mean <- mean(na.omit(res[res$bucket %in% c("train","validation"),"AP"]))
  AP_sd <- sd(na.omit(res[res$bucket %in% c("train","validation"),"AP"]))
  res$AP <- (res$AP-AP_mean)/AP_sd
  
  DP_mean <- mean(na.omit(res[res$bucket %in% c("train","validation"),"DP"]))
  DP_sd <- sd(na.omit(res[res$bucket %in% c("train","validation"),"DP"]))
  res$DP <- (res$DP-DP_mean)/DP_sd
  
  value_mean <- mean(na.omit(res[res$bucket %in% c("train","validation"),"value"]))
  value_sd <- sd(na.omit(res[res$bucket %in% c("train","validation"),"value"]))
  res$value <- (res$value-value_mean)/value_sd
  
  return(list(mean=c(AP_mean,DP_mean,value_mean),sd=c(AP_sd,DP_sd,value_sd),normalized_data=res))
}

# Splits the keras data into training, validation, test and all buckets to facilitate 
# the training of the models. The all bucket contains all of the data points in the 
# training, validation and test bucket.
split_keras_data <- function(keras_data, target_multi=FALSE) {
  in_length <- length(keras_data$paid_lags[[1]])
  
  lags_train <- keras_data %>%
    filter(bucket %in% c("train")) %>%
    select(.data$paid_lags) %>%
    purrr::transpose() %>%
    purrr::map(~ array(unlist(.x), dim = c(1, in_length, 1))) %>%
    abind::abind(along = 1) %>%
    unname()
  
  id_train <- keras_data %>%
    filter(bucket %in% c("train")) %>%
    select(index_company) %>% 
    as.matrix()
  
  if(target_multi) {
    target_train <- keras_data %>%
      filter(bucket %in% c("train")) %>%
      select(.data$paid_target) %>%
      purrr::transpose() %>%
      purrr::map(~ array(unlist(.x), dim = c(1, in_length, 1))) %>%
      abind::abind(along = 1) %>%
      unname()
  } else {
    target_train <- keras_data$value[!is.na(keras_data$bucket) & keras_data$bucket=="train"]
    target_train <- array_reshape(target_train, dim=c(length(target_train),1))
  }
  
  lags_test <- keras_data %>%
    filter(bucket %in% c("test")) %>%
    select(.data$paid_lags) %>%
    purrr::transpose() %>%
    purrr::map(~ array(unlist(.x), dim = c(1, in_length, 1))) %>%
    abind::abind(along = 1) %>%
    unname()
  
  id_test <- keras_data %>%
    filter(bucket %in% c("test")) %>%
    select(index_company) %>% 
    as.matrix()
  
  if(target_multi) {
    target_test <- keras_data %>%
      filter(bucket %in% c("test")) %>%
      select(.data$paid_target) %>%
      purrr::transpose() %>%
      purrr::map(~ array(unlist(.x), dim = c(1, in_length, 1))) %>%
      abind::abind(along = 1) %>%
      unname()
  } else {
    target_test <- keras_data$value[!is.na(keras_data$bucket) & keras_data$bucket=="test"]
    target_test <- array_reshape(target_test, dim=c(length(target_test),1))
  }
  
  lags_validation <- keras_data %>%
    filter(bucket %in% c("validation")) %>%
    select(.data$paid_lags) %>%
    purrr::transpose() %>%
    purrr::map(~ array(unlist(.x), dim = c(1, in_length, 1))) %>%
    abind::abind(along = 1) %>%
    unname()
  
  id_validation <- keras_data %>%
    filter(bucket %in% c("validation")) %>%
    select(index_company) %>% 
    as.matrix()
  
  if(target_multi) {
    target_validation <- keras_data %>%
      filter(bucket %in% c("validation")) %>%
      select(.data$paid_target) %>%
      purrr::transpose() %>%
      purrr::map(~ array(unlist(.x), dim = c(1, in_length, 1))) %>%
      abind::abind(along = 1) %>%
      unname()
  } else {
    target_validation <- keras_data$value[!is.na(keras_data$bucket) & keras_data$bucket=="validation"]
    target_validation <- array_reshape(target_validation, dim=c(length(target_validation),1))
  }
    
  lags_all <- keras_data %>%
    select(.data$paid_lags) %>%
    purrr::transpose() %>%
    purrr::map(~ array(unlist(.x), dim = c(1, in_length, 1))) %>%
    abind::abind(along = 1) %>%
    unname()
  
  id_all <- keras_data %>%
    select(index_company) %>% 
    as.matrix()
  
  if(target_multi) {
    target_all <- keras_data %>%
      select(.data$paid_target) %>%
      purrr::transpose() %>%
      purrr::map(~ array(unlist(.x), dim = c(1, in_length, 1))) %>%
      abind::abind(along = 1) %>%
      unname()
  } else {
    target_all <- array_reshape(keras_data$value, dim=c(length(keras_data$value),1))
  }
    
  train <- list(
                  x=list(
                    ay_seq_input = lags_train,
                    company_input = id_train
                  ),
                  y=target_train
                )
  validation <- list(
                  x=list(
                    ay_seq_input = lags_validation,
                    company_input = id_validation
                  ),
                  y=target_validation
                )
  test <- list(
                  x=list(
                    ay_seq_input = lags_test,
                    company_input = id_test
                  ),
                  y=target_test
                )
  all <- list(
                x=list(
                  ay_seq_input = lags_all,
                  company_input = id_all
                ),
                y=target_all
  )
  
  return(list(train=train,validation=validation,test=test,all=all))
}

# Formats the data of the real data scenarios into a list format that can be used by the machine 
# learning models, includes the ability to create different outputs, including those with
# one output (for the DT2 model) and with or without case reserves. It also includes the 
# company identifier.
prep_keras_data <- function(data, company_index_recipe, keep_case_reserves=1, single_output=1) {
  if(keep_case_reserves) {
    lags <- data %>%
      select(.data$paid_lags, .data$case_lags) %>%
      purrr::transpose() %>%
      purrr::map(~ array(unlist(.x), dim = c(1, 9, 2))) %>%
      abind::abind(along = 1) %>%
      unname()
  } else {
    lags <- data %>%
      select(.data$paid_lags) %>%
      purrr::transpose() %>%
      purrr::map(~ array(unlist(.x), dim = c(1, 9, 1))) %>%
      abind::abind(along = 1) %>%
      unname()
  }
  
  if(single_output) {
    target_paid <- data %>%
      pull(.data$incremental_paid_actual) %>%
      array_reshape(c(nrow(data), 1, 1))
  } else {
    target_paid <- data %>%
      pull(.data$paid_target) %>%
      flatten_dbl() %>%
      array_reshape(c(nrow(data), 9, 1))
  }
  
  if(keep_case_reserves) {
    if(single_output) {
      target_case <- data %>%
        pull(.data$case_reserves_actual) %>%
        array_reshape(c(nrow(data), 1, 1))
    } else {
      target_case <- data %>%
        pull(.data$case_target) %>%
        flatten_dbl() %>%
        array_reshape(c(nrow(data), 9, 1))
    }
  }
  
  company_input <- bake(company_index_recipe, data) %>% as.matrix()
  
  if(keep_case_reserves) {
    return(list(
      x = list(
        ay_seq_input = lags, company_input = company_input
      ),
      y = list(
        paid_output = target_paid,
        case_reserves_output = target_case
      )
    ))
  }
  return(list(
    x = list(
      ay_seq_input = lags, company_input = company_input
    ),
    y = list(
      paid_output = target_paid
    )
  ))
}

#Returns the pdf of a Gaussian distribution at y
GaussianPDF = function(y, alpha, mu, sigma){
  inv_sigma = 1/sigma
  exponent = (-0.5)*k_square((y - mu)/sigma)
  constant = alpha/(k_sqrt(2*pi*k_square(sigma)))
  return (((constant * k_exp(exponent)))   )
}

# Returns the mean of a mixed Gaussian, given the alpha and mu parameters.
# Used when the mse_weight is set
MeanGaussian = function(parameter){
  mean_est = 0
  c = dim(parameter)[[2]]/3
  for (i in 1:c){
    alpha = parameter[,(1*i):(1*i)]
    mu = parameter[,(c+i):(c+i)]
    mean_est = mean_est + alpha*mu
    
  }
  return (mean_est)
}

# Adapted version of a standard mean squared error loss function that ignores 
# output results if it is at a masked portion of the triangle.
masked_mse <- function(mask_value) {
  function(y_true, y_pred) {
    keep_value <- k_cast(k_not_equal(y_true, mask_value), k_floatx())
    sum_squared_error <- k_sum(
      k_square(keep_value * (y_true - y_pred)),
      axis = 2
    )
    sum_squared_error / k_sum(keep_value, axis = 2)
  }
}

# Custom negative log-likelihood function for use in the MDN-based models
NLLcustom = function(y, parameter){
  K = backend()
  NLL_term = 0
  mean_est = 0
  c = dim(parameter)[[2]]/3
  for (i in 1:c){
    alpha = parameter[,(1*i):(1*i)]
    mu = parameter[,(c+i):(c+i)]
    sigma = parameter[,(2*c+i):(2*c+i)]
    NLL_term = NLL_term + GaussianPDF(y, alpha, mu, sigma)
    mean_est = mean_est + alpha*mu
  }
  NLL_term = -k_mean(k_log(NLL_term))
  MSE_term = k_mean(k_square(y - mean_est))
  
  final_output = nll_weight*(NLL_term) + mse_weight*MSE_term
  
  return (final_output)
}


# Filters the data from the real data scenarios for a given line of business and splits 
# this into separate variables for the training, validation and test data. It also creates
# variables containing the data for all these buckets.
get_keras_data_lob <- function(data, sel_lob, company_index_recipe, keep_case_reserves=1, single_output=1) {
  val_data <- data %>%
    filter(bucket %in% c("train", "validation") | development_lag == 1) %>%
    mutate_series() %>%
    filter(bucket %in% "validation",  lob==sel_lob)
  
  training_data <- data %>%
    filter(bucket %in% c("train", "validation") | development_lag == 1) %>%
    mutate_series() %>%
    filter(bucket %in% c("train"), lob==sel_lob)
  
  test_data <- data %>%
    mutate_series() %>%
    filter(bucket %in% "test", lob==sel_lob) # Removed calendar_year == 1998 filter
  
  all_data <- data %>%
    mutate_series() %>%
    filter(lob==sel_lob) # Removed calendar_year == 1998 filter
  
  training_data_keras <- prep_keras_data(training_data, company_index_recipe, keep_case_reserves, single_output)
  val_data_keras <- prep_keras_data(val_data, company_index_recipe, keep_case_reserves, single_output)
  test_data_keras <- prep_keras_data(test_data, company_index_recipe, keep_case_reserves, single_output)
  all_data_keras <- prep_keras_data(all_data, company_index_recipe, keep_case_reserves, single_output)
  
  return(list(training_data=training_data_keras,val_data=val_data_keras,test_data=test_data_keras,all_data=all_data_keras))
}

# Inverses the z-score standardization method to recover the predictions in the correct scale
inv_normalize_sd <- function(preds,mean,sd) {
  res <- copy(preds)
  num_densities <- ncol(res)/3
  sd <- array(unlist(sd))
  mean <- array(unlist(mean))
  
  for(i in 1:num_densities) {
    res[,num_densities+i] <- as.vector(res[,num_densities+i])*sd+mean
    res[,num_densities*2+i] <- as.vector(res[,num_densities*2+i])*sd
  }
  
  #Calculate mean
  mean <- res[,1:num_densities] * res[,(num_densities+1):(num_densities*2)]
  mean <- rowSums(mean)
  
  #Calculate standard deviation
  sigma <- res[,(num_densities*2+1):(num_densities*3)]^2+res[,(num_densities+1):(num_densities*2)]^2
  sigma <- as.vector(rowSums(res[,1:num_densities] * sigma))-mean^2
  sigma <- sqrt(sigma)
  
  res <- cbind(res,mean,sigma)
  colnames(res) <- c(paste0("alpha_",1:num_densities),paste0("mean_",1:num_densities),paste0("sd_",1:num_densities),"mean_tot","sigma_tot")
  
  return(res)
}

# Evaluates the predictions on an individual cell basis, using the metrics presented in the 
# paper. It also determines the average rank and mean metric value for each of the models.
eval_preds_cells <- function(res,mdn_components,mdn_comp_nums,periods=10,all_models=c("odp","DT","DT2","DT_MDN","MDN")) {
  rmse <- data.frame(company=c(),AP=c(),DP=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
  wmape <- data.frame(company=c(),AP=c(),DP=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
  log_score <- data.frame(company=c(),AP=c(),DP=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
  crps <- data.frame(company=c(),AP=c(),DP=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
  qs_50 <- data.frame(company=c(),AP=c(),DP=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
  qs_75 <- data.frame(company=c(),AP=c(),DP=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
  qs_95 <- data.frame(company=c(),AP=c(),DP=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
  qs_995 <- data.frame(company=c(),AP=c(),DP=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
  
  for(company in unique(res$company_input)) {
    res_tmp <- as.data.frame(res[(res$company_input==company & res$AP+res$DP>periods+1),])
    
    mdn_comp_tmp <- mdn_components[(mdn_components$company_input==company & mdn_components$AP+mdn_components$DP>periods+1),]
    dt_mdn_comp_num <- mdn_comp_nums[1]
    mdn_comp_num <- mdn_comp_nums[2]
    dt_mdn_col_names <- paste0("DT_MDN_",c(paste0("alpha_",1:dt_mdn_comp_num),paste0("mu_",1:dt_mdn_comp_num),paste0("sigma_",1:dt_mdn_comp_num)))
    mdn_col_names <- paste0("MDN_",c(paste0("alpha_",1:mdn_comp_num),paste0("mu_",1:mdn_comp_num),paste0("sigma_",1:mdn_comp_num)))
    
    tmp_rmse   <- c(company)
    tmp_wmape  <- c(company)

    for(m in all_models) {
      tmp_rmse   <- c(tmp_rmse,sqrt((mean((res_tmp$loss - res_tmp[,paste0(m,"_mean")]))/sum(res_tmp$loss))^2))
      tmp_wmape  <- c(tmp_wmape,sum(abs(res_tmp$loss-res_tmp[,paste0(m,"_mean")]))/sum(res_tmp$loss))
    }
    
    tmp_ls <- c(company,
                odp=mean(log(dpois(floor(res_tmp$loss/res_tmp$odp_dispersion), res_tmp$odp_mean/res_tmp$odp_dispersion)/res_tmp$odp_dispersion)),
                dt=NA,
                dt2=NA,
                dt_mdn=mean(log_score_mdn(res_tmp$loss,mdn_comp_tmp[,dt_mdn_col_names])),
                mdn=mean(log_score_mdn(res_tmp$loss,mdn_comp_tmp[,mdn_col_names])))
    tmp_crps <- c(company,
                  odp=mean(res_tmp$odp_dispersion*crps_pois(res_tmp$loss/res_tmp$odp_dispersion, res_tmp$odp_mean/res_tmp$odp_dispersion)),
                  dt=NA,
                  dt2=NA,
                  dt_mdn=mean(crps_mixnorm(res_tmp$loss,as.matrix(mdn_comp_tmp[,dt_mdn_col_names[(dt_mdn_comp_num+1):(dt_mdn_comp_num*2)]]),
                                           as.matrix(mdn_comp_tmp[,dt_mdn_col_names[(dt_mdn_comp_num*2+1):(dt_mdn_comp_num*3)]]),
                                           as.matrix(mdn_comp_tmp[,dt_mdn_col_names[1:dt_mdn_comp_num]]))),
                  mdn=mean(crps_mixnorm(res_tmp$loss,as.matrix(mdn_comp_tmp[,mdn_col_names[(mdn_comp_num+1):(mdn_comp_num*2)]]),
                                        as.matrix(mdn_comp_tmp[,mdn_col_names[(mdn_comp_num*2+1):(mdn_comp_num*3)]]),
                                        as.matrix(mdn_comp_tmp[,mdn_col_names[1:mdn_comp_num]]))))
       tmp_qs_50 <- c(company,
                  odp=mean(qs_loss(res_tmp$loss,quantile_ccodp(res_tmp,0.5),0.5,model = "odp")$quant_odp_loss_50),
                  dt=NA, 
                  dt2=NA,
                  dt_mdn=mean(qs_loss(res_tmp$loss,quantile_mdn(res_tmp$DT_MDN_mean,res_tmp$DT_MDN_sigma,mdn_comp_tmp[,dt_mdn_col_names],0.5,set_tol = 0.001,max_iter = 100),0.5,model = "mdn")$quant_mdn_loss_50),
                  mdn=mean(qs_loss(res_tmp$loss,quantile_mdn(res_tmp$MDN_mean,res_tmp$MDN_sigma,mdn_comp_tmp[,mdn_col_names],0.5,set_tol = 0.001,max_iter = 100),0.5,model = "mdn")$quant_mdn_loss_50))
       tmp_qs_75 <- c(company,
                   odp=mean(qs_loss(res_tmp$loss,quantile_ccodp(res_tmp,0.75),0.75,model = "odp")$quant_odp_loss_75),
                   dt=NA, 
                   dt2=NA,
                   dt_mdn=mean(qs_loss(res_tmp$loss,quantile_mdn(res_tmp$DT_MDN_mean,res_tmp$DT_MDN_sigma,mdn_comp_tmp[,dt_mdn_col_names],0.75,set_tol = 0.001,max_iter = 100),0.75,model = "mdn")$quant_mdn_loss_75),
                   mdn=mean(qs_loss(res_tmp$loss,quantile_mdn(res_tmp$MDN_mean,res_tmp$MDN_sigma,mdn_comp_tmp[,mdn_col_names],0.75,set_tol = 0.001,max_iter = 100),0.75,model = "mdn")$quant_mdn_loss_75))
       tmp_qs_95 <- c(company,
                   odp=mean(qs_loss(res_tmp$loss,quantile_ccodp(res_tmp,0.95),0.95,model = "odp")$quant_odp_loss_95),
                   dt=NA, 
                   dt2=NA,
                   dt_mdn=mean(qs_loss(res_tmp$loss,quantile_mdn(res_tmp$DT_MDN_mean,res_tmp$DT_MDN_sigma,mdn_comp_tmp[,dt_mdn_col_names],0.95,set_tol = 0.001,max_iter = 100),0.95,model = "mdn")$quant_mdn_loss_95),
                   mdn=mean(qs_loss(res_tmp$loss,quantile_mdn(res_tmp$MDN_mean,res_tmp$MDN_sigma,mdn_comp_tmp[,mdn_col_names],0.95,set_tol = 0.001,max_iter = 100),0.95,model = "mdn")$quant_mdn_loss_95))
       tmp_qs_995 <- c(company,
                    odp=mean(qs_loss(res_tmp$loss,quantile_ccodp(res_tmp,0.995),0.995,model = "odp")$quant_odp_loss_99.5),
                    dt=NA, 
                    dt2=NA,
                    dt_mdn=mean(qs_loss(res_tmp$loss,quantile_mdn(res_tmp$DT_MDN_mean,res_tmp$DT_MDN_sigma,mdn_comp_tmp[,dt_mdn_col_names],0.995,set_tol = 0.001,max_iter = 100),0.995,model = "mdn")$quant_mdn_loss_99.5),
                    mdn=mean(qs_loss(res_tmp$loss,quantile_mdn(res_tmp$MDN_mean,res_tmp$MDN_sigma,mdn_comp_tmp[,mdn_col_names],0.995,set_tol = 0.001,max_iter = 100),0.995,model = "mdn")$quant_mdn_loss_99.5))
    rmse <- rbind(rmse,tmp_rmse)
    wmape <- rbind(wmape,tmp_wmape)
    log_score <- rbind(log_score,tmp_ls)
    crps <- rbind(crps,tmp_crps)
    qs_50 <- rbind(qs_50,tmp_qs_50)
    qs_75 <- rbind(qs_75,tmp_qs_75)
    qs_95 <- rbind(qs_95,tmp_qs_95)
    qs_995 <- rbind(qs_995,tmp_qs_995)
  }
  colnames(rmse) <- c("company","odp","dt","dt2","dt_mdn","mdn")
  colnames(wmape) <- c("company","odp","dt","dt2","dt_mdn","mdn")
  colnames(log_score) <- c("company","odp","dt","dt2","dt_mdn","mdn")
  colnames(crps) <- c("company","odp","dt","dt2","dt_mdn","mdn")
  colnames(qs_50) <- c("company","odp","dt","dt2","dt_mdn","mdn")
  colnames(qs_75) <- c("company","odp","dt","dt2","dt_mdn","mdn")
  colnames(qs_95) <- c("company","odp","dt","dt2","dt_mdn","mdn")
  colnames(qs_995) <- c("company","odp","dt","dt2","dt_mdn","mdn")
  metrics_list <- list(rmse=rmse,wmape=wmape,log_score=log_score,crps=crps,qs_50=qs_50,qs_75=qs_75,qs_95=qs_95,qs_995=qs_995)
  for(metric_ind in 1:length(metrics_list)) {
    fin_nums <- as.logical(is.finite(metrics_list[[metric_ind]]$odp)*is.finite(metrics_list[[metric_ind]]$dt_mdn)*
      is.finite(metrics_list[[metric_ind]]$mdn))
    if(names(metrics_list)[metric_ind] == "rmse") {
      metric_mean <- colSums(metrics_list[[metric_ind]][fin_nums,])
    } else {
      metric_mean <- colMeans(metrics_list[[metric_ind]][fin_nums,])
    }
    # if(names(metrics_list)[metric_ind]%in%c("wmape","log_score")) {
    #   metric_mean <- round(metric_mean,digits=4)
    # } else {
    #   metric_mean <- round(metric_mean/100000,digits=3)
    # }
    metric_mean[1] <- -1
    metrics_list[[metric_ind]] <- rbind(metrics_list[[metric_ind]],metric_mean)
    
    avg_rank <- rep(0,length(all_models))
    for(r_ind in 1:(nrow(metrics_list[[metric_ind]])-1)) {
      if(names(metrics_list)[metric_ind] == "log_score") {
        avg_rank <- avg_rank+rank(-metrics_list[[metric_ind]][r_ind,2:(length(all_models)+1)])
      } else {
        avg_rank <- avg_rank+rank(metrics_list[[metric_ind]][r_ind,2:(length(all_models)+1)])
      }
    }
    avg_rank <- avg_rank/(nrow(metrics_list[[metric_ind]])-1)
    
    metrics_list[[metric_ind]] <- rbind(metrics_list[[metric_ind]],c(-2,avg_rank))
  }
  
  return(list(rmse=metrics_list$rmse,wmape=metrics_list$wmape,log_score=metrics_list$log_score,
              crps=metrics_list$crps,qs_50=metrics_list$qs_50,qs_75=metrics_list$qs_75,
              qs_95=metrics_list$qs_95,qs_995=metrics_list$qs_995))
}

# Evaluates the predictions on a total reserves basis, using the metrics presented in the paper
eval_preds_TR <- function(preds) {
  tmp_preds <- as.data.frame(melt(data.table(preds[preds$AP+preds$DP>11,c(1:5,7:9,11)]),id.vars = c("company_input","AP","DP")))
  tmp_preds <- aggregate(tmp_preds$value,list(tmp_preds$company_input,tmp_preds$variable),FUN=sum)
  colnames(tmp_preds) <- c("company_input","variable","value")

  tmp_loss <- tmp_preds[tmp_preds$variable=="loss",]
  tmp_preds <- tmp_preds[tmp_preds$variable!="loss",]
  
  rmse <- c()
  wmape <- c()
  for (i in 1:length(unique(tmp_preds$variable))) {
    tmp_var <- as.character(unique(tmp_preds$variable)[i])
    rmse <- c(rmse,sqrt(mean((tmp_preds$value[tmp_preds$variable==tmp_var]-tmp_loss$value)^2)))
    wmape <- c(wmape,sum(abs(tmp_preds$value[tmp_preds$variable==tmp_var]-tmp_loss$value))/sum(tmp_loss$value))
  }
  res <- rbind(rmse,wmape)
  res <- as.data.frame(cbind(c("RMSE","wMAPE"),res))
  res[,2:6] <- lapply(res[,2:6],as.numeric)
  colnames(res) <- c("Metric","ODP","DT1","DT2","DT-MDN","MDN")
  return(res)
}

# Determines the log score for the MDN-based models.
log_score_mdn <- function(loss, mdn_components){
  mdn_components <- data.frame(mdn_components)
  num_dist <- ncol(mdn_components)/3
  log_score = 0
  for (i in 1:num_dist){
    log_score = log_score + mdn_components[,i]*dnorm(loss, mdn_components[,(i+num_dist)],mdn_components[,(i+num_dist*2)])
  }
  log_score <- log(log_score)
  return (log_score)
}

# Determines the value for a given quantile for the ccODP model.
quantile_ccodp <- function(res_odp, quantile){
  res <- as.data.frame(copy(res_odp))
  quant_est_odp = c()
  
  for (i in 1:nrow(res)){
    quant_est_odp <- c(quant_est_odp, res$odp_dispersion[i]*qpois(quantile, res$odp_mean[i]/res$odp_dispersion[i]))
  }
  res <- cbind(res,quant_est_odp)
  colnames(res)[ncol(res)] <- paste0("quant_odp_",100*quantile)
  
  return (res)
}

# Determines the value for a given quantile for MDN-based models.
quantile_mdn <- function(mdn_mean,mdn_sigma,res_mdn, quantile, set_tol=0.001, max_iter=100){
  res <- as.data.frame(res_mdn)
  
  res$quantile_est <- mdn_mean
  res$jump <- mdn_sigma
  res$quantile <- 0
  res$old_quantile <- 0
  res$tol <- 1
  
  num_dist <- ncol(res_mdn)/3
  incomplete <- abs(res$tol) > set_tol
  
  dist_dat <- copy(res)
  
  for (i in 1:num_dist){
    alpha <- dist_dat[incomplete,i]
    mu <- dist_dat[incomplete,i + num_dist]
    sigma <- dist_dat[incomplete, i + 2*num_dist]
    res$quantile[incomplete] <- res$quantile[incomplete] + alpha*pnorm(res$quantile_est[incomplete], mu, sigma)
  }
  
  res$old_quantile <- res$quantile
  res$tol <- (res$quantile - quantile)
  res$jump[res$tol > 0] <- -1*res$jump[res$tol > 0]
  
  ##INITIAL QUANTILE VALUES FILLED IN 
  t <- 0
  
  ##START quantile search
  while (max(abs(res$tol)) > set_tol && t <= max_iter){
    incomplete <- abs(res$tol) > set_tol
    
    res$quantile_est[incomplete] <- res$quantile_est[incomplete] + res$jump[incomplete]
    res$quantile[incomplete] <- 0
    for (i in 1:num_dist){
      alpha <- dist_dat[incomplete,i]
      mu <- dist_dat[incomplete,i + num_dist]
      sigma <- dist_dat[incomplete, i + 2*num_dist]
      res$quantile[incomplete] <- res$quantile[incomplete] + alpha*pnorm(res$quantile_est[incomplete], mu, sigma)
    }
    
    ##NEW QUANTILE VALUES CALCULATED
    res$tol[incomplete] <- (res$quantile[incomplete] - quantile)
    went_above <- res$old_quantile - quantile < 0 & res$quantile - quantile > 0
    went_below <- res$old_quantile - quantile > 0 & res$quantile - quantile < 0
    res$jump[c(went_above & incomplete)] <- -0.5 * res$jump[c(went_above & incomplete)]
    res$jump[c(went_below & incomplete)] <- -0.5 * res$jump[c(went_below & incomplete)]
    res$old_quantile[incomplete] <- res$quantile[incomplete]

    t <- t + 1
  }
  
  colnames(res)[which(colnames(res)== "quantile_est")] <- paste0("quant_mdn_", quantile*100)
  
  res <- as.data.table(res)
  
  res[ ,`:=`(tol = NULL, quantile = NULL, old_quantile = NULL, jump = NULL)]
  return (res)  
}

# Determines the quantile score for the ccODP and MDN-based models. 
qs_loss <- function(loss,res_model, quantile, model){
  res <- as.data.frame(copy(res_model))
  
  if(model=="mdn") {
    ind_str <- paste0("quant_mdn_",100*quantile)
    col_str <- paste0("quant_mdn_loss_",100*quantile)
  } else {
    ind_str <- paste0("quant_odp_",100*quantile)
    col_str <- paste0("quant_odp_loss_",100*quantile)
  }
  ind <- which(colnames(res) == ind_str)
  quantile_loss <- (loss - res[,ind])*(quantile - (loss < res[,ind]))  
  res <- cbind(res,quantile_loss)
  colnames(res)[ncol(res)] = col_str

  return (res)
}

# Aggregates a triangle based on quarterly data into a triangle based on yearly data.
aggregate_qs <- function(q_table) {
  num_companies <- nrow(q_table)/1600
  CALP <- q_table$DP+q_table$AP-1

  y_table <- data.frame(matrix(ncol = ncol(q_table), nrow = 0))
  colnames(y_table) <- colnames(q_table)
  
  for(company in 1:num_companies) {
    for(i in 1:10) {
    AP_min <- 4*(i-1)
    AP_max <- 4*i
      for(j in 1:10) {
        CALP_min <- AP_min+4*(j-1)
        CALP_max <- AP_min+4*j
        tmp <- q_table[(company_input == company) & (AP>AP_min & AP<=AP_max) & (CALP>CALP_min & CALP<=CALP_max)]
        if(j==10) {
          tmp <- q_table[(company_input == company) & (AP>AP_min & AP<=AP_max) & (CALP>CALP_min)]
        }
        tmp <- data.frame(tmp)
        new_row <- c(company,i,j,sum(tmp$loss),
                     sum(tmp$odp_mean),sum(tmp$odp_dispersion),
                     sum(tmp$DT_mean),sum(tmp$DT2_mean),
                     sum(tmp$DT_MDN_mean),sqrt(sum(tmp$DT_MDN_sigma^2)),
                     sum(tmp$MDN_mean),sqrt(sum(tmp$MDN_sigma^2)))
        y_table <- rbind(y_table,new_row)
      }
    }
  }
  colnames(y_table) <- colnames(q_table)
  return(y_table)
}