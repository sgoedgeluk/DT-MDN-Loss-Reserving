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
# Estimation of the ccODP model

source("0.1 support_funcs.R")
source("2.1 rd_keras.R")

## Output directory ##
data_used <- "rd_wc"
sel_lob <- "workers_compensation" #"commercial_auto", "other_liability", "private_passenger_auto", "workers_compensation"

num_periods <- 10

## Data ##
data_lob <- data_with_features_sd[data_with_features_sd$lob==sel_lob,]
data_odp <- cbind(as.numeric(data_lob$group_code),
                  as.numeric(data_lob$accident_year-1987),
                  as.numeric(data_lob$development_lag),
                  as.numeric(data_lob$incremental_paid_loss))

data_odp <- cbind(data_odp,rep(0,nrow(data_odp)),rep(0,nrow(data_odp)))
colnames(data_odp) <- c("company_input","AP","DP","loss","odp_mean","odp_dispersion")
data_odp <- as.data.frame(data_odp)
data_odp[data_odp$loss<0,]$loss <- 0

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

fwrite(data_odp,paste0("output_rd/",data_used,"/",data_used,"_odp.csv"))
