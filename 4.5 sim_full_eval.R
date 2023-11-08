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
# Evaluation of the forecasts for the real data environments

# Import necessary packages, models and support functions
source("0.1 support_funcs.R")
source("0.2 plot_funcs.R")

sel_env <- 4
in_dir <- paste0("output_sim_full/env_",sel_env)
data_env <- paste0("sim_env",sel_env)

#### ccODP model ####
num_periods <- 10

raw_data_dir <- paste0("sim/env ",sel_env)
triangles <- load_sim_triangles(raw_data_dir,50,10)

list_data_sim <- construct_list_data_multi(triangles)
data_odp <- list_data_sim %>% select(index_company,AP,DP,value)

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

fwrite(data_odp, paste0(in_dir,"/", data_env,"_odp.csv"))



#### Evaluation ####
sel_env <- 1
in_dir <- paste0("output_sim_full/env_",sel_env)
data_env <- paste0("sim_env",sel_env)

data_odp <- fread(paste0(in_dir,"/", data_env,"_odp.csv"))
preds <- data_odp
components <- data.frame(preds$company_input,preds$AP,preds$DP)
colnames(components) <- c("company_input","AP","DP")
    
model <- "DT"
data_DT <- fread(paste0(in_dir,"/", data_env,"_",model,".csv"))[1:5000,]
# data_DT <- fread(paste0(in_dir,"/", data_env,"_",model,".csv"))
preds <- cbind(preds,data_DT[,list(DT_pred)])
colnames(preds)[ncol(preds)] <- "DT_mean"
    
model <- "DT2"
data_DT2 <- fread(paste0(in_dir,"/", data_env,"_",model,".csv"))
preds <- cbind(preds,data_DT2[,DT_pred])
colnames(preds)[ncol(preds)] <- "DT2_mean"

mdn_comp_nums <- c()

for(model in c("DT-MDN","MDN")) {
  tmp_data <- fread(paste0(in_dir,"/", data_env,"_",model,".csv"))
  preds <- cbind(preds,tmp_data[,list(mdn_mean,mdn_sigma)])
  colnames(preds)[(ncol(preds)-1):ncol(preds)] <- paste0(gsub("-", "_",model),c("_mean","_sigma"))
  
  num_components <- (ncol(tmp_data)-6)/3
  mdn_comp_nums <- c(mdn_comp_nums,num_components)
  tmp_components <- tmp_data[,4:(num_components*3+3)]
  colnames(tmp_components) <- paste0(gsub("-", "_",model),"_",colnames(tmp_components))
  components <- cbind(components,tmp_components)
}

# Plot of predicts for a given company
companies <- unique(preds$company_input)
company <- companies[8]
aqs <- c(4,8)
plot_compare_all_gg(preds, aqs[1], company, timeframe = "Y", horizon = 10)
plot_compare_all_gg(preds, aqs[2], company, timeframe = "Y", horizon = 10)

# Metrics tables
metrics <- eval_preds_cells(preds,components,mdn_comp_nums = mdn_comp_nums,periods = 10)

plot_eval_box(metrics$rmse,"RMSE",skip_mdn = TRUE)
plot_eval_box(metrics$wmape,"wMAPE",skip_mdn = TRUE)
plot_eval_box(metrics$log_score,"Log Score",only_mdn = TRUE,skip_mdn = TRUE)
plot_eval_box(metrics$crps,"CRPS",only_mdn = TRUE,skip_mdn = TRUE,num_break = 3)

aggr_table <- data.frame(metric=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
for(sel_metric in 1:length(metrics)) {
  tmp_met <- metrics[[sel_metric]]
  if(names(metrics)[sel_metric]%in%c("rmse","wmape","log_score")) {
    new_row <- c(names(metrics)[sel_metric],
                 paste0(round(tmp_met[tmp_met$company==-1,2:6],6)," (",round(tmp_met[tmp_met$company==-2,2:6],2),")"))
  } else {
    new_row <- c(names(metrics)[sel_metric],
                 paste0(round(tmp_met[tmp_met$company==-1,2:6],0)," (",round(tmp_met[tmp_met$company==-2,2:6],2),")"))
  }
  aggr_table <- rbind(aggr_table,new_row)
}

aggr_table <- aggr_table[c(1,2,3,4,6,7,8),]
colnames(aggr_table) <- c("Metric","ODP","DT1","DT2","DT-MDN","MDN")
aggr_table[,1] <- c("RMSE","wMAPE","Log Score","CRPS","QS 75%","QS 95%","QS 99.5%")
aggr_table[3:7,3:4] <- "-"
stargazer(aggr_table,summary = FALSE,rownames = FALSE)

tr_table <- as.data.frame(eval_preds_TR(preds))
tr_table[1:2,2:6] <- round(tr_table[1,2:6],5)
stargazer(tr_table,summary = FALSE,rownames = FALSE,digits = 5)

