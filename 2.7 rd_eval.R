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

sel_data <- "rd_ol" #"rd_ca", "rd_ppa", "rd_wc", "rd_ol"
num_periods <- 10

## Data
data_odp <- as.data.frame(fread(paste0("output_rd/",sel_data,"/",sel_data,"_odp.csv")))
preds <- data_odp
components <- data.frame(preds$company_input,preds$AP,preds$DP)
colnames(components) <- c("company_input","AP","DP")

model <- "DT"
data_DT <- as.data.frame(fread(paste0("output_rd/",sel_data,"/",sel_data,"_",model,".csv")))
preds <- cbind(preds,data_DT$DT_mean)
colnames(preds)[ncol(preds)] <- "DT_mean"

model <- "DT2"
data_DT2 <- as.data.frame(fread(paste0("output_rd/",sel_data,"/",sel_data,"_",model,".csv")))
preds <- cbind(preds,data_DT2$DT2_mean)
colnames(preds)[ncol(preds)] <- "DT2_mean"

for(model in c("DT-MDN","MDN")) {
  tmp_data <- as.data.frame(fread(paste0("output_rd/",sel_data,"/",sel_data,"_",model,".csv")))
  preds <- cbind(preds,tmp_data$mdn_mean,tmp_data$mdn_sigma)
  colnames(preds)[(ncol(preds)-1):ncol(preds)] <- paste0(gsub("-", "_",model),c("_mean","_sigma"))
  
  num_components <- (ncol(tmp_data)-6)/3
  tmp_components <- tmp_data[,4:(num_components*3+3)]
  colnames(tmp_components) <- paste0(gsub("-", "_",model),"_",colnames(tmp_components))
  components <- cbind(components,tmp_components)
}

## Predictions
companies <- unique(preds$company_input)

company <- companies[8]
aqs <- c(4,8)
plot_compare_all_gg(preds, aqs[1], company, timeframe = "Y", horizon = 10)
plot_compare_all_gg(preds, aqs[2], company, timeframe = "Y", horizon = 10)

# Metrics tables
metrics <- eval_preds_cells(preds,components,mdn_comp_nums = c(2,2,2),periods = 10)

plot_eval_box(metrics$rmse,"RMSE")
plot_eval_box(metrics$wmape,"wMAPE")
plot_eval_box(metrics$log_score,"Log Score",only_mdn = TRUE)
plot_eval_box(metrics$crps,"CRPS",only_mdn = TRUE)

aggr_table <- data.frame(metric=c(),odp=c(),dt=c(),dt2=c(),dt_mdn=c(),mdn=c())
for(sel_metric in 1:length(metrics)) {
  tmp_met <- metrics[[sel_metric]]
  new_row <- c(names(metrics)[sel_metric],
               paste0(round(tmp_met[tmp_met$company==-1,2:6],4)," (",round(tmp_met[tmp_met$company==-2,2:6],2),")"))
  aggr_table <- rbind(aggr_table,new_row)
}

aggr_table <- as.data.frame(aggr_table[c(1,2,3,4,6,7,8),])

colnames(aggr_table) <- c("Metric","ODP","DT1","DT2","DT-MDN","MDN")
aggr_table[,1] <- c("RMSE","wMAPE","Log Score","CRPS","QS 75%","QS 95%","QS 99.5%")
aggr_table[3:8,3:4] <- "-"
stargazer(aggr_table,summary = FALSE,rownames = FALSE)

tr_table <- eval_preds_TR(preds)
tr_table[1,2:6] <- round(tr_table[1,2:6],0)
tr_table[2,2:6] <- round(tr_table[2,2:6],4)
stargazer(tr_table,summary = FALSE,rownames = FALSE,digits = 5)

