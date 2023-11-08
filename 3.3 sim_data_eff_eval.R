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
# Evaluation of forecast of the predictions for the data and granularity exercie

# Import necessary packages, models and support functions
source("0.1 support_funcs.R")
source("0.2 plot_funcs.R")

in_dir <- "output_sim_data"
data_env <- "sim_env1"

preds_data <- list()
preds_tables <- list()
mdn_tables <- list()

for(period_type in c("y","q")) {
  odp_data <- fread(paste0(in_dir,"/", data_env,"_odp_",period_type,".csv"))
  for(num_triangles in c(1,5,10,50,100,200)) {
    print(paste0("Period type: ",period_type,". Number of triangles: ",num_triangles,"."))
    curr_table <- odp_data[odp_data$company_input<=num_triangles,list(company_input,AP,DP,loss,odp_mean,odp_dispersion)]
    curr_components <- data.frame(curr_table$company_input,curr_table$AP,curr_table$DP)
    colnames(curr_components) <- c("company_input","AP","DP")
    
    model <- "DT"
    tmp_data <- fread(paste0(in_dir,"/", data_env,"_",num_triangles,"_",period_type,"_DT.csv"))
    preds_data <- append(preds_data,list(list(period_type=period_type,num_triangles=num_triangles,model=model,data=tmp_data)))
    curr_table <- cbind(curr_table,tmp_data[,list(DT_pred)])
    colnames(curr_table)[ncol(curr_table)] <- "DT_mean"
    
    model <- "DT2"
    tmp_data <- fread(paste0(in_dir,"/", data_env,"_",num_triangles,"_",period_type,"_DT2.csv"))
    preds_data <- append(preds_data,list(list(period_type=period_type,num_triangles=num_triangles,model=model,data=tmp_data)))
    curr_table <- cbind(curr_table,tmp_data[,DT_pred])
    colnames(curr_table)[ncol(curr_table)] <- "DT2_mean"
    
    num_comps_tmp <- c()
    for(model in c("DT-MDN","MDN")) {
      tmp_data <- fread(paste0(in_dir,"/", data_env,"_",num_triangles,"_",period_type,"_",model,".csv"))
      preds_data <- append(preds_data,list(list(period_type=period_type,num_triangles=num_triangles,model=model,data=tmp_data)))
      
      curr_table <- cbind(curr_table,tmp_data[,list(mdn_mean,mdn_sigma)])
      colnames(curr_table)[(ncol(curr_table)-1):ncol(curr_table)] <- paste0(gsub("-", "_",model),c("_mean","_sigma"))
      
      num_components <- (ncol(tmp_data)-6)/3
      num_comps_tmp <- c(num_comps_tmp,num_components)
      tmp_components <- tmp_data[,4:(num_components*3+3)]
      colnames(tmp_components) <- paste0(gsub("-", "_",model),"_",colnames(tmp_components))
      curr_components <- cbind(curr_components,tmp_components)
    }
    
    preds_tables <- append(preds_tables,list(list(period_type=period_type,num_triangles=num_triangles,table=curr_table)))
    mdn_tables <- append(mdn_tables,list(list(period_type=period_type,num_triangles=num_triangles,table=curr_components,num_comp=num_comps_tmp)))
  }
}

# Metrics table
metrics_tables <- list()
aggr_table <- data.frame(time_period=c(),num_companies=c(),metric=c(),odp=c(),dt1=c(),dt2=c(),dt_mdn=c(),mdn=c())
for(t in 1:length(preds_tables)) {
  tmp_pt <- preds_tables[[t]]$period_type
  tmp_nt <- preds_tables[[t]]$num_triangles
  print(paste0("Period type: ",tmp_pt,". Number of triangles: ",tmp_nt,"."))
  
  tmp_preds <- preds_tables[[t]]$table
  tmp_components <- mdn_tables[[t]]$table
  tmp_metrics <- eval_preds_cells(tmp_preds,tmp_components,mdn_tables[[t]]$num_comp,periods = 10)
  
  metrics_tables <- append(metrics_tables,list(period_type=tmp_pt,num_triangles=tmp_nt,metrics_info=tmp_metrics))
  
  for(sel_metric in 1:length(tmp_metrics)) {
    tmp_met <- tmp_metrics[[sel_metric]]
    tmp_met_name <- names(tmp_metrics)[sel_metric]
    if(tmp_met_name %in% c("rmse","wmape","log_score")) {
      new_row <- c(tmp_pt,tmp_nt,names(tmp_metrics)[sel_metric],
                   paste0(round(tmp_met[tmp_met$company==-1,2:6],7)," (",round(tmp_met[tmp_met$company==-2,2:6],2),")"))
    } else {
      new_row <- c(tmp_pt,tmp_nt,names(tmp_metrics)[sel_metric],
                   paste0(round(tmp_met[tmp_met$company==-1,2:6],0)," (",round(tmp_met[tmp_met$company==-2,2:6],2),")"))
    }
    aggr_table <- rbind(aggr_table,new_row)
  }
  aggr_table[((nrow(aggr_table)-7):nrow(aggr_table)),3] <- c("RMSE","wMAPE","Log Score","CRPS","QS 50%","QS 75%","QS 95%","QS 99.5%")
}
colnames(aggr_table) <- c("Time Period","Number of Triangles","Metric","ODP","DT","DT2","DT-MDN","MDN")

stargazer(aggr_table[1:(nrow(aggr_table)/2),3:ncol(aggr_table)],summary = FALSE,rownames = FALSE,digits=7)
stargazer(aggr_table[(nrow(aggr_table)/2+1):nrow(aggr_table),3:ncol(aggr_table)],summary = FALSE,rownames = FALSE,digits=5)

# Examples of plots
tmp <- preds_tables[[2]]$table
company <- 5
aqs <- c(3,6,9)
plot_compare_all_gg(tmp, aqs[1], company, timeframe = "Y", horizon = 10)
plot_compare_all_gg(tmp, aqs[2], company, timeframe = "Y", horizon = 10)
plot_compare_all_gg(tmp, aqs[3], company, timeframe = "Y", horizon = 10)

tmp <- preds_tables[[5]]$table
company <- 5
aqs <- c(3,6,9)
plot_compare_all_gg(tmp, aqs[1], company, timeframe = "Y", horizon = 10)
plot_compare_all_gg(tmp, aqs[2], company, timeframe = "Y", horizon = 10)
plot_compare_all_gg(tmp, aqs[3], company, timeframe = "Y", horizon = 10)

tmp <- preds_tables[[9]]$table
company <- 5
aqs <- c(10,20,30)
plot_compare_all_gg(tmp, aqs[1], company, timeframe = "Q", horizon = 40)
plot_compare_all_gg(tmp, aqs[2], company, timeframe = "Q", horizon = 40)
plot_compare_all_gg(tmp, aqs[3], company, timeframe = "Q", horizon = 40)

tmp <- preds_tables[[12]]$table
company <- 5
aqs <- c(10,20,30)
plot_compare_all_gg(tmp, aqs[1], company, timeframe = "Q", horizon = 40)
plot_compare_all_gg(tmp, aqs[2], company, timeframe = "Q", horizon = 40)
plot_compare_all_gg(tmp, aqs[3], company, timeframe = "Q", horizon = 40)


# Boxplots of the metrics
eval_res <- eval_preds_cells(preds_tables[[12]]$table,mdn_tables[[12]]$table,mdn_tables[[12]]$num_comp,periods = 10)
plot_eval_box(eval_res$rmse,"RMSE")
plot_eval_box(eval_res$wmape,"wMAPE")
plot_eval_box(eval_res$log_score,"Log Score")
plot_eval_box(eval_res$crps,"CRPS")