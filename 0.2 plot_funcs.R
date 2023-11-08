# 
# Project code for Master Thesis for MSc Data Science and Business Analytics
# Bocconi University
#
# Name:           Sean Goedgeluk 
# Student number: 3190974
# Supervised by: Prof.dr. Giacomo Zanella
#
# Title: 
#
# File Description:
# Defines the necessary support functions that facilitate any plots required 
# for this research.

# Plots the density of an MDN-based prediction (APA Style)
plot_density <- function(params) {
  num_densities <- length(params)/3
  params_data <- data.frame(density=as.factor(1:num_densities),mu=params[(num_densities+1):(num_densities*2)],sigma=params[(num_densities*2+1):(num_densities*3)],lambda=params[1:num_densities])
  
  density_data <- data.frame()
  for(i in 1:num_densities) {
    density_data <- rbind(density_data,data.frame(density=rep(i,100000),value=rnorm(100000,mean = params[i+num_densities],sd = params[i+2*num_densities])))
  }
  density_data$density <- as.factor(density_data$density)
  
  plot <- ggplot(density_data,aes(x=value,linetype=density,color=density)) + 
    geom_density(adjust=0.1,size=1) + 
    geom_vline(data=data.frame(density=params_data$density,mu=params_data$mu), size=1, 
               aes(xintercept=mu, linetype=density, color=density)) +
    xlab("Loss") +
    ylab("") +
    theme_apa()
  plot
}

# Plots the claims for a triangle based on yearly data for a given set of accident years, using ggplot. (APA Style)
plot_claims_gg = function(data_tab, AYs){
  options(scipen = 999)
  tmp <- data.frame(1:10,t(data_tab[AYs,1:10]))
  colnames(tmp) <- c("DY",paste0("AY ", AYs))
  rownames(tmp) <- 1:10
  tmp_long <- melt(data.table(tmp), id = "DY")
  plot <-ggplot(tmp_long, aes(x=DY, y=value, linetype=variable, colour=variable)) +
    geom_line(size=.75) + 
    ylab("") +
    xlab("Development Year") +
    scale_y_continuous(labels = scales::comma) +
    theme_apa(legend.pos = "none")
  plot
}

# Plots the claims for a triangle based on quarterly data for a given set of accident quarters, using ggplot. (APA Style)
plot_claims_gg_q = function(data_tab, AQs){
  options(scipen = 999)
  tmp <- data.frame(1:40,t(data_tab[AQs,1:40]))
  colnames(tmp) <- c("DQ",paste0("AQ ", AQs))
  rownames(tmp) <- 1:40
  tmp_long <- melt(data.table(tmp), id = "DQ")
  plot <-ggplot(tmp_long, aes(x=DQ, y=value, linetype=variable, colour=variable)) +
    geom_line(size=.75) + 
    ylab("") +
    xlab("Development Quarter") +
    scale_y_continuous(labels = scales::comma) +
    theme_apa(legend.pos = "none")
  plot
}

# Plots predictions for all models, as well as confidence intervals for the MDN-based models,
# as well as the losses. (APA Style)
plot_compare_all_gg <- function(res_data, aqs, company, timeframe, horizon){
  options(scipen = 999)
  
  line_size_set <- .5
  
  plot_res <- as.data.table(copy(res_data))
  plot_res <- plot_res[company_input==company]
  
  min_odp <- min(plot_res$odp_mean)
  max_odp <- max(plot_res$odp_mean)
  min_dt <- min(plot_res$odp_mean)
  max_dt <- max(plot_res$odp_mean)
  min_dt2 <- min(plot_res$odp_mean)
  max_dt2 <- max(plot_res$odp_mean)
  min_mdn <-  min(plot_res$MDN_mean - plot_res$MDN_sigma)
  max_mdn <-  max(plot_res$MDN_mean + plot_res$MDN_sigma)
  min_dtmdn <-  min(plot_res$DT_MDN_mean - plot_res$DT_MDN_sigma)
  max_dtmdn <-  max(plot_res$DT_MDN_mean + plot_res$DT_MDN_sigma)
  min_loss <- min(plot_res$loss)
  max_loss <- max(plot_res$loss)
  
  min_tot <- min(min_odp, min_dt, min_dt2, min_mdn, min_dtmdn, min_loss)
  max_tot <- max(max_odp, max_dt, max_dt2, max_mdn, max_dtmdn, max_loss)
  
  plots_list <- list()
  for (i in 1:length(aqs)){
    x_min_tmp <- min(horizon,(horizon+1.90-aqs[i]))
    print(x_min_tmp)
    
    plot_res_tmp <- plot_res[(AP==aqs[i] & DP <= horizon)]
    
    plot <- ggplot(plot_res_tmp, aes(x=DP)) + 
      geom_rect(aes(xmin = x_min_tmp, xmax = horizon, ymin = min_tot, ymax = max_tot), fill = "grey95", alpha=0.2)
    
    if(x_min_tmp!=horizon) {
      plot <- plot + geom_vline(xintercept = x_min_tmp, color="gray")
    }
    
    plot <- plot + geom_line(aes(y = loss), color = "black", size=line_size_set) + 
      geom_line(aes(y = DT_mean), color="purple", size=line_size_set) +
      geom_line(aes(y = DT2_mean), color="blue", size=line_size_set) +
      geom_line(aes(y = odp_mean), color="forestgreen", size=line_size_set) +
      geom_line(aes(y = MDN_mean), color="red", size=line_size_set) +
      geom_line(aes(y = DT_MDN_mean), color="orange", size=line_size_set) +
      geom_line(aes(y = MDN_mean+MDN_sigma), color="red", linetype="twodash", size=line_size_set) +
      geom_line(aes(y = MDN_mean-MDN_sigma), color="red", linetype="twodash", size=line_size_set) +
      geom_line(aes(y = DT_MDN_mean+DT_MDN_sigma), color="orange", linetype="twodash", size=line_size_set) +
      geom_line(aes(y = DT_MDN_mean-DT_MDN_sigma), color="orange", linetype="twodash", size=line_size_set) +
      xlim(c(1,horizon)) +
      ylim(c(min_tot,max_tot)) +
      scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
      ylab("") +
      theme_apa()
    
    if(timeframe == "Y") {
      plot <- plot + xlab("Development Year")
      if(length(aqs)>1) {
        plot <- plot + ggtitle(paste0("Accident Year ",aqs[i]))
      }
    } else {
      plot <- plot + xlab("Development Quarter")
      if(length(aqs)>1) {
        plot <- plot + ggtitle(paste0("Accident Quarter ",aqs[i]))
      }
    }
    
    plots_list <- append(plots_list,list(plot))
  }
  
  if(length(aqs) > 1) {
    grid.arrange(plots_list[[1]],plots_list[[2]],plots_list[[3]],plots_list[[4]], ncol=2)
  } else {
    plot
  }
}

# Plots a boxplot for a given metric including subsets of models presented in the paper.
plot_eval_box <- function(res_metrics,metric,only_mdn=FALSE,skip_mdn=FALSE,num_break=4) {
  res <- copy(res_metrics)
  colnames(res) <- c("company","ccODP","DT","DT2","DT-MDN","MDN")
  
  if(only_mdn) {
    res <- res[,c(1,2,5,6)]
  }
  
  if(skip_mdn) {
    res <- res[,colnames(res)!="MDN"]
  }
  
  tmp_long <- melt(data.table(res),id.vars = c("company"))
  bounds <- quantile(tmp_long$value, c(0.1, 0.9),na.rm = TRUE)
  tmp_long <- tmp_long[tmp_long$value>=bounds[1] & tmp_long$value<= bounds[2],]
  median_odp <- median(tmp_long$value[tmp_long$variable=="ccODP"])
  plot <- ggplot(tmp_long, aes(x=variable,y=value)) +
    geom_boxplot() + #outlier.shape = NA
    geom_hline(yintercept = median_odp,linetype="dashed") +
    xlab("") +
    ylab("") +
    scale_y_continuous(breaks = scales::pretty_breaks(n = num_break)) +
    coord_flip() +
    theme_apa()
  plot
}
