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
# Prepares the data for the real data cases, including normalization and plotting/calculating
# important metrics on the data

# Import necessary packages and support functions
library(moments)

source("0.1 support_funcs.R")

#### Load data ####
data(schedule_p)

data_with_features_sd <- schedule_p %>%
  right_join(dt_group_codes, by = c("lob", "group_code")) %>%
  mutate(case_reserves = incurred_loss - cumulative_paid_loss) %>%
  group_by(lob, group_code, accident_year) %>%
  arrange(lob, group_code, accident_year, development_lag) %>%
  mutate(
    incremental_paid_actual = incremental_paid_loss,
    incremental_paid = ifelse(
      calendar_year <= 1998, #Changed from 1997
      incremental_paid_actual, 
      NA_real_
    )
  ) %>%
  ungroup() %>%
  mutate(
    bucket = case_when(
      calendar_year <= 1995 & development_lag > 1 ~ "train",
      calendar_year > 1995 & calendar_year <= 1997 &
        development_lag > 1 ~ "validation",
      calendar_year > 1997 ~ "test"
    ),
    cumulative_paid_actual = cumulative_paid_loss,
    cumulative_paid = ifelse(
      calendar_year <= 1997,
      cumulative_paid_actual, NA_real_
    ),
    case_reserves_actual = case_reserves,
    case_reserves = ifelse(
      calendar_year <= 1997,
      case_reserves_actual,
      NA_real_
    )
  )


#### Determine mean and std. dev. for normalization ####
data_tmp <- data_with_features_sd[data_with_features_sd$bucket %in% c("train","validation"),]
  
# mean and sd by lob
mean_lob <- aggregate(data.frame(data_tmp$incremental_paid_actual,data_tmp$cumulative_paid_actual,data_tmp$case_reserves_actual),
                      list(data_tmp$lob),mean)
sd_lob   <- aggregate(data.frame(data_tmp$incremental_paid_actual,data_tmp$cumulative_paid_actual,data_tmp$case_reserves_actual),
                      list(data_tmp$lob),sd)
colnames(mean_lob) <- c("lob","inc_mean","cum_mean","case_mean")
colnames(sd_lob) <- c("lob","inc_sd","cum_sd","case_sd")

# mean and sd by lob and company
mean_lob_company <- aggregate(data.frame(data_tmp$incremental_paid_actual,data_tmp$cumulative_paid_actual,data_tmp$case_reserves_actual),
                              list(data_tmp$lob,data_tmp$group_code),mean)
sd_lob_company   <- aggregate(data.frame(data_tmp$incremental_paid_actual,data_tmp$cumulative_paid_actual,data_tmp$case_reserves_actual),
                              list(data_tmp$lob,data_tmp$group_code),sd)
colnames(mean_lob_company) <- c("lob","group_code","inc_mean","cum_mean","case_mean")
colnames(sd_lob_company) <- c("lob","group_code","inc_sd","cum_sd","case_sd")
mean_lob_company <- arrange(mean_lob_company,lob,group_code)
sd_lob_company <- arrange(sd_lob_company,lob,group_code)

#### Normalize data ####
data_with_features_sd <- data_with_features_sd %>%
  left_join(mean_lob_company) %>%
  left_join(sd_lob_company) %>% 
  mutate(
    incremental_paid = (incremental_paid - inc_mean) / inc_sd,
    incremental_paid_actual = (incremental_paid_actual - inc_mean) / inc_sd,
    cumulative_paid = (cumulative_paid - cum_mean) / cum_sd,
    cumulative_paid_actual = (cumulative_paid_actual - cum_mean) / cum_sd,
    case_reserves = (case_reserves - case_mean) / case_sd,
    case_reserves_actual = (case_reserves_actual - case_mean) / case_sd
  )

#### Indexing company code ####
company_index_recipe_sd <- recipe(~ group_code, data = data_with_features_sd) %>%
  step_integer(group_code, zero_based = TRUE) %>%
  prep()

#### Summary statistics ####
stat_mean <- as.data.frame(t(aggregate(data_with_features_sd$incremental_paid_loss,list(data_with_features_sd$lob),FUN=mean)))
stat_median <- as.data.frame(t(aggregate(data_with_features_sd$incremental_paid_loss,list(data_with_features_sd$lob),FUN=median)))
stat_sd <- as.data.frame(t(aggregate(data_with_features_sd$incremental_paid_loss,list(data_with_features_sd$lob),FUN=sd)))
stat_kurt <- as.data.frame(t(aggregate(data_with_features_sd$incremental_paid_loss,list(data_with_features_sd$lob),FUN=kurtosis)))
stat_skew <- as.data.frame(t(aggregate(data_with_features_sd$incremental_paid_loss,list(data_with_features_sd$lob),FUN=skewness)))
stat_quantile <- as.data.frame(t(aggregate(data_with_features_sd$incremental_paid_loss,list(data_with_features_sd$lob),FUN=quantile)))

stats_tot <- rbind(stat_mean[2,],stat_median[2,],stat_sd[2,],stat_kurt[2,],stat_skew[2,],stat_quantile[2:6,])
colnames(stats_tot) <- c("Commercial Auto (CA)","Other Liability (OL)","Private Passenger Auto (PPA)","Workers' Compensation (WC)")
rownames(stats_tot) <- c("Mean","Median","Standard Deviation","Kurtosis","Skewness","Quantile 0%","Quantile 25%","Quantile 50%","Quantile 75%","Quantile 100%")

stargazer(stats_tot,summary = FALSE,rownames = TRUE)

# Plot boxplots of losses by development year
sel_lob <- "workers_compensation" #"commercial_auto" "other_liability" "private_passenger_auto" "workers_compensation"  
plot_preds <- data_with_features_sd[data_with_features_sd$lob==sel_lob,]
plot_preds <- data.frame(development_lag=plot_preds$development_lag,incremental_paid_loss=plot_preds$incremental_paid_loss)
plot_preds$development_lag <- as.factor(plot_preds$development_lag)

# bounds <- quantile(tmp_long$value, c(0.1, 0.9),na.rm = TRUE)
plot <- ggplot(plot_preds, aes(x=development_lag,y=incremental_paid_loss)) +
  geom_boxplot() +
  xlab("") +
  ylab("") +
  theme_apa()
plot
