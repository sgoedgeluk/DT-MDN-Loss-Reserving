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
# Plots the simulated data for each scenario

source("0.1 support_funcs.R")
source("0.2 plot_funcs.R")

#### Environment 1 (yearly and quarterly) ####
sel_env <- 1
raw_data_dir_y <- paste0("sim/env ",sel_env)
raw_data_dir_q <- paste0("sim/env ",sel_env," q")
triangles_y <- load_sim_triangles(raw_data_dir_y,50,10)
triangles_q <- load_sim_triangles(raw_data_dir_q,50,40)

plot_claims_gg(triangles_y[[2]][[5]], AYs = c(2,4,6,8))
plot_claims_gg(triangles_y[[2]][[45]], AYs = c(2,4,6,8))
plot_claims_gg_q(triangles_q[[2]][[5]], AQs = c(5,15,25,35))
plot_claims_gg_q(triangles_q[[2]][[45]], AQs = c(5,15,25,35))

#### Environment 2 (only yearly) ####
sel_env <- 2
raw_data_dir_y <- paste0("sim/env ",sel_env)
triangles_y <- load_sim_triangles(raw_data_dir_y,50,10)

plot_claims_gg(triangles_y[[2]][[5]], AYs = c(2,4,6,8))
plot_claims_gg(triangles_y[[2]][[45]], AYs = c(2,4,6,8))

#### Environment 3 (only yearly) ####
sel_env <- 3
raw_data_dir_y <- paste0("sim/env ",sel_env)
triangles_y <- load_sim_triangles(raw_data_dir_y,50,10)

plot_claims_gg(triangles_y[[2]][[5]], AYs = c(2,4,6,8))
plot_claims_gg(triangles_y[[2]][[45]], AYs = c(2,4,6,8))

#### Environment 4 (only yearly) ####
sel_env <- 4
raw_data_dir_y <- paste0("sim/env ",sel_env)
triangles_y <- load_sim_triangles(raw_data_dir_y,50,10)

plot_claims_gg(triangles_y[[2]][[5]], AYs = c(2,4,6,8))
plot_claims_gg(triangles_y[[2]][[45]], AYs = c(2,4,6,8))

