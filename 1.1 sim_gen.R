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
# Simulates the loss triangles, both quarterly and yearly, for the four 
# simulated data environments.

# Import necessary packages and support functions
library(SynthETIC)
library(data.table)

source("0.1 support_funcs.R")
source("1.0 sim_funcs.R")

#### Environment 1 ####
# For environment 1, 200 triangles are simulated  to also test for the effects of 
# the amount of data on forecasting performance
n=200
set.seed(-1)

{
  # Set necessary settings and output directories
  env = 1
  runoff = 1
  triangle_directory = paste0("sim/env ",env)
  triangle_directory_q  = paste0("sim/env ",env," q")
  # Create n triangles
  for(i in 1:n) {
    print(i)
    source('env/env_1.R')
    sim_data_env1 <- GenData(seed)
    triangle_dat_env1_q <- gen_loss_triangle_q(sim_data_env1, runoff)
    # Correctly structure output filename
    if(i<10) {
      file_name_q <- paste0(triangle_directory_q,"/sim_00",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_00",as.character(i),".csv")
    } else if(i>=10 & i < 100) {
      file_name_q <- paste0(triangle_directory_q,"/sim_0",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_0",as.character(i),".csv")
    } else {
      file_name_q <- paste0(triangle_directory_q,"/sim_",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_",as.character(i),".csv")
    }
    
    # Convert the simulated quarterly triangle to a yearly triangle and save both
    try <- trq_to_try(triangle_dat_env1_q)
    write.csv(try, file_name, row.names=FALSE)
    write.csv(triangle_dat_env1_q, file_name_q, row.names=FALSE)
  }
}


#### Environment 2 ####
n=50
set.seed(-1)

{
  env = 2
  runoff = 0
  triangle_directory = paste0("sim/env ",env," new")
  triangle_directory_q  = paste0("sim/env ",env," q")
  for(i in 1:n) {
    print(i)
    
    # Given the need for a shift from short to long tails, these are simulated separately
    # and aggregated after simulation.
    short = 1
    source('env/env_2.R')
    sim_data_env2s <- GenData(seed)
    triangle_dat_env2s <- gen_loss_triangle_q(sim_data_env2s, runoff)
    
    short = 0
    source('env/env_2.R')
    sim_data_env2l <- GenData(seed)
    triangle_dat_env2l <- gen_loss_triangle_q(sim_data_env2l, runoff)
    
    triangle_env2f = triangle_dat_env2s + triangle_dat_env2l
    triangle_env2f$AP = triangle_env2f$AP/2
    if(i<10) {
      file_name_q <- paste0(triangle_directory_q,"/sim_00",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_00",as.character(i),".csv")
    } else if(i>=10 & i < 100) {
      file_name_q <- paste0(triangle_directory_q,"/sim_0",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_0",as.character(i),".csv")
    } else {
      file_name_q <- paste0(triangle_directory_q,"/sim_",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_",as.character(i),".csv")
    }
    try <- trq_to_try(triangle_env2f)
    write.csv(try, file_name, row.names=FALSE)
    write.csv(triangle_env2f, file_name_q, row.names=FALSE)
  }
}

#### Environment 3 ####
n=50
set.seed(-1)

{
  env = 3
  runoff = 0
  triangle_directory = paste0("sim/env ",env," new")
  triangle_directory_q  = paste0("sim/env ",env," q")
  for(i in 1:n) {
    print(i)
    source('env/env_3.R')
    sim_data_env3 <- GenData(seed)
    triangle_dat_env3 <- gen_loss_triangle_q(sim_data_env3, runoff)
    if(i<10) {
      file_name_q <- paste0(triangle_directory_q,"/sim_00",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_00",as.character(i),".csv")
    } else if(i>=10 & i < 100) {
      file_name_q <- paste0(triangle_directory_q,"/sim_0",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_0",as.character(i),".csv")
    } else {
      file_name_q <- paste0(triangle_directory_q,"/sim_",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_",as.character(i),".csv")
    }
    try <- trq_to_try(triangle_dat_env3)
    write.csv(try, file_name, row.names=FALSE)
    write.csv(triangle_dat_env3, file_name_q, row.names=FALSE)
  }
}

#### Environment 4 ####
n=50
set.seed(-1)

{
  env = 4
  runoff = 0
  triangle_directory = paste0("sim/env ",env)
  triangle_directory_q  = paste0("sim/env ",env," q")
  for(i in 1:n) {
    print(i)
    source('env/env_4.R')
    sim_data_env4 <- GenData(seed)
    triangle_dat_env4 <- gen_loss_triangle_q(sim_data_env4, runoff)
    if(i<10) {
      file_name_q <- paste0(triangle_directory_q,"/sim_00",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_00",as.character(i),".csv")
    } else if(i>=10 & i < 100) {
      file_name_q <- paste0(triangle_directory_q,"/sim_0",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_0",as.character(i),".csv")
    } else {
      file_name_q <- paste0(triangle_directory_q,"/sim_",as.character(i),"_q.csv")
      file_name <- paste0(triangle_directory,"/sim_",as.character(i),".csv")
    }
    try <- trq_to_try(triangle_dat_env4)
    write.csv(try, file_name, row.names=FALSE)
    write.csv(triangle_dat_env4, file_name_q, row.names=FALSE)
  }
}

