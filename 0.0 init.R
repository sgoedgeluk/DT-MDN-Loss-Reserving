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
# Abstract: The importance of loss reserving in insurance is well-established and 
# the introduction of machine learning into this field has opened many doors into new 
# estimation approaches. One machine learning technique, mixture density networks, 
# widens the use-cases for loss reserving models in the pipeline of actuarial science, 
# as it allows for the estimation of densities where traditional neural networks only 
# offer point estimations. This paper introduces a new neural network architecture that 
# combines the use of this type of neural network with a complex deep learning network, 
# utilizing recurrent units in combination with historical input to better estimate 
# complex patterns in loss triangles. By furthermore incorporating a company identifier 
# into the model, it is capable of using multiple triangles in training without the loss 
# of specificity. In simple loss patterns, this model closely matches the performance of 
# traditional cross-classified over-dispersed Poisson models, but shines in comparison 
# on more complex loss patterns. In addition, it proves useful on a range of cases, 
# from more granular environments with quarterly data or a high number of loss triangles 
# to cases with very sparsely available data. As such, the model stacks up well against 
# existing approaches. As its density and quantile predictions can facilitate further 
# improvements across the business, this type of architecture can be preferential over 
# models with point estimations.        

# File Description:
# Configures the environment and initializes the necessary packages for use in this study.

# install.packages("abind")
# install.packages("SynthETIC")
# install.packages('tidymodels')
# install.packages("str2str")
# install.packages("devtools")
# install.packages("keras")
# install.packages("tensorflow")
# install.packages("tidyverse")
# install.packages("moments")

# devtools::install_github("kasaai/deeptriangle")
# devtools::install_github("kasaai/insurance")

# Library for models
library(recipes)
library(insurance)
library(tidyverse)
library(deeptriangle)
library(keras)
library(tensorflow)
library(str2str)
library(reticulate)
library(abind)
library(dplyr)
library(scoringRules)
# library(moments)

# Library for simulation
library(SynthETIC)
library(data.table)

# Library for plotting and tables
library(ggplot2)
library(jtools)
library(gridExtra)
library(stargazer)

# Set working directory to current folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Install correct version of Tensorflow
# install_tensorflow() #Version used for DT-MND and MDN models
# install_tensorflow(version="1.15",python_version = "3.6") #Version used for DT and DT2 models

