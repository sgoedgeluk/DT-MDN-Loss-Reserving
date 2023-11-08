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
# Setup of simulation data environment 3

SI_inf <- 0.065+0.03**((i-1)/n)
seed=i

## MODULE 1: CLAIM OCCURANCE
# Set seed for experiment replication and set time unit and reference claim size for SynthETIC
ref_claim <- 200000
time_unit <- 1/4
set_parameters(ref_claim = ref_claim, time_unit = time_unit)

years = 10
I <- years / time_unit   # Total number of occurance periods, equal to number of quarters
E <- c(rep(60000, I))   # Effective exposure rates for each quarter
lambda <- c(rep(0.1, I)) # Expected claim frequencies per quarter

## MODULE 2: CLAIM SIZE
# Default claim occurance distribution: power normal S^0.2 ~ N(9.5, 3), left truncated at 30
S_df <- function(s) {
  # truncate and rescale
  if (s < 30) {
    return(0)
  } else {
    p_trun <- pnorm(s^0.25, 9.5, 3) - pnorm(30^0.25, 9.5, 3)
    p_rescaled <- p_trun/(1 - pnorm(30^0.25, 9.5, 3))
    return(p_rescaled)
  }
}

## MODULE 3: CLAIM NOTIFICATION DATE
# Specify the Weibull parameters (in this case using fixed parameters for the
# distribution)
notidel_param <- function(claim_size, occurrence_period) {
  # NOTE: users may add to, but not remove these two arguments (claim_size,
  # occurrence_period) as they are part of SynthETIC's internal structure
  
  # specify the target mean and target coefficient of variation
  target_mean <- 2.465258
  target_cv <- 1.535595
  # convert to Weibull parameters
  shape <- get_Weibull_parameters(target_mean, target_cv)[1]
  scale <- get_Weibull_parameters(target_mean, target_cv)[2]
  
  c(shape = shape, scale = scale)
}

## MODULE 4: CLAIM CLOSURE (settlement delay)
# Specify the Weibull parameters (in this case using fixed parameters for the
# distribution)
setldel_param <- function(claim_size, occurrence_period) {
  # NOTE: users may add to, but not remove these two arguments (claim_size,
  # occurrence_period) as they are part of SynthETIC's internal structure
  
  # specify the target Weibull mean
  target_mean <- 11.74211
  
  # specify the target Weibull coefficient of variation
  target_cv <- 0.613554
  
  c(shape = get_Weibull_parameters(target_mean, target_cv)[1, ],
    scale = get_Weibull_parameters(target_mean, target_cv)[2, ])
}

## MODULE 5: NUMBER OF PARTIAL PAYMENTS
# Default values for benchmark values
benchmark_1 <- 0.0375 * ref_claim
benchmark_2 <- 0.075 * ref_claim

# Default random generating function
rmixed_payment_no <- function(n, claim_size, claim_size_benchmark_1, claim_size_benchmark_2) {
  # construct the range indicators
  test_1 <- (claim_size_benchmark_1 < claim_size & claim_size <= claim_size_benchmark_2)
  test_2 <- (claim_size > claim_size_benchmark_2)
  
  # if claim_size <= claim_size_benchmark_1
  no_pmt <- sample(c(1, 2), size = n, replace = T, prob = c(1/2, 1/2))
  # if claim_size is between the two benchmark values
  no_pmt[test_1] <- sample(c(2, 3), size = sum(test_1), replace = T, prob = c(1/3, 2/3))
  # if claim_size > claim_size_benchmark_2
  no_pmt_mean <- pmin(8, 4 + log(claim_size/claim_size_benchmark_2))
  prob <- 1 / (no_pmt_mean - 3)
  no_pmt[test_2] <- stats::rgeom(n = sum(test_2), prob = prob[test_2]) + 4
  
  no_pmt
}

## MODULE 6: SIZES OF PARTIAL PAYMENTS (w/o allowance for inflation)
# The default function samples the sizes of partial payments conditional on the
# number of partial payments, and the size of the claim
rmixed_payment_size <- function(n, claim_size) {
  # n = number of simulations, here n should be the number of partial payments
  if (n >= 4) {
    # 1) Simulate the "complement" of the proportion of total claim size
    #    represented by the last two payments
    p_mean <- 1 - min(0.95, 0.75 + 0.04*log(claim_size/(0.10 * ref_claim)))
    p_CV <- 0.20
    p_parameters <- get_Beta_parameters(target_mean = p_mean, target_cv = p_CV)
    last_two_pmts_complement <- stats::rbeta(
      1, shape1 = p_parameters[1], shape2 = p_parameters[2])
    last_two_pmts <- 1 - last_two_pmts_complement
    
    # 2) Simulate the proportion of last_two_pmts paid in the second last payment
    q_mean <- 0.9
    q_CV <- 0.03
    q_parameters <- get_Beta_parameters(target_mean = q_mean, target_cv = q_CV)
    q <- stats::rbeta(1, shape1 = q_parameters[1], shape2 = q_parameters[2])
    
    # 3) Calculate the respective proportions of claim amount paid in the
    #    last 2 payments
    p_second_last <- q * last_two_pmts
    p_last <- (1-q) * last_two_pmts
    
    # 4) Simulate the "unnormalised" proportions of claim amount paid
    #    in the first (m - 2) payments
    p_unnorm_mean <- last_two_pmts_complement/(n - 2)
    p_unnorm_CV <- 0.10
    p_unnorm_parameters <- get_Beta_parameters(
      target_mean = p_unnorm_mean, target_cv = p_unnorm_CV)
    amt <- stats::rbeta(
      n - 2, shape1 = p_unnorm_parameters[1], shape2 = p_unnorm_parameters[2])
    
    # 5) Normalise the proportions simulated in step 4
    amt <- last_two_pmts_complement * (amt/sum(amt))
    # 6) Attach the last 2 proportions, p_second_last and p_last
    amt <- append(amt, c(p_second_last, p_last))
    # 7) Multiply by claim_size to obtain the actual payment amounts
    amt <- claim_size * amt
    
  } else if (n == 2 | n == 3) {
    p_unnorm_mean <- 1/n
    p_unnorm_CV <- 0.10
    p_unnorm_parameters <- get_Beta_parameters(
      target_mean = p_unnorm_mean, target_cv = p_unnorm_CV)
    amt <- stats::rbeta(
      n, shape1 = p_unnorm_parameters[1], shape2 = p_unnorm_parameters[2])
    # Normalise the proportions and multiply by claim_size to obtain the actual payment amounts
    amt <- claim_size * amt/sum(amt)
    
  } else {
    # when there is a single payment
    amt <- claim_size
  }
  return(amt)
}

## MODULE 7: DISTRIBUTION OF PAYMENTS OVER TIME
# The simulation of the inter-partial delays is almost identical to that of
# partial payment sizes, except that it also depends on the claim settlement delay
# - the inter-partial delays should add up to the settlement delay.
r_pmtdel <- function(n, claim_size, setldel, setldel_mean) {
  result <- c(rep(NA, n))
  
  # First simulate the unnormalised values of d, sampled from a Weibull distribution
  if (n >= 4) {
    # 1) Simulate the last payment delay
    unnorm_d_mean <- (1 / 4) / time_unit
    unnorm_d_cv <- 0.20
    parameters <- get_Weibull_parameters(target_mean = unnorm_d_mean, target_cv = unnorm_d_cv)
    result[n] <- stats::rweibull(1, shape = parameters[1], scale = parameters[2])
    
    # 2) Simulate all the other payment delays
    for (i in 1:(n - 1)) {
      unnorm_d_mean <- setldel_mean / n
      unnorm_d_cv <- 0.35
      parameters <- get_Weibull_parameters(target_mean = unnorm_d_mean, target_cv = unnorm_d_cv)
      result[i] <- stats::rweibull(1, shape = parameters[1], scale = parameters[2])
    }
    
  } else {
    for (i in 1:n) {
      unnorm_d_mean <- setldel_mean / n
      unnorm_d_cv <- 0.35
      parameters <- get_Weibull_parameters(target_mean = unnorm_d_mean, target_cv = unnorm_d_cv)
      result[i] <- stats::rweibull(1, shape = parameters[1], scale = parameters[2])
    }
  }
  
  # Normalise d such that sum(inter-partial delays) = settlement delay
  # To make sure that the pmtdels add up exactly to setldel, we treat the last one separately
  result[1:n-1] <- (setldel/sum(result)) * result[1:n-1]
  result[n] <- setldel - sum(result[1:n-1])
  
  return(result)
}

param_pmtdel <- function(claim_size, setldel, occurrence_period) {
  # mean settlement delay
  if (claim_size < (0.10 * ref_claim) & occurrence_period >= 21) {
    a <- min(0.85, 0.65 + 0.02 * (occurrence_period - 21))
  } else {
    a <- max(0.85, 1 - 0.0075 * occurrence_period)
  }
  mean_quarter <- a * min(25, max(1, 6 + 4*log(claim_size/(0.10 * ref_claim))))
  target_mean <- mean_quarter / 4 / time_unit
  
  c(claim_size = claim_size,
    setldel = setldel,
    setldel_mean = target_mean)
}

## MODULE 8: CLAIM INFLATION
# Base inflation rates
demo_rate <- (1 + 0.02)^(1/4) - 1
base_inflation_past <- rep(demo_rate, times = 40)
base_inflation_future <- rep(demo_rate, times = 40)
base_inflation_vector <- c(base_inflation_past, base_inflation_future)

# Superimposed inflation:
# 1) With respect to occurrence "time" (continuous scale)
# Set to have no superimposed inflation in occurance time
SI_occurrence <- function(occurrence_time, claim_size) {
  {1}
}

# 2) With respect to payment "time" (continuous scale)
# -> compounding by user-defined time unit
# Set to have no SI for the first 30 periods, after which there is a SI of 0.08
SI_payment <- function(payment_time, claim_size) {
  to_return = c()
  
  # Change of SI after period 30
  inf_rate = c(rep((1.0^0.25) - 1, 30), rep(((1+SI_inf)^0.25) - 1, 50))
  
  inf_vector = 1 + inf_rate
  cumInf = cumprod(inf_vector)
  
  for (i in 1:length(payment_time)){
    time = payment_time[i]
    Periodic = ifelse(floor(time) == 0, 1, cumInf[min(floor(time), length(inf_vector))])
    Residual = inf_vector[min(ceiling(time) + (time == 0), length(inf_vector))]^(time - min(length(inf_vector),floor(time)))
    to_return[i] = Periodic*Residual
  }
  return(to_return)
}