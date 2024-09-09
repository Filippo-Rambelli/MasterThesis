library(GPvecchia)
library(Matrix)
library(fields)
library(readr)

options(mc.cores = 8)
#we manually define a function to optimize all the covariance parameters but the
#smoothness. This is achiever by modyfing the vecchia_estimate function in the
#package GPVecchia
vecchia_estimate_fixed_smoothness_1.5 <- function(data, locs, X, m = 20, 
                                                  covmodel = 'matern', theta.ini, 
                                                  output.level = 1,
                                                  reltol = sqrt(.Machine$double.eps), ...) {
  
  ## default trend is constant over space (intercept)
  if (missing(X)) {
    beta.hat <- mean(data)
    z <- data - beta.hat
    trend <- 'constant'
  } else if (is.null(X)) {
    ## if X=NULL, do not estimate any trend
    beta.hat <- c()
    z <- data
    trend <- 'none'
  } else {
    ## otherwise, estimate and de-trend
    beta.hat <- Matrix::solve(crossprod(X), crossprod(X, data))
    z <- data - X %*% beta.hat
    trend <- 'userspecified'
  }
  
  ## specify vecchia approximation
  vecchia.approx <- vecchia_specify(locs, m, ...)
  
  ## initial covariance parameter values
  if (all(is.character(covmodel)) && covmodel == 'matern') {
    if (missing(theta.ini) || any(is.na(theta.ini))) {
      var.res <- stats::var(z)
      n <- length(z)
      dists.sample <- fields::rdist(locs[sample(1:n, min(n, 300)), ])
      theta.ini <- c(.9 * var.res, mean(dists.sample) / 4, .1 * var.res) # var, range, nugget
    }
  }
  
  ## specify vecchia loglikelihood
  negloglik.vecchia <- function(lgparms) {
    full_params <- exp(lgparms)
    full_params <- c(full_params[1:2], 1.5, full_params[3]) # Insert the fixed smoothness value
    l <- -vecchia_likelihood(z, vecchia.approx, full_params[-length(full_params)], 
                             full_params[length(full_params)], covmodel = covmodel)
    return(l)
  }
  
  ## find MLE of theta (given beta.hat)
  non1pars <- which(theta.ini != 1)
  parscale <- rep(1, length(theta.ini))
  parscale[non1pars] <- log(theta.ini[non1pars])
  
  opt.result <- stats::optim(par = log(theta.ini),
                             fn = negloglik.vecchia,
                             method = "Nelder-Mead",
                             control = list(
                               trace = 100, maxit = 300, parscale = parscale,
                               reltol = reltol
                             )) # trace=1 outputs iteration counts
  
  theta.hat <- exp(opt.result$par)
  theta.hat <- c(theta.hat[1:2], 1.5, theta.hat[3]) # Fix smoothness to 1.5
  names(theta.hat) <- c("variance", "range", "smoothness", "nugget")
  
  ## return estimated parameters
  if (output.level > 0) {
    cat('estimated trend coefficients:\n')
    print(beta.hat)
    cat('estimated covariance parameters:\n')
    print(theta.hat)
  }
  return(list(z = z, beta.hat = beta.hat, theta.hat = theta.hat,
              trend = trend, locs = locs, covmodel = covmodel))
}

#we define a function that will be called for every tuning parameter
run_mra<-function(range=0.2,r=2){
  set.seed(r)
  if (range == 0.2) {
    full_data <- read_csv("/data/combined_data_100k_02.csv")
  } else if (range == 0.5) {
    full_data <- read.csv("/data/combined_data_100k_05.csv")
  } else if (range == 0.05) {  
    full_data <- read.csv("/data/combined_data_100k_005.csv")
  } else {stop("Invalid range value") }
  
  eff_range=range/4.74
  nugget=0.5
  marginal_var=1
  nu=1.5
  
  nrep=20
  
  rmse_train_list<-rep(0,nrep/2); rmse_inter_list<-rep(0,nrep/2); rmse_extra_list<-rep(0,nrep/2)
  score_train_list<-rep(0,nrep/2); score_inter_list<-rep(0,nrep/2); score_extra_list<-rep(0,nrep/2)
  pred_train_times<-rep(0,nrep/2);pred_inter_times<-rep(0,nrep/2);pred_extra_times<-rep(0,nrep/2)
  fitting_times<-rep(0,nrep)
  variance_list<-rep(0,nrep); range_list<-rep(0,nrep); nugget_list<-rep(0,nrep)
  true_negloglik_list<-rep(0,nrep); fake_negloglik_list<-rep(NA,nrep)
  true_negloglik_eval_times<-rep(0,nrep); fake_negloglik_eval_times<-rep(NA,nrep)
  
  for (i in 1:nrep){
    data<-full_data[full_data$rep==i,]
    train<-data[data$which=="train",]
    interp<-data[data$which=="interpolation",]
    extrap<-data[data$which=="extrapolation",]
    
    mra.options.mra = list(r = c(r), J = 2)
    
    # Fit the model
    fitting_times[i] <- system.time({
      vecchia.est <- vecchia_estimate_fixed_smoothness_1.5(
        data = train$y,
        locs = as.matrix(train[,1:2]),
        X = NULL,conditioning="mra", mra.options = mra.options.mra,reltol = 1e-6)
      
      vecchia.approx <- vecchia_specify(
        as.matrix(train[, c(1:2)]), 
        mra.options = mra.options.mra, conditioning = 'mra', verbose = TRUE)})[3]
    
    variance_list[i]<-vecchia.est$theta.hat[1]
    range_list[i]<-vecchia.est$theta.hat[2]
    nugget_list[i]<-vecchia.est$theta.hat[4]
    
    #negloglik evaluation
    tryCatch({
      true_negloglik_eval_times[i] <- system.time({
        true_negloglik_list[[i]] <- -vecchia_likelihood(train$y, vecchia.approx, 
                                                        c(marginal_var, eff_range, nu), nugget)
      })[3]
    }, error = function(e) {
      message("Error at iteration ", i, ": ", e$message)
      true_negloglik_eval_times[i] <- NA  
      true_negloglik_list[[i]] <- NA  
    })
    
    tryCatch({
      fake_negloglik_eval_times[i] <- system.time({
        fake_negloglik_list[[i]] <- -vecchia_likelihood(train$y, vecchia.approx,  
                                                        c(2 *marginal_var, 2 *eff_range, nu), 2 * nugget)
      })[3]
    }, error = function(e) {
      message("Error at iteration ", i, ": ", e$message)
      fake_negloglik_eval_times[i] <- NA  
      fake_negloglik_list[[i]] <- NA  
    })
    
    print("outside")
    print(i)
    
    if(i<=nrep/2){
      # Predict on the training set
      pred_train_times[i] <- system.time({
        pred_train <- vecchia_prediction(
          train$y, vecchia.approx, c(marginal_var, eff_range, nu), nugget
        )
      })[3]
      # Predict on interpolation set
      pred_inter_times[i] <- system.time({
        vecchia.approx <- vecchia_specify(as.matrix(train[, c(1:2)]), 
                                          mra.options = mra.options.mra,
                                          conditioning = 'mra', verbose = TRUE,
                                          locs.pred = as.matrix(interp[, c(1:2)]))
        pred_inter <- vecchia_prediction(train$y, vecchia.approx, 
                                         c(marginal_var, eff_range, nu), nugget)})[3]
      
      # Predict on extrapolation set
      pred_extra_times[i] <- system.time({
        vecchia.approx <- vecchia_specify(as.matrix(train[, c(1:2)]), 
                                          mra.options = mra.options.mra,conditioning = 'mra',
                                          verbose = TRUE,locs.pred = as.matrix(extrap[, c(1:2)]))
        pred_extra <- vecchia_prediction(train$y, vecchia.approx, 
                                         c(marginal_var, eff_range, nu), nugget)})[3]
      
      #RMSE
      rmse_train_list[i]<-sqrt(mean((train$f - pred_train$mu.obs)^2))
      rmse_inter_list[i]<-sqrt(mean((interp$f - pred_inter$mu.pred)^2))
      rmse_extra_list[i]<-sqrt(mean((extrap$f - pred_extra$mu.pred)^2))
      
      #log_score
      score_train_list[i]<-mean( (0.5*(pred_train$mu.obs-train$f )^2)/pred_train$var.obs +
                                   0.5*log(2*pi*pred_train$var.obs) )
      score_inter_list[i]<-mean( (0.5*(pred_inter$mu.pred-interp$f)^2)/pred_inter$var.pred +
                                   0.5*log(2*pi*pred_inter$var.pred) )
      score_extra_list[i]<-mean( (0.5*(pred_extra$mu.pred-extrap$f)^2)/pred_extra$var.pred +
                                   0.5*log(2*pi*pred_extra$var.pred) )
      
      rm(pred_inter,pred_train,pred_extra)
    }
    rm(data,train,interp,extrap,vecchia.approx)
  }
  
  
  # Create the filename
  filename <- paste0("mra_100k_",range,"_",r)
  
  # Open the file for writing
  file_path <- paste0(filename, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("mra ", r),
    paste0("True range: ",range/4.74 ),
    
    paste0("bias for GP range: ", mean(range_list - eff_range)),
    paste0("MSE for GP range: ", mean((range_list - eff_range)^2)),
    paste0("bias for GP variance: ", mean(variance_list - marginal_var)),
    paste0("MSE for GP variance: ", mean((variance_list - marginal_var)^2)),
    paste0("bias for error term variance: ", mean(nugget_list - nugget)),
    paste0("MSE for error term variance: ", mean((nugget_list - nugget)^2)),
    paste0("variance for bias of GP range: ", var(range_list)/nrep),
    paste0("variance for bias GP of variance: ", var(variance_list)/nrep),
    paste0("variance for bias error of term variance: ", var(nugget_list)/nrep),
    paste0("variance for MSE GP range: ", var((range_list - eff_range)^2)/nrep),
    paste0("variance for MSE GP variance: ", var((variance_list - marginal_var)^2)/nrep),
    paste0("variance for MSE error term variance: ", var((nugget_list - nugget)^2)/nrep),
    paste0("mean time for parameter estimation: ", mean(fitting_times)),
    paste0("mean estimated negloglik true pars: ", mean(true_negloglik_list, na.rm = TRUE)),
    paste0("mean estimated negloglik fake pars: ", mean(fake_negloglik_list, na.rm = TRUE)),
    paste0("mean time for true loglik evaluation: ", mean(true_negloglik_eval_times, na.rm = TRUE)),
    paste0("mean time for fake loglik evaluation: ", mean(fake_negloglik_eval_times, na.rm = TRUE)),
    paste0("variance for negloglik true pars: ", var(true_negloglik_list, na.rm = TRUE)/nrep),
    paste0("variance for negloglik fake pars: ", var(fake_negloglik_list, na.rm = TRUE)/nrep),
    paste0("mean univariate score train: ", mean(score_train_list)),
    paste0("mean univariate score interpolation: ", mean(score_inter_list)),
    paste0("mean univariate score extrapolation: ", mean(score_extra_list)),
    paste0("variance univariate score train: ", var(score_train_list)*2/nrep),
    paste0("variance univariate score interpolation: ", var(score_inter_list)*2/nrep),
    paste0("variance univariate score extrapolation: ", var(score_extra_list)*2/nrep),
    paste0("mean time for train univariate prediction: ", mean(pred_train_times)),
    paste0("mean time for interpolation univariate prediction: ", mean(pred_inter_times)),
    paste0("mean time for extrapolation univariate prediction: ", mean(pred_extra_times)),
    paste0("RMSE train: ", mean(rmse_train_list)),
    paste0("RMSE inter: ", mean(rmse_inter_list)),
    paste0("RMSE extra: ", mean(rmse_extra_list)),
    paste0("variance for RMSE train: ", var(rmse_train_list)*2/nrep),
    paste0("variance for RMSE inter: ", var(rmse_inter_list)*2/nrep),
    paste0("variance for RMSE extra: ", var(rmse_extra_list)*2/nrep),
    
    #if we have some missing log-likelihood, we have to adjust the variances
    #for this reason we also save the number of effective entries
    paste0("not missing true: ",sum(!is.na(true_negloglik_list))),
    paste0("not missing fake: ",sum(!is.na(fake_negloglik_list)))
    
    
  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}

#example usage
run_mra(range=0.05,r=5)