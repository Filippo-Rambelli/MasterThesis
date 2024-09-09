library(Matrix)
library(fields)
library(readr)
library(INLA)
library(rSPDE)
rm(list = ls())

options(mc.cores = 8)

run_spde_100k<-function(range=0.2,max.edge=0.1){
  if (range == 0.2) {
    full_data <- read_csv("/data/combined_data_100k_02.csv")
  } else if (range == 0.5) {
    full_data <- read.csv("/data/combined_data_100k_05.csv")
  } else if (range == 0.05) {  
    full_data <- read.csv("/data/combined_data_100k_005.csv")
  } else {stop("Invalid range value") }
  
  eff_range=range/4.74
  true_nugget=0.5
  
  nrep=20
  
  rmse_train_list<-rep(0,nrep/2); rmse_inter_list<-rep(0,nrep/2); rmse_extra_list<-rep(0,nrep/2)
  score_train_list<-rep(0,nrep/2); score_inter_list<-rep(0,nrep/2); score_extra_list<-rep(0,nrep/2)
  pred_train_times<-rep(0,nrep/2);pred_inter_times<-rep(0,nrep/2);pred_extra_times<-rep(0,nrep/2)
  fitting_times<-rep(0,nrep); nugget_list<-rep(0,nrep)
  
  for (i in 1:nrep){
    data<-full_data[full_data$rep==i,]
    train<-data[data$which=="train",]
    interp<-data[data$which=="interpolation",]
    extrap<-data[data$which=="extrapolation",]
    
    #define the mesh limits
    pl.dom <- cbind(c(0, 1, 1, 0), c(0, 0, 1, 1))
    
    coords_train <- as.matrix(train[, 1:2])
    coords_inter<-as.matrix(interp[,c(1:2)])
    coords_extra<-as.matrix(extrap[,c(1:2)])
    
    #only fit
    fitting_times[i] <- system.time({
      #construct the mesh
      mesh <- inla.mesh.2d(loc.domain = pl.dom, max.e = max.edge)
      spde <-  rspde.matern(mesh = mesh,nu=1.5)
      
      A <- inla.spde.make.A(mesh, loc = coords_train)
      stk <- inla.stack(
        data = list(resp = train$y),
        A = list(A),
        effects = list(i = 1:spde$n.spde),
        tag = 'train')
      
      #only fit this time
      res_eb<-inla(resp~0+ f(i,model=spde),
                   data=inla.stack.data(stk),
                   control.predictor=list(A=inla.stack.A(stk)),
                   control.inla = list(int.strategy = "eb"),num.threads=8)
      
    })[3]
    nugget_list[i]<-1/res_eb$summary.hyperpar[1,"0.5quant"]
    
    if(i<=nrep/2){
      #fit and predict together, but speedup such that it is mostly prediction time
      pred_inter_times[i] <- system.time({A_inter<-inla.spde.make.A(mesh,loc=coords_inter)
      stk.inter <- inla.stack(
        data = list(resp = NA),
        A = list(A_inter),
        effects = list(i = 1:spde$n.spde),
        tag = 'inter')})[3]
      
      pred_extra_times[i] <- system.time({A_extra<-inla.spde.make.A(mesh,loc=coords_extra)
      stk.extra <- inla.stack(
        data = list(resp = NA),
        A = list(A_extra),
        effects = list(i = 1:spde$n.spde),
        tag = 'extra')})
      
      #join all data stacks
      stk.full <- inla.stack(stk, stk.inter,stk.extra)
      
      #fit and prediction together, but using previous estimates to reduce time
      time_joint_pred<-system.time({ rpmu <- inla(resp ~ 0+f(i, model = spde),
                                                  data = inla.stack.data(stk.full),
                                                  control.mode = list(theta = res_eb$mode$theta, restart = FALSE),
                                                  control.predictor = list(A = inla.stack.A(
                                                    stk.full), compute = TRUE),
                                                  control.inla = list(int.strategy = "eb"),num.threads=8) })[3]
      pred_train_times[i]<-time_joint_pred
      pred_inter_times[i]<- pred_inter_times[i]+time_joint_pred
      pred_extra_times[i]<-pred_extra_times[i]+time_joint_pred
      
      #extract train preds
      train_index<-inla.stack.index(stk.full, 'train')$data
      pred_train_mean<-rpmu$summary.fitted.values[train_index, 1]
      pred_train_sd<-rpmu$summary.fitted.values[train_index, 2]
      
      #extract inter preds
      inter_index<-inla.stack.index(stk.full, 'inter')$data
      pred_inter_mean<-rpmu$summary.fitted.values[inter_index, 1]
      pred_inter_sd<-rpmu$summary.fitted.values[inter_index, 2]
      
      #extract extra preds
      extra_index<-inla.stack.index(stk.full, 'extra')$data
      pred_extra_mean<-rpmu$summary.fitted.values[extra_index, 1]
      pred_extra_sd<-rpmu$summary.fitted.values[extra_index, 2]
      
      #RMSE
      rmse_train_list[i]<-sqrt(mean((train$f - pred_train_mean)^2))
      rmse_inter_list[i]<-sqrt(mean((interp$f - pred_inter_mean)^2))
      rmse_extra_list[i]<-sqrt(mean((extrap$f - pred_extra_mean)^2))
      
      #log_score
      score_train_list[i]<-mean( (0.5*(pred_train_mean-train$f )^2)/pred_train_sd^2 + 
                                   0.5*log(2*pi*pred_train_sd^2) )
      score_inter_list[i]<-mean( (0.5*(pred_inter_mean-interp$f)^2)/pred_inter_sd^2 + 
                                   0.5*log(2*pi*pred_inter_sd^2) )
      score_extra_list[i]<-mean( (0.5*(pred_extra_mean-extrap$f)^2)/pred_extra_sd^2 + 
                                   0.5*log(2*pi*pred_extra_sd^2) )
      
      rm(A_inter,A_extra,stk.full,stk.inter,stk.extra,rpmu,pred_train_mean,pred_train_sd,
         pred_inter_mean,pred_inter_sd,pred_extra_mean,pred_extra_sd)
    }
    print(paste0("rep ",i," done"))
    rm(data,train,interp,extrap,mesh,spde,A,stk,res_eb,coords_train,coords_inter,coords_extra)
  }
  
  # Create the filename
  filename <- paste0("spde_100k_",range,"_",max.edge)
  
  # Open the file for writing
  file_path <- paste0(filename, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("spde ", max.edge),
    paste0("True range: ",range/4.74 ),
    
    paste0("bias for GP range: "),
    paste0("MSE for GP range: "),
    paste0("bias for GP variance: "),
    paste0("MSE for GP variance: "),
    paste0("bias for error term variance: ", mean(nugget_list - true_nugget)),
    paste0("MSE for error term variance: ", mean((nugget_list - true_nugget)^2)),
    paste0("variance for bias of GP range: "),
    paste0("variance for bias GP of variance: "),
    paste0("variance for bias error of term variance: ", var(nugget_list)/nrep),
    paste0("variance for MSE GP range: "),
    paste0("variance for MSE GP variance: "),
    paste0("variance for MSE error term variance: ", var((nugget_list - true_nugget)^2)/nrep),
    paste0("mean time for parameter estimation: ", mean(fitting_times)),
    paste0("mean estimated negloglik true pars: "),
    paste0("mean estimated negloglik fake pars: ",),
    paste0("mean time for true loglik evaluation: "),
    paste0("mean time for fake loglik evaluation: "),
    paste0("variance for negloglik true pars: "),
    paste0("variance for negloglik fake pars: "),
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
    paste0("variance for RMSE extra: ", var(rmse_extra_list)*2/nrep)
  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}

#example usage
run_spde_100k(range=0.05,max.edge = 0.02)
