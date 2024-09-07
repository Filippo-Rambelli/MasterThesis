#This code rely on a implementation from Fabio Sigrist, December 2023

#install.packages("RandomFields")
library(RandomFields)
#install.packages("remotes")
library(remotes)
#install_version("RandomFieldsUtils", "1.2.5")
#install_version("RandomFields", "3.3.14")
#Simulating data random locations n=10k
rm(list=ls())



#general function and global parameters----
sim_gp_given_coords<-function(coords,rho=0.1,sigma2=1,nu=1.5,sigma2_error=0.5,n){
  if (rho == 0) {
    iid_no_GP <- TRUE
  } else {
    iid_no_GP <- FALSE
  }
  if (iid_no_GP) {
    eps <- rnorm(n,sd = sqrt(sigma2))
  } else {
    if (nu == 0.5) {
      RFmodel <- RMexp(var=sigma2, scale=rho)
    } else if (nu > 1e3) {
      RFmodel <- RMgauss(var=sigma2, scale=rho)
    } else {
      RFmodel <- RMmatern(var=sigma2, scale=rho, nu=nu)
    }
    sim <- RFsimulate(RFmodel, x=coords) # ignore warning
    eps <- sim$variable1
  }
  y <- eps + rnorm(n, sd=sqrt(sigma2_error))
  yeps<- cbind(y,eps)
  return(yeps)
}

sigma2 <- 1 # marginal variance
nu <- 1.5 # smoothness
sigma2_error <- 0.5 # variance of error (=nugget)
list_05<-list_02<-list_005<-list()


#n=10'000, range 0.05/4.74----
rho <- 0.05/4.74 # range
n<-10000
reps<-100
set.seed(500)
for(j in 1:reps){
  coords_train <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_train)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_train <- rbind(coords_train,coord_i)
    }
  }
  coords_inter <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_inter)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_inter <- rbind(coords_inter,coord_i)
    }
  }
  
  coords_extra <- matrix(1-runif(n*2)/2, ncol=2)
  
  coords <- rbind(coords_train, coords_inter, coords_extra)
  
  yeps<-sim_gp_given_coords(coords,n=3*n,rho=rho)
  which<-c(rep("train",n),rep("interpolation",n),rep("extrapolation",n))
  
  full_data<-cbind(coords,yeps,which,j)
  colnames(full_data)<-c("x1","x2","y","f","which","rep")
  list_005[[j]]<-full_data
}
combined_data_005 <- do.call(rbind, list_005)
write.csv(combined_data_005, file = "combined_data_005.csv", row.names = FALSE)


#n=10'000, range 0.2/4.74----
rho <- 0.2/4.74 # range
n<-10000
reps<-100
set.seed(20)
for(j in 1:reps){
  coords_train <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_train)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_train <- rbind(coords_train,coord_i)
    }
  }
  
  coords_inter <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_inter)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_inter <- rbind(coords_inter,coord_i)
    }
  }
  
  coords_extra <- matrix(1-runif(n*2)/2, ncol=2)
  
  coords <- rbind(coords_train, coords_inter, coords_extra)
  
  yeps<-sim_gp_given_coords(coords,n=3*n,rho=rho)
  which<-c(rep("train",n),rep("interpolation",n),rep("extrapolation",n))
  
  full_data<-cbind(coords,yeps,which,j)
  colnames(full_data)<-c("x1","x2","y","f","which","rep")
  list_02[[j]]<-full_data
}
combined_data_02 <- do.call(rbind, list_02)
write.csv(combined_data_02, file = "combined_data_02.csv", row.names = FALSE)


#n=10'000, range 0.5/4.74----
rho <- 0.5/4.74 # range
n<-10000
reps<-100
set.seed(50)
for(j in 1:reps){
  coords_train <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_train)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_train <- rbind(coords_train,coord_i)
    }
  }
  coords_inter <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_inter)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_inter <- rbind(coords_inter,coord_i)
    }
  }
  
  coords_extra <- matrix(1-runif(n*2)/2, ncol=2)
  
  coords <- rbind(coords_train, coords_inter, coords_extra)
  
  yeps<-sim_gp_given_coords(coords,n=3*n,rho=rho)
  which<-c(rep("train",n),rep("interpolation",n),rep("extrapolation",n))
  
  full_data<-cbind(coords,yeps,which,j)
  colnames(full_data)<-c("x1","x2","y","f","which","rep")
  list_05[[j]]<-full_data
}
combined_data_05 <- do.call(rbind, list_05)
write.csv(combined_data_05, file = "combined_data_05.csv", row.names = FALSE)


#n=100'000, range 0.05/4.74----
rho <- 0.05/4.74 # range
n<-100000
reps<-20
set.seed(501)
for(j in 1:reps){
  coords_train <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_train)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_train <- rbind(coords_train,coord_i)
    }
  }
  coords_inter <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_inter)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_inter <- rbind(coords_inter,coord_i)
    }
  }
  
  coords_extra <- matrix(1-runif(n*2)/2, ncol=2)
  
  coords <- rbind(coords_train, coords_inter, coords_extra)
  
  yeps<-sim_gp_given_coords(coords,n=3*n,rho=rho)
  which<-c(rep("train",n),rep("interpolation",n),rep("extrapolation",n))
  
  full_data<-cbind(coords,yeps,which,j)
  colnames(full_data)<-c("x1","x2","y","f","which","rep")
  list_005[[j]]<-full_data
}
combined_data_005 <- do.call(rbind, list_005)
write.csv(combined_data_005, file = "combined_data_100k_005.csv", row.names = FALSE)


#n=100'000, range 0.2/4.74----
rho <- 0.2/4.74 # range
n<-100000
reps<-20
set.seed(21)
for(j in 1:reps){
  coords_train <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_train)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_train <- rbind(coords_train,coord_i)
    }
  }
  
  coords_inter <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_inter)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_inter <- rbind(coords_inter,coord_i)
    }
  }
  
  coords_extra <- matrix(1-runif(n*2)/2, ncol=2)
  
  coords <- rbind(coords_train, coords_inter, coords_extra)
  
  yeps<-sim_gp_given_coords(coords,n=3*n,rho=rho)
  which<-c(rep("train",n),rep("interpolation",n),rep("extrapolation",n))
  
  full_data<-cbind(coords,yeps,which,j)
  colnames(full_data)<-c("x1","x2","y","f","which","rep")
  list_02[[j]]<-full_data
}
combined_data_02 <- do.call(rbind, list_02)
write.csv(combined_data_02, file = "combined_data_100k_02.csv", row.names = FALSE)


#n=100'000, range 0.5/4.74----
rho <- 0.5/4.74 # range
n<-100000
reps<-20
set.seed(51)
for(j in 1:reps){
  coords_train <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_train)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_train <- rbind(coords_train,coord_i)
    }
  }
  coords_inter <- matrix(runif(2)/2,ncol=2)
  # exclude upper right corner
  while (dim(coords_inter)[1]<n) {
    coord_i <- runif(2) 
    if (!(coord_i[1]>=0.5 & coord_i[2]>=0.5)) {
      coords_inter <- rbind(coords_inter,coord_i)
    }
  }
  
  coords_extra <- matrix(1-runif(n*2)/2, ncol=2)
  
  coords <- rbind(coords_train, coords_inter, coords_extra)
  
  yeps<-sim_gp_given_coords(coords,n=3*n,rho=rho)
  which<-c(rep("train",n),rep("interpolation",n),rep("extrapolation",n))
  
  full_data<-cbind(coords,yeps,which,j)
  colnames(full_data)<-c("x1","x2","y","f","which","rep")
  list_05[[j]]<-full_data
}
combined_data_05 <- do.call(rbind, list_05)
write.csv(combined_data_05, file = "combined_data_100k_05.csv", row.names = FALSE)
