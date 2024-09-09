import pandas as pd
import gpboost as gpb
import numpy as np
import math
from scipy.stats import norm
import time
import torch

def crps_gaussian(mu, sigma, x):
    """Compute the CRPS for a Gaussian predictive distribution."""
    standardized = (x - mu) / sigma
    return sigma * (standardized * (2 * norm.cdf(standardized) - 1) + 2 * norm.pdf(standardized) - 1 / np.sqrt(np.pi))

def compute_kl(var1,var2,mean1,mean2):
    kl = torch.log(torch.sqrt(var2)/torch.sqrt(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
    return kl.sum()

#load datasets
train_set = pd.read_csv("/data/house_train.csv")
interpolation_set = pd.read_csv("/data/house_interpolation.csv")
extrapolation_set = pd.read_csv("/data/house_extrapolation.csv")

#we center the data to match the 0 mean assumption
coords_train = train_set[['long', 'lat']].values 
y_train = train_set['log_price'].values - train_set['log_price'].mean()
coords_interpolation = interpolation_set[['long', 'lat']].values
y_interpolation = interpolation_set['log_price'].values -train_set['log_price'].mean()
coords_extrapolation = extrapolation_set[['long', 'lat']].values 
y_extrapolation = extrapolation_set['log_price'].values - train_set['log_price'].mean()

#load results from exact calculations to perform a comparison

exact_pred_mean_train= np.loadtxt("/data/pred_mean_train.txt")
exact_pred_var_train = np.loadtxt("/data/pred_var_train.txt")
exact_pred_mean_inter = np.loadtxt("/data/pred_mean_inter.txt")
exact_pred_var_inter = np.loadtxt("/data/pred_var_inter.txt")
exact_pred_mean_extra = np.loadtxt("/data/pred_mean_extra.txt")
exact_pred_var_extra = np.loadtxt("/data/pred_var_extra.txt")

#covariance parameters estimated via exact computations
truth = np.array([0.0787152849929405, 0.48967778700797876 ,618.2155717795291])

#define a function for the Vecchia approximation that can be called for each tuning parameter
def vecchia_run(num_neighbors):
    #fitting
    gp_model = gpb.GPModel(gp_coords=coords_train,  cov_function="matern",cov_fct_shape=1.5,
                likelihood="gaussian", gp_approx="vecchia",num_neighbors=num_neighbors)
    start_time = time.time()
    gp_model.fit(y=y_train)
    end_time = time.time()
    fitting_time = end_time - start_time

    #likelihood evaluation
    start_time= time.time()
    true_negloglik = gp_model.neg_log_likelihood(cov_pars=truth,y=y_train)
    end_time = time.time()
    true_loglik_eval_time = end_time - start_time

    start_time= time.time()
    fake_negloglik = gp_model.neg_log_likelihood(cov_pars=truth*2,y=y_train)
    end_time = time.time()
    fake_loglik_eval_time = end_time - start_time
    
    #train
    start_time= time.time() 
    pred_train = gp_model.predict(gp_coords_pred=coords_train,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_train = end_time - start_time
    rmse_train = math.sqrt(np.mean((pred_train['mu'] - y_train)**2))
    pred_mean_train= pred_train['mu']
    pred_var_train = pred_train['var']
    score_train = np.mean((0.5*(pred_mean_train - y_train)**2)/pred_var_train + 0.5*np.log(2*np.pi*pred_var_train))
    rmse_mean_train_comparison = math.sqrt(np.mean((pred_mean_train - exact_pred_mean_train)**2))
    rmse_var_train_comparison = math.sqrt(np.mean((pred_var_train - exact_pred_var_train)**2))
    kl_train = compute_kl(torch.from_numpy(exact_pred_var_train),torch.from_numpy(pred_var_train),torch.from_numpy(exact_pred_mean_train),torch.from_numpy(pred_mean_train))
    crps_train = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_train, np.sqrt(pred_var_train), y_train)])

    #interpolation
    start_time= time.time()
    pred_inter = gp_model.predict(gp_coords_pred= coords_interpolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_inter = end_time - start_time
    rmse_inter = math.sqrt(np.mean((pred_inter['mu'] - y_interpolation)**2))
    pred_mean_inter = pred_inter['mu']
    pred_var_inter = pred_inter['var']
    score_inter = np.mean((0.5*(pred_mean_inter - y_interpolation)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
    rmse_mean_inter_comparison = math.sqrt(np.mean((pred_mean_inter - exact_pred_mean_inter)**2))
    rmse_var_inter_comparison = math.sqrt(np.mean((pred_var_inter - exact_pred_var_inter)**2))
    kl_inter = compute_kl(torch.from_numpy(exact_pred_var_inter),torch.from_numpy(pred_var_inter),torch.from_numpy(exact_pred_mean_inter),torch.from_numpy(pred_mean_inter))
    crps_inter = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_inter, np.sqrt(pred_var_inter), y_interpolation)])

    #extrapolation
    start_time= time.time()
    pred_extra = gp_model.predict(gp_coords_pred= coords_extrapolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_extra = end_time - start_time
    rmse_extra = math.sqrt(np.mean((pred_extra['mu'] - y_extrapolation)**2))
    pred_mean_extra = pred_extra['mu']
    pred_var_extra = pred_extra['var']
    score_extra = np.mean((0.5*(pred_mean_extra - y_extrapolation)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
    rmse_mean_extra_comparison = math.sqrt(np.mean((pred_mean_extra - exact_pred_mean_extra)**2))
    rmse_var_extra_comparison = math.sqrt(np.mean((pred_var_extra - exact_pred_var_extra)**2))
    kl_extra = compute_kl(torch.from_numpy(exact_pred_var_extra),torch.from_numpy(pred_var_extra),torch.from_numpy(exact_pred_mean_extra),torch.from_numpy(pred_mean_extra))
    crps_extra = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_extra, np.sqrt(pred_var_extra), y_extrapolation)])

    #saving results
    filename = "house_vecchia_" + str(num_neighbors)
    with open(filename + '.txt', 'w') as file:
        file.write('Vecchia ' + str(num_neighbors)  + '\n')

        file.write('time for fitting: ' + str(fitting_time) + '\n')
        file.write('univariate score train: ' + str(score_train) + '\n')
        file.write('univariate score interpolation: ' + str(score_inter) + '\n')
        file.write('univariate score extrapolation: ' + str(score_extra) + '\n')
        file.write('time for train univariate prediction: ' + str(pred_time_train) + '\n')
        file.write('time for interpolation univariate prediction: ' + str(pred_time_inter) + '\n')
        file.write('time for extrapolation univariate prediction: ' + str(pred_time_extra) + '\n')
        file.write('rmse train: ' + str(rmse_train) + '\n')
        file.write('rmse interpolation: ' + str(rmse_inter) + '\n')
        file.write('rmse extrapolation: ' + str(rmse_extra) + '\n')
        file.write('rmse mean train: ' + str(rmse_mean_train_comparison) + '\n')
        file.write('rmse mean interpolation: ' + str(rmse_mean_inter_comparison) + '\n')
        file.write('rmse mean extrapolation: ' + str(rmse_mean_extra_comparison) + '\n')
        file.write('rmse var train: ' + str(rmse_var_train_comparison) + '\n')
        file.write('rmse var interpolation: ' + str(rmse_var_inter_comparison) + '\n')
        file.write('rmse var extrapolation: ' + str(rmse_var_extra_comparison) + '\n')
        file.write('kl train: ' + str(kl_train.item()) + '\n')
        file.write('kl interpolation: ' + str(kl_inter.item()) + '\n')
        file.write('kl extrapolation: ' + str(kl_extra.item()) + '\n')
        file.write('crps train: ' + str(crps_train) + '\n')
        file.write('crps interpolation: ' + str(crps_inter) + '\n')
        file.write('crps extrapolation: ' + str(crps_extra) + '\n')
        file.write('true negloglik: ' + str(true_negloglik) + '\n')
        file.write('fake negloglik: ' + str(fake_negloglik) + '\n')
        file.write('time for true negloglik evaluation: ' + str(true_loglik_eval_time) + '\n')
        file.write('time for fake negloglik evaluation: ' + str(fake_loglik_eval_time) + '\n')

    del gp_model, pred_train, pred_inter, pred_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra, true_negloglik, fake_negloglik

#define a function for the FITC approximation that can be called for each tuning parameter
def fitc_run(num_ind_points):
    #fitting
    gp_model = gpb.GPModel(gp_coords=coords_train,  cov_function="matern",cov_fct_shape=1.5,
                likelihood="gaussian", gp_approx="fitc",num_ind_points=num_ind_points)
    start_time = time.time()
    gp_model.fit(y=y_train)
    end_time = time.time()
    fitting_time = end_time - start_time

    #likelihood evaluation
    start_time= time.time()
    true_negloglik = gp_model.neg_log_likelihood(cov_pars=truth,y=y_train)
    end_time = time.time()
    true_loglik_eval_time = end_time - start_time

    start_time= time.time()
    fake_negloglik = gp_model.neg_log_likelihood(cov_pars=truth*2,y=y_train)
    end_time = time.time()
    fake_loglik_eval_time = end_time - start_time
    
    #train
    start_time= time.time() 
    pred_train = gp_model.predict(gp_coords_pred=coords_train,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_train = end_time - start_time
    rmse_train = math.sqrt(np.mean((pred_train['mu'] - y_train)**2))
    pred_mean_train= pred_train['mu']
    pred_var_train = pred_train['var']
    score_train = np.mean((0.5*(pred_mean_train - y_train)**2)/pred_var_train + 0.5*np.log(2*np.pi*pred_var_train))
    rmse_mean_train_comparison = math.sqrt(np.mean((pred_mean_train - exact_pred_mean_train)**2))
    rmse_var_train_comparison = math.sqrt(np.mean((pred_var_train - exact_pred_var_train)**2))
    kl_train = compute_kl(torch.from_numpy(exact_pred_var_train),torch.from_numpy(pred_var_train),torch.from_numpy(exact_pred_mean_train),torch.from_numpy(pred_mean_train))
    crps_train = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_train, np.sqrt(pred_var_train), y_train)])

    #interpolation
    start_time= time.time()
    pred_inter = gp_model.predict(gp_coords_pred= coords_interpolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_inter = end_time - start_time
    rmse_inter = math.sqrt(np.mean((pred_inter['mu'] - y_interpolation)**2))
    pred_mean_inter = pred_inter['mu']
    pred_var_inter = pred_inter['var']
    score_inter = np.mean((0.5*(pred_mean_inter - y_interpolation)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
    rmse_mean_inter_comparison = math.sqrt(np.mean((pred_mean_inter - exact_pred_mean_inter)**2))
    rmse_var_inter_comparison = math.sqrt(np.mean((pred_var_inter - exact_pred_var_inter)**2))
    kl_inter = compute_kl(torch.from_numpy(exact_pred_var_inter),torch.from_numpy(pred_var_inter),torch.from_numpy(exact_pred_mean_inter),torch.from_numpy(pred_mean_inter))
    crps_inter = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_inter, np.sqrt(pred_var_inter), y_interpolation)])

    #extrapolation
    start_time= time.time()
    pred_extra = gp_model.predict(gp_coords_pred= coords_extrapolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_extra = end_time - start_time
    rmse_extra = math.sqrt(np.mean((pred_extra['mu'] - y_extrapolation)**2))
    pred_mean_extra = pred_extra['mu']
    pred_var_extra = pred_extra['var']
    score_extra = np.mean((0.5*(pred_mean_extra - y_extrapolation)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
    rmse_mean_extra_comparison = math.sqrt(np.mean((pred_mean_extra - exact_pred_mean_extra)**2))
    rmse_var_extra_comparison = math.sqrt(np.mean((pred_var_extra - exact_pred_var_extra)**2))
    kl_extra = compute_kl(torch.from_numpy(exact_pred_var_extra),torch.from_numpy(pred_var_extra),torch.from_numpy(exact_pred_mean_extra),torch.from_numpy(pred_mean_extra))
    crps_extra = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_extra, np.sqrt(pred_var_extra), y_extrapolation)])

    #saving results
    filename = "house_fitc_" + str(num_ind_points)
    with open(filename + '.txt', 'w') as file:
        file.write('Fitc ' + str(num_ind_points)  + '\n')

        file.write('time for fitting: ' + str(fitting_time) + '\n')
        file.write('univariate score train: ' + str(score_train) + '\n')
        file.write('univariate score interpolation: ' + str(score_inter) + '\n')
        file.write('univariate score extrapolation: ' + str(score_extra) + '\n')
        file.write('time for train univariate prediction: ' + str(pred_time_train) + '\n')
        file.write('time for interpolation univariate prediction: ' + str(pred_time_inter) + '\n')
        file.write('time for extrapolation univariate prediction: ' + str(pred_time_extra) + '\n')
        file.write('rmse train: ' + str(rmse_train) + '\n')
        file.write('rmse interpolation: ' + str(rmse_inter) + '\n')
        file.write('rmse extrapolation: ' + str(rmse_extra) + '\n')
        file.write('rmse mean train: ' + str(rmse_mean_train_comparison) + '\n')
        file.write('rmse mean interpolation: ' + str(rmse_mean_inter_comparison) + '\n')
        file.write('rmse mean extrapolation: ' + str(rmse_mean_extra_comparison) + '\n')
        file.write('rmse var train: ' + str(rmse_var_train_comparison) + '\n')
        file.write('rmse var interpolation: ' + str(rmse_var_inter_comparison) + '\n')
        file.write('rmse var extrapolation: ' + str(rmse_var_extra_comparison) + '\n')
        file.write('kl train: ' + str(kl_train.item()) + '\n')
        file.write('kl interpolation: ' + str(kl_inter.item()) + '\n')
        file.write('kl extrapolation: ' + str(kl_extra.item()) + '\n')
        file.write('crps train: ' + str(crps_train) + '\n')
        file.write('crps interpolation: ' + str(crps_inter) + '\n')
        file.write('crps extrapolation: ' + str(crps_extra) + '\n')
        file.write('true negloglik: ' + str(true_negloglik) + '\n')
        file.write('fake negloglik: ' + str(fake_negloglik) + '\n')
        file.write('time for true negloglik evaluation: ' + str(true_loglik_eval_time) + '\n')
        file.write('time for fake negloglik evaluation: ' + str(fake_loglik_eval_time) + '\n') 
  
    del gp_model, pred_train, pred_inter, pred_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra, true_negloglik, fake_negloglik

#define a function for the covariance tapering approximation that can be called for each tuning parameter
def tapering_run(cov_fct_taper_range):
    #fitting
    gp_model = gpb.GPModel(gp_coords=coords_train,  cov_function="matern",cov_fct_shape=1.5,
                likelihood="gaussian", gp_approx="tapering",cov_fct_taper_range=cov_fct_taper_range)
    start_time = time.time()
    gp_model.fit(y=y_train)
    end_time = time.time()
    fitting_time = end_time - start_time

    #likelihood evaluation
    start_time= time.time()
    true_negloglik = gp_model.neg_log_likelihood(cov_pars=truth,y=y_train)
    end_time = time.time()
    true_loglik_eval_time = end_time - start_time

    start_time= time.time()
    fake_negloglik = gp_model.neg_log_likelihood(cov_pars=truth*2,y=y_train)
    end_time = time.time()
    fake_loglik_eval_time = end_time - start_time
    
    #train
    start_time= time.time() 
    pred_train = gp_model.predict(gp_coords_pred=coords_train,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_train = end_time - start_time
    rmse_train = math.sqrt(np.mean((pred_train['mu'] - y_train)**2))
    pred_mean_train= pred_train['mu']
    pred_var_train = pred_train['var']
    score_train = np.mean((0.5*(pred_mean_train - y_train)**2)/pred_var_train + 0.5*np.log(2*np.pi*pred_var_train))
    rmse_mean_train_comparison = math.sqrt(np.mean((pred_mean_train - exact_pred_mean_train)**2))
    rmse_var_train_comparison = math.sqrt(np.mean((pred_var_train - exact_pred_var_train)**2))
    kl_train = compute_kl(torch.from_numpy(exact_pred_var_train),torch.from_numpy(pred_var_train),torch.from_numpy(exact_pred_mean_train),torch.from_numpy(pred_mean_train))
    crps_train = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_train, np.sqrt(pred_var_train), y_train)])

    #interpolation
    start_time= time.time()
    pred_inter = gp_model.predict(gp_coords_pred= coords_interpolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_inter = end_time - start_time
    rmse_inter = math.sqrt(np.mean((pred_inter['mu'] - y_interpolation)**2))
    pred_mean_inter = pred_inter['mu']
    pred_var_inter = pred_inter['var']
    score_inter = np.mean((0.5*(pred_mean_inter - y_interpolation)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
    rmse_mean_inter_comparison = math.sqrt(np.mean((pred_mean_inter - exact_pred_mean_inter)**2))
    rmse_var_inter_comparison = math.sqrt(np.mean((pred_var_inter - exact_pred_var_inter)**2))
    kl_inter = compute_kl(torch.from_numpy(exact_pred_var_inter),torch.from_numpy(pred_var_inter),torch.from_numpy(exact_pred_mean_inter),torch.from_numpy(pred_mean_inter))
    crps_inter = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_inter, np.sqrt(pred_var_inter), y_interpolation)])

    #extrapolation
    start_time= time.time()
    pred_extra = gp_model.predict(gp_coords_pred= coords_extrapolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_extra = end_time - start_time
    rmse_extra = math.sqrt(np.mean((pred_extra['mu'] - y_extrapolation)**2))
    pred_mean_extra = pred_extra['mu']
    pred_var_extra = pred_extra['var']
    score_extra = np.mean((0.5*(pred_mean_extra - y_extrapolation)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
    rmse_mean_extra_comparison = math.sqrt(np.mean((pred_mean_extra - exact_pred_mean_extra)**2))
    rmse_var_extra_comparison = math.sqrt(np.mean((pred_var_extra - exact_pred_var_extra)**2))
    kl_extra = compute_kl(torch.from_numpy(exact_pred_var_extra),torch.from_numpy(pred_var_extra),torch.from_numpy(exact_pred_mean_extra),torch.from_numpy(pred_mean_extra))
    crps_extra = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_extra, np.sqrt(pred_var_extra), y_extrapolation)])

    #saving results
    filename = "house_tapering_" + str(cov_fct_taper_range)
    with open(filename + '.txt', 'w') as file:
        file.write('Tapering ' + str(cov_fct_taper_range)  + '\n')

        file.write('time for fitting: ' + str(fitting_time) + '\n')
        file.write('univariate score train: ' + str(score_train) + '\n')
        file.write('univariate score interpolation: ' + str(score_inter) + '\n')
        file.write('univariate score extrapolation: ' + str(score_extra) + '\n')
        file.write('time for train univariate prediction: ' + str(pred_time_train) + '\n')
        file.write('time for interpolation univariate prediction: ' + str(pred_time_inter) + '\n')
        file.write('time for extrapolation univariate prediction: ' + str(pred_time_extra) + '\n')
        file.write('rmse train: ' + str(rmse_train) + '\n')
        file.write('rmse interpolation: ' + str(rmse_inter) + '\n')
        file.write('rmse extrapolation: ' + str(rmse_extra) + '\n')
        file.write('rmse mean train: ' + str(rmse_mean_train_comparison) + '\n')
        file.write('rmse mean interpolation: ' + str(rmse_mean_inter_comparison) + '\n')
        file.write('rmse mean extrapolation: ' + str(rmse_mean_extra_comparison) + '\n')
        file.write('rmse var train: ' + str(rmse_var_train_comparison) + '\n')
        file.write('rmse var interpolation: ' + str(rmse_var_inter_comparison) + '\n')
        file.write('rmse var extrapolation: ' + str(rmse_var_extra_comparison) + '\n')
        file.write('kl train: ' + str(kl_train.item()) + '\n')
        file.write('kl interpolation: ' + str(kl_inter.item()) + '\n')
        file.write('kl extrapolation: ' + str(kl_extra.item()) + '\n')
        file.write('crps train: ' + str(crps_train) + '\n')
        file.write('crps interpolation: ' + str(crps_inter) + '\n')
        file.write('crps extrapolation: ' + str(crps_extra) + '\n')
        file.write('true negloglik: ' + str(true_negloglik) + '\n')
        file.write('fake negloglik: ' + str(fake_negloglik) + '\n')
        file.write('time for true negloglik evaluation: ' + str(true_loglik_eval_time) + '\n')
        file.write('time for fake negloglik evaluation: ' + str(fake_loglik_eval_time) + '\n')    

    del gp_model, pred_train, pred_inter, pred_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra, true_negloglik, fake_negloglik


#define a function for the full-scale approximation that can be called for each set of tuning parameters
def fullscale_run(cov_fct_taper_range, num_ind_points):
    #fitting
    gp_model = gpb.GPModel(gp_coords=coords_train,  cov_function="matern",cov_fct_shape=1.5,
                likelihood="gaussian", gp_approx="full_scale_tapering",cov_fct_taper_range=cov_fct_taper_range, num_ind_points=num_ind_points)
    start_time = time.time()
    gp_model.fit(y=y_train)
    end_time = time.time()
    fitting_time = end_time - start_time

    #likelihood evaluation
    start_time= time.time()
    true_negloglik = gp_model.neg_log_likelihood(cov_pars=truth,y=y_train)
    end_time = time.time()
    true_loglik_eval_time = end_time - start_time

    start_time= time.time()
    fake_negloglik = gp_model.neg_log_likelihood(cov_pars=truth*2,y=y_train)
    end_time = time.time()
    fake_loglik_eval_time = end_time - start_time
    
    #train
    start_time= time.time() 
    pred_train = gp_model.predict(gp_coords_pred=coords_train,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_train = end_time - start_time
    rmse_train = math.sqrt(np.mean((pred_train['mu'] - y_train)**2))
    pred_mean_train= pred_train['mu']
    pred_var_train = pred_train['var']
    score_train = np.mean((0.5*(pred_mean_train - y_train)**2)/pred_var_train + 0.5*np.log(2*np.pi*pred_var_train))
    rmse_mean_train_comparison = math.sqrt(np.mean((pred_mean_train - exact_pred_mean_train)**2))
    rmse_var_train_comparison = math.sqrt(np.mean((pred_var_train - exact_pred_var_train)**2))
    kl_train = compute_kl(torch.from_numpy(exact_pred_var_train),torch.from_numpy(pred_var_train),torch.from_numpy(exact_pred_mean_train),torch.from_numpy(pred_mean_train))
    crps_train = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_train, np.sqrt(pred_var_train), y_train)])

    #interpolation
    start_time= time.time()
    pred_inter = gp_model.predict(gp_coords_pred= coords_interpolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_inter = end_time - start_time
    rmse_inter = math.sqrt(np.mean((pred_inter['mu'] - y_interpolation)**2))
    pred_mean_inter = pred_inter['mu']
    pred_var_inter = pred_inter['var']
    score_inter = np.mean((0.5*(pred_mean_inter - y_interpolation)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
    rmse_mean_inter_comparison = math.sqrt(np.mean((pred_mean_inter - exact_pred_mean_inter)**2))
    rmse_var_inter_comparison = math.sqrt(np.mean((pred_var_inter - exact_pred_var_inter)**2))
    kl_inter = compute_kl(torch.from_numpy(exact_pred_var_inter),torch.from_numpy(pred_var_inter),torch.from_numpy(exact_pred_mean_inter),torch.from_numpy(pred_mean_inter))
    crps_inter = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_inter, np.sqrt(pred_var_inter), y_interpolation)])

    #extrapolation
    start_time= time.time()
    pred_extra = gp_model.predict(gp_coords_pred= coords_extrapolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_extra = end_time - start_time
    rmse_extra = math.sqrt(np.mean((pred_extra['mu'] - y_extrapolation)**2))
    pred_mean_extra = pred_extra['mu']
    pred_var_extra = pred_extra['var']
    score_extra = np.mean((0.5*(pred_mean_extra - y_extrapolation)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
    rmse_mean_extra_comparison = math.sqrt(np.mean((pred_mean_extra - exact_pred_mean_extra)**2))
    rmse_var_extra_comparison = math.sqrt(np.mean((pred_var_extra - exact_pred_var_extra)**2))
    kl_extra = compute_kl(torch.from_numpy(exact_pred_var_extra),torch.from_numpy(pred_var_extra),torch.from_numpy(exact_pred_mean_extra),torch.from_numpy(pred_mean_extra))
    crps_extra = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_extra, np.sqrt(pred_var_extra), y_extrapolation)])

    #saving results
    filename = "house_fullscale_" + str(cov_fct_taper_range) + "_" + str(num_ind_points)
    with open(filename + '.txt', 'w') as file:
        file.write('Full scale ' + str(cov_fct_taper_range) + str(" , ") + str(num_ind_points)  + '\n')

        file.write('time for fitting: ' + str(fitting_time) + '\n')
        file.write('univariate score train: ' + str(score_train) + '\n')
        file.write('univariate score interpolation: ' + str(score_inter) + '\n')
        file.write('univariate score extrapolation: ' + str(score_extra) + '\n')
        file.write('time for train univariate prediction: ' + str(pred_time_train) + '\n')
        file.write('time for interpolation univariate prediction: ' + str(pred_time_inter) + '\n')
        file.write('time for extrapolation univariate prediction: ' + str(pred_time_extra) + '\n')
        file.write('rmse train: ' + str(rmse_train) + '\n')
        file.write('rmse interpolation: ' + str(rmse_inter) + '\n')
        file.write('rmse extrapolation: ' + str(rmse_extra) + '\n')
        file.write('rmse mean train: ' + str(rmse_mean_train_comparison) + '\n')
        file.write('rmse mean interpolation: ' + str(rmse_mean_inter_comparison) + '\n')
        file.write('rmse mean extrapolation: ' + str(rmse_mean_extra_comparison) + '\n')
        file.write('rmse var train: ' + str(rmse_var_train_comparison) + '\n')
        file.write('rmse var interpolation: ' + str(rmse_var_inter_comparison) + '\n')
        file.write('rmse var extrapolation: ' + str(rmse_var_extra_comparison) + '\n')
        file.write('kl train: ' + str(kl_train.item()) + '\n')
        file.write('kl interpolation: ' + str(kl_inter.item()) + '\n')
        file.write('kl extrapolation: ' + str(kl_extra.item()) + '\n')
        file.write('crps train: ' + str(crps_train) + '\n')
        file.write('crps interpolation: ' + str(crps_inter) + '\n')
        file.write('crps extrapolation: ' + str(crps_extra) + '\n')
        file.write('true negloglik: ' + str(true_negloglik) + '\n')
        file.write('fake negloglik: ' + str(fake_negloglik) + '\n')
        file.write('time for true negloglik evaluation: ' + str(true_loglik_eval_time) + '\n')
        file.write('time for fake negloglik evaluation: ' + str(fake_loglik_eval_time) + '\n')

    del gp_model, pred_train, pred_inter, pred_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra, true_negloglik, fake_negloglik


#example usage
vecchia_run(40)
fitc_run(220)
tapering_run(400)
fullscale_run(285,252)
