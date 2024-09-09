import pandas as pd
import gpboost as gpb
import numpy as np
from scipy.stats import multivariate_normal
import random
from scipy.spatial import distance
import time
import torch
import math


#define a function for the Vecchia approximation that can be called for each tuning parameter
def vecchia_run(range_par,num_neighbors):
    if range_par == 0.5:
        df = pd.read_csv("/data/combined_data_100k_05.csv")
    elif range_par==0.2:
        df = pd.read_csv("/data/combined_data_100k_02.csv")
    elif range_par==0.05:
        df = pd.read_csv("/data/combined_data_100k_005.csv")
    else:
        print('wrong range given')
        exit
    nrep = max(df['rep'])

    #global parameters
    true_range = range_par/4.74
    true_gp_var = 1
    true_error_term = 0.5
    truth = np.array([[true_error_term], [true_gp_var], [true_range]]).flatten()

    #Parameter estimation 
    gp_range_hat= list(); gp_var_hat = list(); error_term_hat = list(); param_estimation_time = list()

    #Likelihood evaluation & comparison
    true_negloglik_eval_time= list(); fake_negloglik_eval_time= list()
    true_estimated_negloglik_values = list(); fake_estimated_negloglik_values = list() 

    #Prediction accuracy 
    scores_train= list(); scores_inter = list(); scores_extra = list()
    train_pred_accuracy_time = list(); inter_pred_accuracy_time = list(); extra_pred_accuracy_time = list()

    #rmse eval
    rmse_train_list = list()
    rmse_inter_list = list()
    rmse_extra_list = list() 

    for i in range(1, nrep + 1):

        data_rep = df[df['rep'] == i]
        train_df = data_rep[data_rep['which'] == 'train']
        coords_train = train_df[['x1', 'x2']].values 
        y_train = train_df['y'].values
        f_train = train_df['f'].values

        ####GPBOOST  
        gp_model = gpb.GPModel(gp_coords=coords_train,  cov_function="matern",cov_fct_shape=1.5,
                        likelihood="gaussian", gp_approx="vecchia",num_neighbors=num_neighbors)
        
        #Parameter estimation
        start_time = time.time()
        gp_model.fit(y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        param_estimation_time.append(execution_time)

        gp_range_value = gp_model.get_cov_pars().loc['Param.', 'GP_range']
        gp_var_value = gp_model.get_cov_pars().loc['Param.', 'GP_var']
        error_term_value = gp_model.get_cov_pars().loc['Param.', 'Error_term']

        gp_range_hat.append(gp_range_value)
        gp_var_hat.append(gp_var_value)
        error_term_hat.append(error_term_value)

        #Likelihood evaluation

        #true
        start_time = time.time()
        true_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        true_negloglik_eval_time.append(execution_time)
        true_estimated_negloglik_values.append(true_negloglik_eval)
        
        #fake
        start_time = time.time()
        fake_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth*2, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        fake_negloglik_eval_time.append(execution_time)
        fake_estimated_negloglik_values.append(fake_negloglik_eval)

        #Prediction accuracy
        if i<=nrep/2:
            ####TRAIN
            #univariate gpboost
            start_time = time.time()
            pred_resp_train = gp_model.predict(gp_coords_pred=coords_train, cov_pars=truth,
                            predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            train_pred_accuracy_time.append(elapsed_time)

            pred_mean_train = pred_resp_train['mu']
            pred_var_train = pred_resp_train['var']
            score_train = np.mean(((pred_mean_train - f_train)**2)/(2*pred_var_train) + 0.5*np.log(2*np.pi*pred_var_train))
            scores_train.append(score_train)

            rmse_train_list.append(np.sqrt(np.mean((f_train - pred_mean_train) ** 2)).item())
            
            ####INTERPOLATION 
            #univariate gpboost
            inter_df = data_rep[data_rep['which'] == 'interpolation']
            coords_inter = inter_df[['x1', 'x2']].values
            f_inter = inter_df['f'].values

            start_time = time.time()
            pred_resp_inter= gp_model.predict(gp_coords_pred=coords_inter, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time= end_time - start_time
            inter_pred_accuracy_time.append(elapsed_time)

            pred_mean_inter = pred_resp_inter['mu']
            pred_var_inter = pred_resp_inter['var']
            score_inter = np.mean((0.5*(pred_mean_inter - f_inter)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
            scores_inter.append(score_inter)

            rmse_inter_list.append(np.sqrt(np.mean((f_inter - pred_mean_inter) ** 2)).item())

            #####EXTRAPOLATION 
            #univariate gpboost
            extra_df = data_rep[data_rep['which'] == 'extrapolation']
            coords_extra = extra_df[['x1', 'x2']].values
            f_extra = extra_df['f'].values

            gp_model.set_prediction_data(vecchia_pred_type="order_obs_first_cond_obs_only")
            start_time = time.time()
            pred_resp_extra= gp_model.predict(gp_coords_pred=coords_extra, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time += end_time - start_time
            extra_pred_accuracy_time.append(elapsed_time)

            pred_mean_extra = pred_resp_extra['mu']
            pred_var_extra = pred_resp_extra['var']
            score_extra = np.mean((0.5*(pred_mean_extra - f_extra)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
            scores_extra.append(score_extra)
            
            rmse_extra_list.append(np.sqrt(np.mean((f_extra - pred_mean_extra) ** 2)).item())

            del inter_df, extra_df, coords_inter, coords_extra, f_inter, f_extra, pred_resp_train, pred_resp_inter, pred_resp_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra

        print('rep done:', i)
        del train_df, data_rep, coords_train, f_train,  y_train, gp_model
        torch.cuda.empty_cache()
   
    #computing results
    mse_gp_range = np.mean((np.array(gp_range_hat) - true_range) ** 2)
    bias_gp_range = np.mean(np.array(gp_range_hat) - true_range)
    mse_gp_var = np.mean((np.array(gp_var_hat) - true_gp_var) ** 2)
    bias_gp_var = np.mean(np.array(gp_var_hat) - true_gp_var)
    mse_error_term = np.mean((np.array(error_term_hat) - true_error_term) ** 2)
    bias_error_term = np.mean(np.array(error_term_hat) - true_error_term)
    
    mean_time_param_estimation = np.mean(param_estimation_time)
    mean_estimated_negloglik_true_pars = np.mean(true_estimated_negloglik_values)
    mean_estimated_negloglik_fake_pars = np.mean(fake_estimated_negloglik_values)
    mean_time_eval_negloglik_true_pars = np.mean(true_negloglik_eval_time)
    mean_time_eval_negloglik_fake_pars = np.mean(fake_negloglik_eval_time)
    mean_univ_score_train = np.mean(scores_train)
    mean_univ_score_inter = np.mean(scores_inter)
    mean_univ_score_extra = np.mean(scores_extra)

    mean_time_univ_pred_train = np.mean(train_pred_accuracy_time)
    mean_time_univ_pred_inter = np.mean(inter_pred_accuracy_time)
    mean_time_univ_pred_extra = np.mean(extra_pred_accuracy_time)

    mean_rmse_train = np.mean(rmse_train_list)
    mean_rmse_inter = np.mean(rmse_inter_list)
    mean_rmse_extra = np.mean(rmse_extra_list)

    #saving results
    filename = "vecchia_100k_" +str(range_par) +"_"+ str(num_neighbors)
    with open(filename + '.txt', 'w') as file:

        file.write('Vecchia ' + str(num_neighbors)  + '\n')
        file.write('True range: ' + str(true_range) + '\n')

        file.write('bias for GP range: ' + str(bias_gp_range) + '\n')
        file.write('MSE for GP range: ' + str(mse_gp_range) + '\n')
        file.write('bias for GP variance: ' + str(bias_gp_var) + '\n')
        file.write('MSE for GP variance: ' + str(mse_gp_var) + '\n')
        file.write('bias for error term variance: ' + str(bias_error_term) + '\n')
        file.write('MSE for error term variance: ' + str(mse_error_term) + '\n')
        file.write('variance for bias of GP range: ' + str(np.var(gp_range_hat)/nrep) + '\n')
        file.write('variance for bias GP of variance: ' + str(np.var(gp_var_hat)/nrep) + '\n')
        file.write('variance for bias error of term variance: ' + str(np.var(error_term_hat)/nrep) + '\n')
        file.write('variance for MSE GP range: ' + str(np.var((np.array(gp_range_hat)-true_range)**2)/nrep) + '\n')
        file.write('variance for MSE GP variance: ' + str(np.var((np.array(gp_var_hat)-true_gp_var)**2)/nrep) + '\n')
        file.write('variance for MSE error term variance: ' + str(np.var((np.array(error_term_hat)-true_error_term)**2)/nrep) + '\n')
        file.write('mean time for parameter estimation: ' + str(mean_time_param_estimation) + '\n')

        file.write('mean estimated negloglik true pars: '  + str(mean_estimated_negloglik_true_pars) + '\n')
        file.write('mean estimated negloglik fake pars: ' + str(mean_estimated_negloglik_fake_pars) + '\n')
        file.write('mean time for true loglik evaluation: ' + str(mean_time_eval_negloglik_true_pars) + '\n')
        file.write('mean time for fake loglik evaluation: ' + str(mean_time_eval_negloglik_fake_pars) + '\n')
        file.write('variance for negloglik true pars: ' + str(np.var(true_estimated_negloglik_values)/nrep) + '\n')
        file.write('variance for negloglik fake pars: ' + str(np.var(fake_estimated_negloglik_values)/nrep) + '\n')

        file.write('mean univariate score train: ' + str(mean_univ_score_train) + '\n')
        file.write('mean univariate score interpolation: ' + str(mean_univ_score_inter) + '\n')
        file.write('mean univariate score extrapolation: ' + str(mean_univ_score_extra) + '\n')
        file.write('variance univariate score train: ' + str(np.var(scores_train)*2/nrep) + '\n')
        file.write('variance univariate score interpolation: ' + str(np.var(scores_inter)*2/nrep) + '\n')
        file.write('variance univariate score extrapolation: ' + str(np.var(scores_extra)*2/nrep) + '\n')
        file.write('mean time for train univariate prediction: ' + str(mean_time_univ_pred_train) + '\n')
        file.write('mean time for interpolation univariate prediction: ' + str(mean_time_univ_pred_inter) + '\n')
        file.write('mean time for extrapolation univariate prediction: ' + str(mean_time_univ_pred_extra) + '\n')

        file.write('RMSE train: ' + str(mean_rmse_train) + '\n')
        file.write('RMSE inter: ' + str(mean_rmse_inter) + '\n')
        file.write('RMSE extra: ' + str(mean_rmse_extra) + '\n')
        file.write('variance for RMSE train: ' + str(np.var(rmse_train_list)*2/nrep) + '\n')
        file.write('variance for RMSE inter: ' + str(np.var(rmse_inter_list)*2/nrep) + '\n')
        file.write('variance for RMSE extra: ' + str(np.var(rmse_extra_list)*2/nrep) + '\n')

#define a function for the tapering approximation that can be called for each tuning parameter
def tapering_run(range_par,cov_fct_taper_range):
    if range_par == 0.5:
        df = pd.read_csv("/data/combined_data_100k_05.csv")
    elif range_par==0.2:
        df = pd.read_csv("/data/combined_data_100k_02.csv")
    elif range_par==0.05:
        df = pd.read_csv("/data/combined_data_100k_005.csv")
    else:
        print('wrong range given')
        exit
    nrep = max(df['rep'])

    #global parameters
    true_range = range_par/4.74
    true_gp_var = 1
    true_error_term = 0.5
    truth = np.array([[true_error_term], [true_gp_var], [true_range]]).flatten()

    #Parameter estimation 
    gp_range_hat= list(); gp_var_hat = list(); error_term_hat = list(); param_estimation_time = list()

    #Likelihood evaluation & comparison
    true_negloglik_eval_time= list(); fake_negloglik_eval_time= list()
    true_estimated_negloglik_values = list(); fake_estimated_negloglik_values = list() 

    #Prediction accuracy 
    scores_train= list(); scores_inter = list(); scores_extra = list()
    train_pred_accuracy_time = list(); inter_pred_accuracy_time = list(); extra_pred_accuracy_time = list()

    #rmse eval
    rmse_train_list = list()
    rmse_inter_list = list()
    rmse_extra_list = list() 

    for i in range(1, nrep + 1):

        data_rep = df[df['rep'] == i]
        train_df = data_rep[data_rep['which'] == 'train']
        coords_train = train_df[['x1', 'x2']].values 
        y_train = train_df['y'].values
        f_train = train_df['f'].values

        ####GPBOOST  
        gp_model = gpb.GPModel(gp_coords=coords_train,  cov_function="matern",cov_fct_shape=1.5,
                        likelihood="gaussian", gp_approx="tapering",num_neighbors=cov_fct_taper_range)
        
        #Parameter estimation
        start_time = time.time()
        gp_model.fit(y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        param_estimation_time.append(execution_time)

        gp_range_value = gp_model.get_cov_pars().loc['Param.', 'GP_range']
        gp_var_value = gp_model.get_cov_pars().loc['Param.', 'GP_var']
        error_term_value = gp_model.get_cov_pars().loc['Param.', 'Error_term']

        gp_range_hat.append(gp_range_value)
        gp_var_hat.append(gp_var_value)
        error_term_hat.append(error_term_value)

        #Likelihood evaluation

        #true
        start_time = time.time()
        true_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        true_negloglik_eval_time.append(execution_time)
        true_estimated_negloglik_values.append(true_negloglik_eval)
        
        #fake
        start_time = time.time()
        fake_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth*2, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        fake_negloglik_eval_time.append(execution_time)
        fake_estimated_negloglik_values.append(fake_negloglik_eval)

        #Prediction accuracy
        if i<=nrep/2:
            ####TRAIN
            #univariate gpboost
            start_time = time.time()
            pred_resp_train = gp_model.predict(gp_coords_pred=coords_train, cov_pars=truth,
                            predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            train_pred_accuracy_time.append(elapsed_time)

            pred_mean_train = pred_resp_train['mu']
            pred_var_train = pred_resp_train['var']
            score_train = np.mean(((pred_mean_train - f_train)**2)/(2*pred_var_train) + 0.5*np.log(2*np.pi*pred_var_train))
            scores_train.append(score_train)

            rmse_train_list.append(np.sqrt(np.mean((f_train - pred_mean_train) ** 2)).item())
            
            ####INTERPOLATION 
            #univariate gpboost
            inter_df = data_rep[data_rep['which'] == 'interpolation']
            coords_inter = inter_df[['x1', 'x2']].values
            f_inter = inter_df['f'].values

            start_time = time.time()
            pred_resp_inter= gp_model.predict(gp_coords_pred=coords_inter, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time= end_time - start_time
            inter_pred_accuracy_time.append(elapsed_time)

            pred_mean_inter = pred_resp_inter['mu']
            pred_var_inter = pred_resp_inter['var']
            score_inter = np.mean((0.5*(pred_mean_inter - f_inter)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
            scores_inter.append(score_inter)

            rmse_inter_list.append(np.sqrt(np.mean((f_inter - pred_mean_inter) ** 2)).item())

            #####EXTRAPOLATION 
            #univariate gpboost
            extra_df = data_rep[data_rep['which'] == 'extrapolation']
            coords_extra = extra_df[['x1', 'x2']].values
            f_extra = extra_df['f'].values

            gp_model.set_prediction_data(vecchia_pred_type="order_obs_first_cond_obs_only")
            start_time = time.time()
            pred_resp_extra= gp_model.predict(gp_coords_pred=coords_extra, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time += end_time - start_time
            extra_pred_accuracy_time.append(elapsed_time)

            pred_mean_extra = pred_resp_extra['mu']
            pred_var_extra = pred_resp_extra['var']
            score_extra = np.mean((0.5*(pred_mean_extra - f_extra)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
            scores_extra.append(score_extra)
            
            rmse_extra_list.append(np.sqrt(np.mean((f_extra - pred_mean_extra) ** 2)).item())

            del inter_df, extra_df, coords_inter, coords_extra, f_inter, f_extra, pred_resp_train, pred_resp_inter, pred_resp_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra

        print('rep done:', i)
        del train_df, data_rep, coords_train, f_train,  y_train, gp_model
        torch.cuda.empty_cache()
   
    #computing results
    mse_gp_range = np.mean((np.array(gp_range_hat) - true_range) ** 2)
    bias_gp_range = np.mean(np.array(gp_range_hat) - true_range)
    mse_gp_var = np.mean((np.array(gp_var_hat) - true_gp_var) ** 2)
    bias_gp_var = np.mean(np.array(gp_var_hat) - true_gp_var)
    mse_error_term = np.mean((np.array(error_term_hat) - true_error_term) ** 2)
    bias_error_term = np.mean(np.array(error_term_hat) - true_error_term)

    mean_time_param_estimation = np.mean(param_estimation_time)
    mean_estimated_negloglik_true_pars = np.mean(true_estimated_negloglik_values)
    mean_estimated_negloglik_fake_pars = np.mean(fake_estimated_negloglik_values)
    mean_time_eval_negloglik_true_pars = np.mean(true_negloglik_eval_time)
    mean_time_eval_negloglik_fake_pars = np.mean(fake_negloglik_eval_time)
    mean_univ_score_train = np.mean(scores_train)
    mean_univ_score_inter = np.mean(scores_inter)
    mean_univ_score_extra = np.mean(scores_extra)

    mean_time_univ_pred_train = np.mean(train_pred_accuracy_time)
    mean_time_univ_pred_inter = np.mean(inter_pred_accuracy_time)
    mean_time_univ_pred_extra = np.mean(extra_pred_accuracy_time)

    mean_rmse_train = np.mean(rmse_train_list)
    mean_rmse_inter = np.mean(rmse_inter_list)
    mean_rmse_extra = np.mean(rmse_extra_list)

    #saving results
    filename = "tapering_100k_" +str(range_par) +"_"+ str(cov_fct_taper_range)
    with open(filename + '.txt', 'w') as file:

        file.write('Tapering ' + str(cov_fct_taper_range)  + '\n')
        file.write('True range: ' + str(true_range) + '\n')

        file.write('bias for GP range: ' + str(bias_gp_range) + '\n')
        file.write('MSE for GP range: ' + str(mse_gp_range) + '\n')
        file.write('bias for GP variance: ' + str(bias_gp_var) + '\n')
        file.write('MSE for GP variance: ' + str(mse_gp_var) + '\n')
        file.write('bias for error term variance: ' + str(bias_error_term) + '\n')
        file.write('MSE for error term variance: ' + str(mse_error_term) + '\n')
        file.write('variance for bias of GP range: ' + str(np.var(gp_range_hat)/nrep) + '\n')
        file.write('variance for bias GP of variance: ' + str(np.var(gp_var_hat)/nrep) + '\n')
        file.write('variance for bias error of term variance: ' + str(np.var(error_term_hat)/nrep) + '\n')
        file.write('variance for MSE GP range: ' + str(np.var((np.array(gp_range_hat)-true_range)**2)/nrep) + '\n')
        file.write('variance for MSE GP variance: ' + str(np.var((np.array(gp_var_hat)-true_gp_var)**2)/nrep) + '\n')
        file.write('variance for MSE error term variance: ' + str(np.var((np.array(error_term_hat)-true_error_term)**2)/nrep) + '\n')
        file.write('mean time for parameter estimation: ' + str(mean_time_param_estimation) + '\n')

        file.write('mean estimated negloglik true pars: '  + str(mean_estimated_negloglik_true_pars) + '\n')
        file.write('mean estimated negloglik fake pars: ' + str(mean_estimated_negloglik_fake_pars) + '\n')
        file.write('mean time for true loglik evaluation: ' + str(mean_time_eval_negloglik_true_pars) + '\n')
        file.write('mean time for fake loglik evaluation: ' + str(mean_time_eval_negloglik_fake_pars) + '\n')
        file.write('variance for negloglik true pars: ' + str(np.var(true_estimated_negloglik_values)/nrep) + '\n')
        file.write('variance for negloglik fake pars: ' + str(np.var(fake_estimated_negloglik_values)/nrep) + '\n')

        file.write('mean univariate score train: ' + str(mean_univ_score_train) + '\n')
        file.write('mean univariate score interpolation: ' + str(mean_univ_score_inter) + '\n')
        file.write('mean univariate score extrapolation: ' + str(mean_univ_score_extra) + '\n')
        file.write('variance univariate score train: ' + str(np.var(scores_train)*2/nrep) + '\n')
        file.write('variance univariate score interpolation: ' + str(np.var(scores_inter)*2/nrep) + '\n')
        file.write('variance univariate score extrapolation: ' + str(np.var(scores_extra)*2/nrep) + '\n')
        file.write('mean time for train univariate prediction: ' + str(mean_time_univ_pred_train) + '\n')
        file.write('mean time for interpolation univariate prediction: ' + str(mean_time_univ_pred_inter) + '\n')
        file.write('mean time for extrapolation univariate prediction: ' + str(mean_time_univ_pred_extra) + '\n')

        file.write('RMSE train: ' + str(mean_rmse_train) + '\n')
        file.write('RMSE inter: ' + str(mean_rmse_inter) + '\n')
        file.write('RMSE extra: ' + str(mean_rmse_extra) + '\n')
        file.write('variance for RMSE train: ' + str(np.var(rmse_train_list)*2/nrep) + '\n')
        file.write('variance for RMSE inter: ' + str(np.var(rmse_inter_list)*2/nrep) + '\n')
        file.write('variance for RMSE extra: ' + str(np.var(rmse_extra_list)*2/nrep) + '\n')

#define a function for the fitc approximation that can be called for each tuning parameter
def fitc_run_run(range_par,num_ind_points):
    if range_par == 0.5:
        df = pd.read_csv("/data/combined_data_100k_05.csv")
    elif range_par==0.2:
        df = pd.read_csv("/data/combined_data_100k_02.csv")
    elif range_par==0.05:
        df = pd.read_csv("/data/combined_data_100k_005.csv")
    else:
        print('wrong range given')
        exit
    nrep = max(df['rep'])

    #global parameters
    true_range = range_par/4.74
    true_gp_var = 1
    true_error_term = 0.5
    truth = np.array([[true_error_term], [true_gp_var], [true_range]]).flatten()

    #Parameter estimation 
    gp_range_hat= list(); gp_var_hat = list(); error_term_hat = list(); param_estimation_time = list()

    #Likelihood evaluation & comparison
    true_negloglik_eval_time= list(); fake_negloglik_eval_time= list()
    true_estimated_negloglik_values = list(); fake_estimated_negloglik_values = list() 

    #Prediction accuracy 
    scores_train= list(); scores_inter = list(); scores_extra = list()
    train_pred_accuracy_time = list(); inter_pred_accuracy_time = list(); extra_pred_accuracy_time = list()

    #rmse eval
    rmse_train_list = list()
    rmse_inter_list = list()
    rmse_extra_list = list() 

    for i in range(1, nrep + 1):

        data_rep = df[df['rep'] == i]
        train_df = data_rep[data_rep['which'] == 'train']
        coords_train = train_df[['x1', 'x2']].values 
        y_train = train_df['y'].values
        f_train = train_df['f'].values

        ####GPBOOST  
        gp_model = gpb.GPModel(gp_coords=coords_train,  cov_function="matern",cov_fct_shape=1.5,
                        likelihood="gaussian", gp_approx="fitc",num_neighbors=num_ind_points)
        
        #Parameter estimation
        start_time = time.time()
        gp_model.fit(y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        param_estimation_time.append(execution_time)

        gp_range_value = gp_model.get_cov_pars().loc['Param.', 'GP_range']
        gp_var_value = gp_model.get_cov_pars().loc['Param.', 'GP_var']
        error_term_value = gp_model.get_cov_pars().loc['Param.', 'Error_term']

        gp_range_hat.append(gp_range_value)
        gp_var_hat.append(gp_var_value)
        error_term_hat.append(error_term_value)

        #Likelihood evaluation

        #true
        start_time = time.time()
        true_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        true_negloglik_eval_time.append(execution_time)
        true_estimated_negloglik_values.append(true_negloglik_eval)
        
        #fake
        start_time = time.time()
        fake_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth*2, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        fake_negloglik_eval_time.append(execution_time)
        fake_estimated_negloglik_values.append(fake_negloglik_eval)

        #Prediction accuracy
        if i<=nrep/2:
            ####TRAIN
            #univariate gpboost
            start_time = time.time()
            pred_resp_train = gp_model.predict(gp_coords_pred=coords_train, cov_pars=truth,
                            predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            train_pred_accuracy_time.append(elapsed_time)

            pred_mean_train = pred_resp_train['mu']
            pred_var_train = pred_resp_train['var']
            score_train = np.mean(((pred_mean_train - f_train)**2)/(2*pred_var_train) + 0.5*np.log(2*np.pi*pred_var_train))
            scores_train.append(score_train)

            rmse_train_list.append(np.sqrt(np.mean((f_train - pred_mean_train) ** 2)).item())
            
            ####INTERPOLATION 
            #univariate gpboost
            inter_df = data_rep[data_rep['which'] == 'interpolation']
            coords_inter = inter_df[['x1', 'x2']].values
            f_inter = inter_df['f'].values

            start_time = time.time()
            pred_resp_inter= gp_model.predict(gp_coords_pred=coords_inter, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time= end_time - start_time
            inter_pred_accuracy_time.append(elapsed_time)

            pred_mean_inter = pred_resp_inter['mu']
            pred_var_inter = pred_resp_inter['var']
            score_inter = np.mean((0.5*(pred_mean_inter - f_inter)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
            scores_inter.append(score_inter)

            rmse_inter_list.append(np.sqrt(np.mean((f_inter - pred_mean_inter) ** 2)).item())

            #####EXTRAPOLATION 
            #univariate gpboost
            extra_df = data_rep[data_rep['which'] == 'extrapolation']
            coords_extra = extra_df[['x1', 'x2']].values
            f_extra = extra_df['f'].values

            gp_model.set_prediction_data(vecchia_pred_type="order_obs_first_cond_obs_only")
            start_time = time.time()
            pred_resp_extra= gp_model.predict(gp_coords_pred=coords_extra, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time += end_time - start_time
            extra_pred_accuracy_time.append(elapsed_time)

            pred_mean_extra = pred_resp_extra['mu']
            pred_var_extra = pred_resp_extra['var']
            score_extra = np.mean((0.5*(pred_mean_extra - f_extra)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
            scores_extra.append(score_extra)
            
            rmse_extra_list.append(np.sqrt(np.mean((f_extra - pred_mean_extra) ** 2)).item())

            del inter_df, extra_df, coords_inter, coords_extra, f_inter, f_extra, pred_resp_train, pred_resp_inter, pred_resp_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra

        print('rep done:', i)
        del train_df, data_rep, coords_train, f_train,  y_train, gp_model
        torch.cuda.empty_cache()
   
    #computing results
    mse_gp_range = np.mean((np.array(gp_range_hat) - true_range) ** 2)
    bias_gp_range = np.mean(np.array(gp_range_hat) - true_range)
    mse_gp_var = np.mean((np.array(gp_var_hat) - true_gp_var) ** 2)
    bias_gp_var = np.mean(np.array(gp_var_hat) - true_gp_var)
    mse_error_term = np.mean((np.array(error_term_hat) - true_error_term) ** 2)
    bias_error_term = np.mean(np.array(error_term_hat) - true_error_term)

    mean_time_param_estimation = np.mean(param_estimation_time)
    mean_estimated_negloglik_true_pars = np.mean(true_estimated_negloglik_values)
    mean_estimated_negloglik_fake_pars = np.mean(fake_estimated_negloglik_values)
    mean_time_eval_negloglik_true_pars = np.mean(true_negloglik_eval_time)
    mean_time_eval_negloglik_fake_pars = np.mean(fake_negloglik_eval_time)
    mean_univ_score_train = np.mean(scores_train)
    mean_univ_score_inter = np.mean(scores_inter)
    mean_univ_score_extra = np.mean(scores_extra)

    mean_time_univ_pred_train = np.mean(train_pred_accuracy_time)
    mean_time_univ_pred_inter = np.mean(inter_pred_accuracy_time)
    mean_time_univ_pred_extra = np.mean(extra_pred_accuracy_time)

    mean_rmse_train = np.mean(rmse_train_list)
    mean_rmse_inter = np.mean(rmse_inter_list)
    mean_rmse_extra = np.mean(rmse_extra_list)

    #saving results
    filename = "fitc_100k_" +str(range_par) +"_"+ str(num_ind_points)
    with open(filename + '.txt', 'w') as file:

        file.write('Fitc ' + str(num_ind_points)  + '\n')
        file.write('True range: ' + str(true_range) + '\n')

        file.write('bias for GP range: ' + str(bias_gp_range) + '\n')
        file.write('MSE for GP range: ' + str(mse_gp_range) + '\n')
        file.write('bias for GP variance: ' + str(bias_gp_var) + '\n')
        file.write('MSE for GP variance: ' + str(mse_gp_var) + '\n')
        file.write('bias for error term variance: ' + str(bias_error_term) + '\n')
        file.write('MSE for error term variance: ' + str(mse_error_term) + '\n')
        file.write('variance for bias of GP range: ' + str(np.var(gp_range_hat)/nrep) + '\n')
        file.write('variance for bias GP of variance: ' + str(np.var(gp_var_hat)/nrep) + '\n')
        file.write('variance for bias error of term variance: ' + str(np.var(error_term_hat)/nrep) + '\n')
        file.write('variance for MSE GP range: ' + str(np.var((np.array(gp_range_hat)-true_range)**2)/nrep) + '\n')
        file.write('variance for MSE GP variance: ' + str(np.var((np.array(gp_var_hat)-true_gp_var)**2)/nrep) + '\n')
        file.write('variance for MSE error term variance: ' + str(np.var((np.array(error_term_hat)-true_error_term)**2)/nrep) + '\n')
        file.write('mean time for parameter estimation: ' + str(mean_time_param_estimation) + '\n')

        file.write('mean estimated negloglik true pars: '  + str(mean_estimated_negloglik_true_pars) + '\n')
        file.write('mean estimated negloglik fake pars: ' + str(mean_estimated_negloglik_fake_pars) + '\n')
        file.write('mean time for true loglik evaluation: ' + str(mean_time_eval_negloglik_true_pars) + '\n')
        file.write('mean time for fake loglik evaluation: ' + str(mean_time_eval_negloglik_fake_pars) + '\n')
        file.write('variance for negloglik true pars: ' + str(np.var(true_estimated_negloglik_values)/nrep) + '\n')
        file.write('variance for negloglik fake pars: ' + str(np.var(fake_estimated_negloglik_values)/nrep) + '\n')

        file.write('mean univariate score train: ' + str(mean_univ_score_train) + '\n')
        file.write('mean univariate score interpolation: ' + str(mean_univ_score_inter) + '\n')
        file.write('mean univariate score extrapolation: ' + str(mean_univ_score_extra) + '\n')
        file.write('variance univariate score train: ' + str(np.var(scores_train)*2/nrep) + '\n')
        file.write('variance univariate score interpolation: ' + str(np.var(scores_inter)*2/nrep) + '\n')
        file.write('variance univariate score extrapolation: ' + str(np.var(scores_extra)*2/nrep) + '\n')
        file.write('mean time for train univariate prediction: ' + str(mean_time_univ_pred_train) + '\n')
        file.write('mean time for interpolation univariate prediction: ' + str(mean_time_univ_pred_inter) + '\n')
        file.write('mean time for extrapolation univariate prediction: ' + str(mean_time_univ_pred_extra) + '\n')

        file.write('RMSE train: ' + str(mean_rmse_train) + '\n')
        file.write('RMSE inter: ' + str(mean_rmse_inter) + '\n')
        file.write('RMSE extra: ' + str(mean_rmse_extra) + '\n')
        file.write('variance for RMSE train: ' + str(np.var(rmse_train_list)*2/nrep) + '\n')
        file.write('variance for RMSE inter: ' + str(np.var(rmse_inter_list)*2/nrep) + '\n')
        file.write('variance for RMSE extra: ' + str(np.var(rmse_extra_list)*2/nrep) + '\n')

#define a function for the full-scale approximation that can be called for each set of tuning parameters
def fullscale_run(range_par,num_ind_points,cov_fct_taper_range):
    if range_par == 0.5:
        df = pd.read_csv("/data/combined_data_100k_05.csv")
    elif range_par==0.2:
        df = pd.read_csv("/data/combined_data_100k_02.csv")
    elif range_par==0.05:
        df = pd.read_csv("/data/combined_data_100k_005.csv")
    else:
        print('wrong range given')
        exit
    nrep = max(df['rep'])

    #global parameters
    true_range = range_par/4.74
    true_gp_var = 1
    true_error_term = 0.5
    truth = np.array([[true_error_term], [true_gp_var], [true_range]]).flatten()

    #Parameter estimation 
    gp_range_hat= list(); gp_var_hat = list(); error_term_hat = list(); param_estimation_time = list()

    #Likelihood evaluation & comparison
    true_negloglik_eval_time= list(); fake_negloglik_eval_time= list()
    true_estimated_negloglik_values = list(); fake_estimated_negloglik_values = list() 

    #Prediction accuracy 
    scores_train= list(); scores_inter = list(); scores_extra = list()
    train_pred_accuracy_time = list(); inter_pred_accuracy_time = list(); extra_pred_accuracy_time = list()

    #rmse eval
    rmse_train_list = list()
    rmse_inter_list = list()
    rmse_extra_list = list() 

    for i in range(1, nrep + 1):

        data_rep = df[df['rep'] == i]
        train_df = data_rep[data_rep['which'] == 'train']
        coords_train = train_df[['x1', 'x2']].values 
        y_train = train_df['y'].values
        f_train = train_df['f'].values

        ####GPBOOST  
        gp_model = gpb.GPModel(gp_coords=coords_train,  cov_function="matern",cov_fct_shape=1.5,
                        likelihood="gaussian", gp_approx="full_scale_tapering",cov_fct_taper_range=cov_fct_taper_range,num_ind_points=num_ind_points)
        
        #Parameter estimation
        start_time = time.time()
        gp_model.fit(y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        param_estimation_time.append(execution_time)

        gp_range_value = gp_model.get_cov_pars().loc['Param.', 'GP_range']
        gp_var_value = gp_model.get_cov_pars().loc['Param.', 'GP_var']
        error_term_value = gp_model.get_cov_pars().loc['Param.', 'Error_term']

        gp_range_hat.append(gp_range_value)
        gp_var_hat.append(gp_var_value)
        error_term_hat.append(error_term_value)

        #Likelihood evaluation

        #true
        start_time = time.time()
        true_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        true_negloglik_eval_time.append(execution_time)
        true_estimated_negloglik_values.append(true_negloglik_eval)
        
        #fake
        start_time = time.time()
        fake_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth*2, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        fake_negloglik_eval_time.append(execution_time)
        fake_estimated_negloglik_values.append(fake_negloglik_eval)

        #Prediction accuracy
        if i<=nrep/2:
            ####TRAIN
            #univariate gpboost
            start_time = time.time()
            pred_resp_train = gp_model.predict(gp_coords_pred=coords_train, cov_pars=truth,
                            predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            train_pred_accuracy_time.append(elapsed_time)

            pred_mean_train = pred_resp_train['mu']
            pred_var_train = pred_resp_train['var']
            score_train = np.mean(((pred_mean_train - f_train)**2)/(2*pred_var_train) + 0.5*np.log(2*np.pi*pred_var_train))
            scores_train.append(score_train)

            rmse_train_list.append(np.sqrt(np.mean((f_train - pred_mean_train) ** 2)).item())
            
            ####INTERPOLATION 
            #univariate gpboost
            inter_df = data_rep[data_rep['which'] == 'interpolation']
            coords_inter = inter_df[['x1', 'x2']].values
            f_inter = inter_df['f'].values

            start_time = time.time()
            pred_resp_inter= gp_model.predict(gp_coords_pred=coords_inter, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time= end_time - start_time
            inter_pred_accuracy_time.append(elapsed_time)

            pred_mean_inter = pred_resp_inter['mu']
            pred_var_inter = pred_resp_inter['var']
            score_inter = np.mean((0.5*(pred_mean_inter - f_inter)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
            scores_inter.append(score_inter)

            rmse_inter_list.append(np.sqrt(np.mean((f_inter - pred_mean_inter) ** 2)).item())

            #####EXTRAPOLATION 
            #univariate gpboost
            extra_df = data_rep[data_rep['which'] == 'extrapolation']
            coords_extra = extra_df[['x1', 'x2']].values
            f_extra = extra_df['f'].values

            gp_model.set_prediction_data(vecchia_pred_type="order_obs_first_cond_obs_only")
            start_time = time.time()
            pred_resp_extra= gp_model.predict(gp_coords_pred=coords_extra, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time += end_time - start_time
            extra_pred_accuracy_time.append(elapsed_time)

            pred_mean_extra = pred_resp_extra['mu']
            pred_var_extra = pred_resp_extra['var']
            score_extra = np.mean((0.5*(pred_mean_extra - f_extra)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
            scores_extra.append(score_extra)
            
            rmse_extra_list.append(np.sqrt(np.mean((f_extra - pred_mean_extra) ** 2)).item())

            del inter_df, extra_df, coords_inter, coords_extra, f_inter, f_extra, pred_resp_train, pred_resp_inter, pred_resp_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra

        print('rep done:', i)
        del train_df, data_rep, coords_train, f_train,  y_train, gp_model
        torch.cuda.empty_cache()
   
    #computing results
    mse_gp_range = np.mean((np.array(gp_range_hat) - true_range) ** 2)
    bias_gp_range = np.mean(np.array(gp_range_hat) - true_range)
    mse_gp_var = np.mean((np.array(gp_var_hat) - true_gp_var) ** 2)
    bias_gp_var = np.mean(np.array(gp_var_hat) - true_gp_var)
    mse_error_term = np.mean((np.array(error_term_hat) - true_error_term) ** 2)
    bias_error_term = np.mean(np.array(error_term_hat) - true_error_term)

    mean_time_param_estimation = np.mean(param_estimation_time)
    mean_estimated_negloglik_true_pars = np.mean(true_estimated_negloglik_values)
    mean_estimated_negloglik_fake_pars = np.mean(fake_estimated_negloglik_values)
    mean_time_eval_negloglik_true_pars = np.mean(true_negloglik_eval_time)
    mean_time_eval_negloglik_fake_pars = np.mean(fake_negloglik_eval_time)
    mean_univ_score_train = np.mean(scores_train)
    mean_univ_score_inter = np.mean(scores_inter)
    mean_univ_score_extra = np.mean(scores_extra)

    mean_time_univ_pred_train = np.mean(train_pred_accuracy_time)
    mean_time_univ_pred_inter = np.mean(inter_pred_accuracy_time)
    mean_time_univ_pred_extra = np.mean(extra_pred_accuracy_time)

    mean_rmse_train = np.mean(rmse_train_list)
    mean_rmse_inter = np.mean(rmse_inter_list)
    mean_rmse_extra = np.mean(rmse_extra_list)

    #saving results
    filename = "fullscale_100k_" +str(range_par) +"_"+ str(cov_fct_taper_range) + "_" + str(num_ind_points)
    with open(filename + '.txt', 'w') as file:

        file.write('Full scale ' + str(cov_fct_taper_range) + str(" , ") + str(num_ind_points)  + '\n')
        file.write('True range: ' + str(true_range) + '\n')

        file.write('bias for GP range: ' + str(bias_gp_range) + '\n')
        file.write('MSE for GP range: ' + str(mse_gp_range) + '\n')
        file.write('bias for GP variance: ' + str(bias_gp_var) + '\n')
        file.write('MSE for GP variance: ' + str(mse_gp_var) + '\n')
        file.write('bias for error term variance: ' + str(bias_error_term) + '\n')
        file.write('MSE for error term variance: ' + str(mse_error_term) + '\n')
        file.write('variance for bias of GP range: ' + str(np.var(gp_range_hat)/nrep) + '\n')
        file.write('variance for bias GP of variance: ' + str(np.var(gp_var_hat)/nrep) + '\n')
        file.write('variance for bias error of term variance: ' + str(np.var(error_term_hat)/nrep) + '\n')
        file.write('variance for MSE GP range: ' + str(np.var((np.array(gp_range_hat)-true_range)**2)/nrep) + '\n')
        file.write('variance for MSE GP variance: ' + str(np.var((np.array(gp_var_hat)-true_gp_var)**2)/nrep) + '\n')
        file.write('variance for MSE error term variance: ' + str(np.var((np.array(error_term_hat)-true_error_term)**2)/nrep) + '\n')
        file.write('mean time for parameter estimation: ' + str(mean_time_param_estimation) + '\n')

        file.write('mean estimated negloglik true pars: '  + str(mean_estimated_negloglik_true_pars) + '\n')
        file.write('mean estimated negloglik fake pars: ' + str(mean_estimated_negloglik_fake_pars) + '\n')
        file.write('mean time for true loglik evaluation: ' + str(mean_time_eval_negloglik_true_pars) + '\n')
        file.write('mean time for fake loglik evaluation: ' + str(mean_time_eval_negloglik_fake_pars) + '\n')
        file.write('variance for negloglik true pars: ' + str(np.var(true_estimated_negloglik_values)/nrep) + '\n')
        file.write('variance for negloglik fake pars: ' + str(np.var(fake_estimated_negloglik_values)/nrep) + '\n')

        file.write('mean univariate score train: ' + str(mean_univ_score_train) + '\n')
        file.write('mean univariate score interpolation: ' + str(mean_univ_score_inter) + '\n')
        file.write('mean univariate score extrapolation: ' + str(mean_univ_score_extra) + '\n')
        file.write('variance univariate score train: ' + str(np.var(scores_train)*2/nrep) + '\n')
        file.write('variance univariate score interpolation: ' + str(np.var(scores_inter)*2/nrep) + '\n')
        file.write('variance univariate score extrapolation: ' + str(np.var(scores_extra)*2/nrep) + '\n')
        file.write('mean time for train univariate prediction: ' + str(mean_time_univ_pred_train) + '\n')
        file.write('mean time for interpolation univariate prediction: ' + str(mean_time_univ_pred_inter) + '\n')
        file.write('mean time for extrapolation univariate prediction: ' + str(mean_time_univ_pred_extra) + '\n')

        file.write('RMSE train: ' + str(mean_rmse_train) + '\n')
        file.write('RMSE inter: ' + str(mean_rmse_inter) + '\n')
        file.write('RMSE extra: ' + str(mean_rmse_extra) + '\n')
        file.write('variance for RMSE train: ' + str(np.var(rmse_train_list)*2/nrep) + '\n')
        file.write('variance for RMSE inter: ' + str(np.var(rmse_inter_list)*2/nrep) + '\n')
        file.write('variance for RMSE extra: ' + str(np.var(rmse_extra_list)*2/nrep) + '\n')





#example usage
vecchia_run(0.5, 340)






