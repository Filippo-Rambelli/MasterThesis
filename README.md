This is the code for the master thesis "An accuracy-runtime trade-off comparison of large-data Gaussian process approximations for spatial data" written by Filippo Rambelli @ETH ZURICH.

Our repository contains scripts to carry out comparisons between eight Gaussian process approximations. Specifically, we included Vecchia's approximation, covariance tapering, modified predictive process/ FITC, full-scale tapering, fixed rank kriging, the SPDE approach, and periodic embedding. The first four approximations are implemented in Python in the GPboost package, while the remaining four in different R libraries. 

The comparison was carried out on four real-world datasets, available in the 'data' folder, and on six types of simulated datasets. Due to the large size, we did not upload the synthetic datasets in the repository, however, it is possible to generate them by running the script simulate_data.R located in the 'code' folder. There, in the subfolder names as the involved dataset(s), one can also find the scripts for running the various approximations.

In the 'saved_values_exactGP' folder we also stored some results from exact computations on datasets with moderate sample size, particularly the house price dataset and the simulated datasets with sample size N=10'000.
