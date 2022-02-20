functions {
  /**
  * Log hazard for Weibull distribution
  *
  * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @param shape Real, Weibull shape
  * @return A vector
  */
  real weibull_log_haz(real eta, real t, real shape) {
    real res;
    res = log(shape) + (shape - 1) * log(t) + eta;
    return res;
  }

  /**
  * Log survival and log CDF for Weibull distribution
  *
  * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @param shape Real, Weibull shape
  * @return A vector
  */
  real weibull_log_surv(real eta, real t, real shape) {
    real res;
    res = - pow(t, shape) .* exp(eta);
    return res;
  }

  /*** This function is a helper that implements the colon function
  * @param lower lower bound
  * @param upper upper bound
  * @return int[] of number range
  */
  int[] colon(int lower, int upper) {
    int length = upper - lower + 1;
    int res[length];
    for(l in 1:length) {
      res[l] = lower + l - 1;
    }
    return res;
  }
  /**
  * GP: computes sigma_yless Gaussian Process
  * @param volatility volatility of gaussian process
  * @param amplitude
  * @param GP_z_scores
  * @param n_x number of x
  * @param x
  * @return A vector
  */
  vector GP(real volatility, real amplitude, vector GP_z_scores, int n_x, real[] x ) {
    matrix[n_x,n_x] cov_mat ;
    real amplitude_sq_plus_jitter ;
    amplitude_sq_plus_jitter = amplitude^2 + 0.001 ;
    cov_mat = gp_exp_quad_cov(x, amplitude, 1/volatility) ;
    cov_mat = add_diag(cov_mat, amplitude_sq_plus_jitter);
    return(cholesky_decompose(cov_mat) * GP_z_scores ) ;
  }
}

data {
  // N: number of observations in y
  int<lower=1> N;
  // time: vector of observations for time
  vector<lower = 0>[N] time;
  int<lower=0, upper = 1> status[N]; // status indicator

  /*** GP stuff ***/

  // n_w: number of biomarkers in predictor array
  int n_w;
  // data on biomarkers
  matrix[N, n_w] w; // array of intercept and treatments

  // n_x: number of unique x values
  int<lower=1> n_x ;
  // x: unique values of x
  //     should be scaled to min=0,max=1
  real x[n_x] ;
  // x_index: vector indicating which x is associated with each y
  int x_index[N] ;
}

transformed data {
  // hyper-parameters
  real tau_shape = 2.0; // dirichlet alpha hyper-prior for M-splines
  real tau_mu = 10.0;
  real tau_b = 10.0;
}

parameters {
  // intercept
  real mu_raw;
  real shape_raw;
  // volatility_helper: helper for cauchy-distributed volatility (see transformed parameters)
  vector<lower=0,upper=pi()/2>[n_w] volatility_helper ;

  // amplitude: amplitude of GPs
  vector<lower=0>[n_w] amplitude ;

  // f_GP_z_scores: helper variable for GPs (see transformed parameters)
  matrix[n_x,n_w] f_GP_z_scores;
}

transformed parameters {

  // intercept of hazard model
  real mu = tau_mu * mu_raw;
  real shape = tau_shape * shape_raw;
  // volatility: volatility of GPs (a.k.a. inverse-lengthscale)
  vector[n_w] volatility ;
  // f: GPs
  matrix[n_x,n_w] f ;

  //next line implies volatility ~ cauchy(0,1)
  volatility = 1 * tan (volatility_helper) ;

  // loop over predictors, computing GPs for each predictor
  for(wi in 1:n_w){
    f[,wi] = GP (volatility[wi] , amplitude[wi] , f_GP_z_scores[ ,wi] , n_x , x);
  }

}

model {

  /*** priors for survival model ***/
  target += std_normal_lpdf (mu_raw); // priors for baseline hazard
  target += std_normal_lpdf (shape_raw);

  /*** priors for GP  ***/
  target += std_normal_lpdf( to_vector(f_GP_z_scores));   // normal(0,1) priors on GP_z_scores
  amplitude ~ weibull (2, 1);   // amplitude prior peaked near .8

  for ( i in 1:N ) {

    real eta = sum(w[i,] .* f[x_index[i], ]);
    eta += mu; // add intercept

    // survival likelihood
    if(status[i] == 1) {

      target += weibull_log_haz (eta, time[i], shape);
      target += weibull_log_surv (eta, time[i], shape);

    } else { // censored observation

      target += weibull_log_surv (eta, time[i], shape);

    }

  } // close for loop of individuals!
}
