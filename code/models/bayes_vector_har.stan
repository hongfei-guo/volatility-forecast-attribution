data {
  int<lower=2> N;
  int<lower=1> K;
  matrix[N, K] X;
  matrix[N, 2] Y;
}
parameters {
  matrix[K, 2] B;
  vector<lower=0>[2] sigma;
  real<lower=-1, upper=1> rho;
}
transformed parameters {
  matrix[2, 2] L_Omega = rep_matrix(0, 2, 2);
  L_Omega[1, 1] = 1;
  L_Omega[2, 1] = rho;
  L_Omega[2, 2] = sqrt(1 - square(rho));
  matrix[2, 2] L_Sigma = diag_pre_multiply(sigma, L_Omega);
}
model {
  to_vector(B) ~ normal(0, 2);
  sigma ~ normal(0, 1);
  // For a two-dimensional correlation matrix, this is exactly LKJ(eta=2).
  target += log1m(square(rho));
  for (t in 1:N) {
    Y[t]' ~ multi_normal_cholesky((X[t] * B)', L_Sigma);
  }
}
generated quantities {
  matrix[2, 2] residual_covariance = multiply_lower_tri_self_transpose(L_Sigma);
  vector[N] log_lik;
  for (t in 1:N) {
    log_lik[t] = multi_normal_cholesky_lpdf(Y[t]' | (X[t] * B)', L_Sigma);
  }
}
