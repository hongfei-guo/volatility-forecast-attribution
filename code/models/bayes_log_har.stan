data {
  int<lower=2> N;
  int<lower=1> K;
  matrix[N, K] X;
  vector[N] y;
}
parameters {
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 2);
  sigma ~ normal(0, 1);
  y ~ normal(X * beta, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] fitted_mean = X * beta;
  for (t in 1:N) {
    log_lik[t] = normal_lpdf(y[t] | fitted_mean[t], sigma);
  }
}
