functions {
  real standardized_t_logpdf_from_log_sd(real z, real nu, real log_conditional_sd) {
    real log_scale = log_conditional_sd + 0.5 * (log(nu - 2) - log(nu));
    real log_ratio_squared = z == 0 ? negative_infinity()
      : 2 * log(abs(z)) - log(nu) - 2 * log_scale;
    return lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu)
      - 0.5 * log(nu * pi()) - log_scale
      - 0.5 * (nu + 1) * log1p_exp(log_ratio_squared);
  }
}
data {
  int<lower=2> N;
  vector[N] z;
  real mu_prior_mean;
  real<lower=0> mu_prior_sd;
  real phi_logit_loc;
  real<lower=0> phi_logit_scale;
  real<lower=0> sigma_h_prior_sd;
  real<lower=0> h0_prior_sd;
  real nu_minus_two_log_mean;
  real<lower=0> nu_minus_two_log_sd;
}
parameters {
  real mu_std;
  real phi_std;
  real<lower=0> sigma_eta_std;
  real nu_std;
  real h0_raw;
  vector[N - 1] alpha;
}
transformed parameters {
  real mu = mu_prior_mean + mu_prior_sd * mu_std;
  real phi_raw = phi_logit_loc + phi_logit_scale * phi_std;
  real log_phi = log_inv_logit(phi_raw);
  real<lower=0, upper=1> phi = exp(log_phi);
  real log_stationary_shrink = 0.5 * log1m_exp(2 * log_phi);
  real<lower=0> stationary_shrink = exp(log_stationary_shrink);
  real<lower=0> sigma_eta = sigma_h_prior_sd * sigma_eta_std;
  real<lower=0> sigma_h_std = sigma_eta_std / stationary_shrink;
  real<lower=0> sigma_h = sigma_h_prior_sd * sigma_h_std;
  real<lower=0> nu_minus_two = exp(
    nu_minus_two_log_mean + nu_minus_two_log_sd * nu_std);
  real<lower=2> nu = 2 + nu_minus_two;
  vector[N] h;
  // The initial AR state is independent of the sampled long-run level.
  h[1] = mu_prior_mean + h0_prior_sd * h0_raw;
  for (t in 2:N) {
    h[t] = mu + phi * (h[t - 1] - mu) + sigma_eta * alpha[t - 1];
  }
}
model {
  mu_std ~ std_normal();
  phi_std ~ std_normal();
  sigma_eta_std ~ normal(0, stationary_shrink);
  nu_std ~ std_normal();
  h0_raw ~ std_normal();
  alpha ~ std_normal();
  for (t in 1:N) {
    target += standardized_t_logpdf_from_log_sd(z[t], nu, 0.5 * h[t]);
  }
}
generated quantities {
  vector[N] conditional_variance;
  vector[N] log_lik;
  real phi_gap = 1 - phi;
  for (t in 1:N) {
    real sd_t = exp(0.5 * h[t]);
    conditional_variance[t] = square(sd_t);
    log_lik[t] = standardized_t_logpdf_from_log_sd(z[t], nu, 0.5 * h[t]);
  }
}
