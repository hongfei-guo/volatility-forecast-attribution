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
  int<lower=1> D;
  int<lower=1> H;
  vector[N] z;
  matrix[N, D] X;
  vector[N] q;
  real mu_prior_mean;
  real<lower=0> mu_prior_sd;
  real phi_logit_loc;
  real<lower=0> phi_logit_scale;
  real<lower=0> sigma_h_prior_sd;
  real<lower=0> h0_prior_sd;
  real<lower=0> w1_base_sd;
  real<lower=0> b1_prior_sd;
  real output_log_mean;
  real<lower=0> output_log_sd;
  real<lower=0> gamma_A_prior_sd;
  real nu_minus_two_log_mean;
  real<lower=0> nu_minus_two_log_sd;
}
parameters {
  real mu_std;
  real phi_std;
  real<lower=0> sigma_eta_std;
  real nu_std;

  matrix[D, H] w1_raw;
  vector[H] b1_std;
  ordered[H] log_output_weight_std;
  real gamma_A_std;

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

  matrix[D, H] W1 = (w1_base_sd / sqrt(D)) * w1_raw;
  vector[H] b1 = b1_prior_sd * b1_std;
  vector[H] log_output_weight = output_log_mean
    + output_log_sd * log_output_weight_std;
  vector[H] output_weight;
  matrix[N, H] hidden;
  vector[N] raw_f;
  vector[N] correction_RV;
  real centering_mean;
  real q_centering_mean = mean(q);
  vector[N] q_centered = q - rep_vector(q_centering_mean, N);
  real gamma_A = gamma_A_prior_sd * gamma_A_std;
  vector[N] correction_A = gamma_A * q_centered;
  vector[N] correction;
  vector[N] b;
  vector[N] h;

  for (j in 1:H) {
    output_weight[j] = exp(log_output_weight[j]) / sqrt(H);
  }

  hidden = tanh(X * W1 + rep_matrix(b1', N));
  raw_f = hidden * output_weight;
  centering_mean = mean(raw_f);
  correction_RV = raw_f - rep_vector(centering_mean, N);

  // Setting gamma_A to zero recovers the RV-NN-SV specification.
  correction = correction_RV + correction_A;
  b[1] = mu_prior_mean + h0_prior_sd * h0_raw;
  for (t in 2:N) {
    b[t] = mu + phi * (b[t - 1] - mu)
           + sigma_eta * alpha[t - 1];
  }
  h = b + correction;
}
model {
  mu_std ~ std_normal();
  phi_std ~ std_normal();
  nu_std ~ std_normal();
  sigma_eta_std ~ normal(0, stationary_shrink);

  to_vector(w1_raw) ~ std_normal();
  b1_std ~ std_normal();
  log_output_weight_std ~ std_normal();
  gamma_A_std ~ std_normal();
  h0_raw ~ std_normal();
  alpha ~ std_normal();

  for (t in 1:N) {
    target += standardized_t_logpdf_from_log_sd(z[t], nu, 0.5 * h[t]);
  }
}
generated quantities {
  vector[N] conditional_sd;
  vector[N] conditional_variance;
  vector[N] log_lik;
  real mean_correction = mean(correction);
  real mean_residual_asymmetry_correction = mean(correction_A);
  real phi_gap = 1 - phi;
  for (t in 1:N) {
    conditional_sd[t] = exp(0.5 * h[t]);
    conditional_variance[t] = square(conditional_sd[t]);
    log_lik[t] = standardized_t_logpdf_from_log_sd(z[t], nu, 0.5 * h[t]);
  }
}
