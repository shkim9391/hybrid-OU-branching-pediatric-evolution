# ============================================================
# figure4_profile_likelihoods_wt_priA_recG_vertical.R
# ============================================================
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(cowplot)
  library(viridis)
  library(readr)
})

# ------------------------------
# 0) Load data and robust column auto-mapping
# ------------------------------
df_raw <- read_csv(
  here("Revision", "Fig4_Tables", "mut_freq_data_clean.csv"),
  show_col_types = FALSE
)
message("Original columns: ", paste(colnames(df_raw), collapse = ", "))

lname <- tolower(trimws(colnames(df_raw)))
pick_one <- function(exact_opts, pattern_opts) {
  exact_hit <- which(lname %in% exact_opts)
  if (length(exact_hit) > 0) return(exact_hit[1])
  for (pat in pattern_opts) {
    hit <- which(grepl(pat, lname, perl = TRUE))
    if (length(hit) > 0) return(hit[1])
  }
  integer(0)
}

i_background <- pick_one(c("background","strain","group"), c("background|strain|group"))
i_replicate  <- pick_one(c("replicate","rep","sample"),  c("^rep$","replicate","sample"))
i_t          <- pick_one(c("t","time","day","generation"), c("^t$","^time$","^day$","generation"))
i_x          <- pick_one(c("x","freq","frequency","mutation_freq","value"), c("^x$","freq|frequency|mutation|mut|value"))

idx <- c(background = i_background, replicate = i_replicate, t = i_t, x = i_x)
if (any(idx == 0) || any(is.na(idx))) {
  stop("Could not find required columns background/replicate/t/x after auto-mapping.\n",
       "Columns present: ", paste(names(df_raw), collapse = ", "))
}

df <- df_raw[, unname(idx)]
names(df) <- c("background","replicate","t","x")

df <- df %>%
  mutate(
    background = tolower(as.character(background)),
    replicate  = suppressWarnings(as.integer(replicate)),
    t = suppressWarnings(as.numeric(t)),
    x = suppressWarnings(as.numeric(x))
  ) %>%
  tidyr::drop_na(background, replicate, t, x)

# ---- log10-transform with per-background epsilon for zeros ----
df <- df %>%
  group_by(background) %>%
  mutate(
    eps = {
      nz <- x[x > 0 & is.finite(x)]
      if (length(nz) == 0) NA_real_ else 0.5 * min(nz)
    },
    x_adj = ifelse(x > 0, x, eps),
    y = log10(x_adj)
  ) %>%
  ungroup()

if (any(!is.finite(df$y))) {
  stop("Non-finite log10 values produced. Check for a background with all x <= 0.")
}

message("Auto-mapping used -> background: ", colnames(df_raw)[i_background],
        ", replicate: ", colnames(df_raw)[i_replicate],
        ", t: ", colnames(df_raw)[i_t],
        ", x: ", colnames(df_raw)[i_x])
message("Backgrounds found: ", paste(sort(unique(df$background)), collapse = ", "))

# ------------------------------
# 1) OU exact-transition negative log-likelihood
# ------------------------------
nll_ou <- function(par, t, y) {
  mu     <- par[1]
  theta  <- exp(par[2])
  sigma  <- exp(par[3])
  
  ord <- order(t)
  t <- as.numeric(t[ord]); y <- as.numeric(y[ord])
  n <- length(y)
  if (n < 2) return(1e12)
  
  ll <- 0
  for (i in 2:n) {
    dt <- t[i] - t[i-1]
    if (!is.finite(dt) || dt <= 0) next
    e <- exp(-theta * dt)
    m <- mu + (y[i-1] - mu) * e
    v <- (sigma^2 / (2 * theta)) * (1 - e^2)
    v <- max(v, 1e-12)
    ll <- ll + 0.5 * (log(2*pi*v) + (y[i] - m)^2 / v)
  }
  ll
}

fit_ou_group <- function(dfbg, n_restarts = 25, seed = 123) {
  set.seed(seed)
  
  dfbg_ord <- dplyr::arrange(dfbg, replicate, t)
  groups <- split(dfbg_ord, dfbg_ord$replicate)
  
  fn <- function(par) sum(vapply(groups, function(g) nll_ou(par, g$t, g$y), 0.0))
  
  y_all <- dfbg_ord$y
  y_min <- min(y_all, na.rm = TRUE); y_max <- max(y_all, na.rm = TRUE)
  pad <- 2 * (y_max - y_min + 1e-6)
  
  lower <- c(mu = y_min - pad, logtheta = log(1e-6), logsigma = log(1e-6))
  upper <- c(mu = y_max + pad, logtheta = log(1e3),  logsigma = log(1e2))
  
  mu0 <- mean(y_all, na.rm = TRUE)
  theta0 <- 0.5
  sigma0 <- sd(diff(sort(y_all)), na.rm = TRUE)
  if (!is.finite(sigma0) || sigma0 <= 0) sigma0 <- sd(y_all, na.rm = TRUE)
  sigma0 <- max(sigma0, 1e-3)
  
  best <- list(value = Inf, par = c(mu0, log(theta0), log(sigma0)), conv = 999)
  
  for (k in 1:n_restarts) {
    start <- c(
      mu = mu0 + rnorm(1, 0, 0.25 * (y_max - y_min + 1e-6)),
      logtheta = log(theta0) + rnorm(1, 0, 0.75),
      logsigma = log(sigma0) + rnorm(1, 0, 0.75)
    )
    opt <- optim(
      start, fn, method = "L-BFGS-B", lower = lower, upper = upper,
      control = list(maxit = 500)
    )
    if (is.finite(opt$value) && opt$value < best$value) best <- opt
  }
  
  list(mu = best$par[1], theta = exp(best$par[2]), sigma = exp(best$par[3]), nll = best$value)
}

# ------------------------------
# 2) Fit per background: WT, priA, recG (ordered)
# ------------------------------
df_wt   <- df %>% filter(background == "wt")
df_priA <- df %>% filter(background == "pria")
df_recG <- df %>% filter(background == "recg")

if (min(nrow(df_wt), nrow(df_priA), nrow(df_recG)) < 2) {
  stop("Not enough data points after filtering for wt/priA/recG.")
}

fit_wt   <- fit_ou_group(df_wt)
fit_priA <- fit_ou_group(df_priA)
fit_recG <- fit_ou_group(df_recG)

print(fit_wt); print(fit_priA); print(fit_recG)

# ------------------------------
# 2.5) Likelihood ratio test: shared-parameter OU (null) vs lineage-specific OU (alt)
# ------------------------------

# ALT: sum of per-lineage NLLs (already computed)
nll_alt <- fit_wt$nll + fit_priA$nll + fit_recG$nll

# NULL: fit one shared OU across *all* observations.
# IMPORTANT: replicate IDs overlap across backgrounds (e.g., replicate==1 in WT and priA),
# so we must make them unique before splitting into replicate trajectories.
df_all <- df %>%
  filter(background %in% c("wt","pria","recg")) %>%
  mutate(replicate = as.integer(as.factor(interaction(background, replicate, drop = TRUE))))

fit_null <- fit_ou_group(df_all, n_restarts = 50, seed = 123)  # more restarts helps
nll_null <- fit_null$nll

# LRT statistic (same as 2ΔlogL)
lrt <- 2 * (nll_null - nll_alt)
pval <- pchisq(lrt, df = 6, lower.tail = FALSE)

cat(sprintf("NLL_null = %.4f\nNLL_alt = %.4f\nLRT=%.4f, df=6, p=%.3e\n",
            nll_null, nll_alt, lrt, pval))

# ------------------------------
# Table 2: Model comparison (shared OU vs lineage-specific OU)
# ------------------------------
k_null <- 3          # (mu, theta, sigma)
k_alt  <- 9          # 3 lineages × (mu, theta, sigma)

tab2 <- data.frame(
  Model = c("Shared-parameter OU (null)", "Lineage-specific OU (alt)"),
  k = c(k_null, k_alt),
  NLL = c(nll_null, nll_alt),
  AIC = c(2*k_null + 2*nll_null, 2*k_alt + 2*nll_alt)
)

tab2$DeltaAIC <- tab2$AIC - min(tab2$AIC)

# attach LRT stats on the alt row (or keep separate columns)
tab2$LRT_2DeltaLogL <- c(NA, lrt)
tab2$df <- c(NA, 6)
tab2$p_value <- c(NA, pval)

print(tab2)
write.csv(tab2, "Table2_model_comparison_LRT.csv", row.names = FALSE)
message("Saved: Table2_model_comparison_LRT.csv")

# ------------------------------
# 3) Profile likelihood surfaces (θ, σ) with μ fixed
# ------------------------------
profile_surface_grouped <- function(dfbg, mu_fix, theta_grid, sigma_grid) {
  dfbg_ord <- dplyr::arrange(dfbg, replicate, t)
  groups <- split(dfbg_ord, dfbg_ord$replicate)
  
  Z <- matrix(NA_real_, nrow = length(theta_grid), ncol = length(sigma_grid))
  for (i in seq_along(theta_grid)) {
    for (j in seq_along(sigma_grid)) {
      par <- c(mu_fix, log(theta_grid[i]), log(sigma_grid[j]))
      Z[i, j] <- sum(vapply(groups, function(g) nll_ou(par, g$t, g$y), 0.0))
    }
  }
  Z - min(Z, na.rm = TRUE)
}

gr_wt   <- make_grids(fit_wt)
gr_priA <- make_grids(fit_priA)
gr_recG <- make_grids(fit_recG)

Z_wt   <- profile_surface_grouped(df_wt,   fit_wt$mu,   gr_wt$theta,   gr_wt$sigma)
Z_priA <- profile_surface_grouped(df_priA, fit_priA$mu, gr_priA$theta, gr_priA$sigma)
Z_recG <- profile_surface_grouped(df_recG, fit_recG$mu, gr_recG$theta, gr_recG$sigma)

surf_wt <- expand.grid(theta = gr_wt$theta, sigma = gr_wt$sigma) %>%
  mutate(deltaLL = as.vector(Z_wt))
surf_priA <- expand.grid(theta = gr_priA$theta, sigma = gr_priA$sigma) %>%
  mutate(deltaLL = as.vector(Z_priA))
surf_recG <- expand.grid(theta = gr_recG$theta, sigma = gr_recG$sigma) %>%
  mutate(deltaLL = as.vector(Z_recG))

# ------------------------------
# 4) Plot panels with contours (WT top → priA → recG bottom)
# ------------------------------
plot_surface <- function(surf, fit, title, point_color) {
  levels <- c(1.15, 3.0, 6.0, 10.0)  # ~68%, ~95%, wider
  ggplot(surf, aes(theta, sigma)) +
    geom_tile(aes(fill = deltaLL)) +
    geom_contour(aes(z = deltaLL), breaks = levels, colour = "black", linewidth = 0.5) +
    geom_point(aes(x = fit$theta, y = fit$sigma), inherit.aes = FALSE,
               colour = point_color, size = 2.6, stroke = 0.3) +
    scale_x_log10() + scale_y_log10() +
    scale_fill_viridis(option = "C", direction = -1, name = expression(Delta*"NLL")) +
    labs(title = title, x = expression(theta), y = expression(sigma~~"(log10 scale)")) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"),
          legend.position = "right")
}

p_wt   <- plot_surface(surf_wt,   fit_wt,   "A. WT: profile ΔNLL(θ, σ)",   "navy")
p_priA <- plot_surface(surf_priA, fit_priA, "B. priA: profile ΔNLL(θ, σ)", "darkred")
p_recG <- plot_surface(surf_recG, fit_recG, "C. recG: profile ΔNLL(θ, σ)", "darkgreen")

fig_vertical <- cowplot::plot_grid(p_wt, p_priA, p_recG, ncol = 1, rel_heights = c(1,1,1))
ggsave("figure4_profile_likelihoods_wt_priA_recG_vertical.png",
       fig_vertical, width = 6.6, height = 12.8, dpi = 600)

# ------------------------------
# 5) AIC comparison table (WT / priA / recG)
# ------------------------------
k <- 3
aic_tbl <- tibble::tibble(
  Background = c("WT","priA","recG"),
  Mu    = c(fit_wt$mu,   fit_priA$mu,   fit_recG$mu),
  Theta = c(fit_wt$theta,fit_priA$theta,fit_recG$theta),
  Sigma = c(fit_wt$sigma,fit_priA$sigma,fit_recG$sigma),
  NLL   = c(fit_wt$nll,  fit_priA$nll,  fit_recG$nll)
) %>%
  mutate(AIC = 2*k + 2*NLL) %>%
  arrange(AIC) %>%
  mutate(DeltaAIC = AIC - min(AIC))

message("\n===== AIC Comparison (Figure 4) =====")
print(aic_tbl)
write_csv(aic_tbl, "AIC_comparison_Fig4_R.csv")
message("\nSaved: figure4_profile_likelihoods_wt_priA_recG_vertical.png and AIC_comparison_Fig4_R.csv")

# ------------------------------
# (Optional) also save a horizontal 1×3 layout
# ------------------------------
fig_horizontal <- cowplot::plot_grid(p_wt, p_priA, p_recG, ncol = 3, rel_widths = c(1,1,1))
ggsave("/Revision/Fig4_Tables/figure4_profile_likelihoods_wt_priA_recG_horizontal.png",
       fig_horizontal, width = 14.0, height = 4.8, dpi = 600)

write_csv(aic_tbl, "/Revision/Fig4_Tables/AIC_comparison_Fig4_R.csv")
print(fit_priA)

# ------------------------------
# 6) Grouped bootstrap Table 1 (matches Figure 4 grouped likelihood)
# ------------------------------
set.seed(123)

bootstrap_grouped <- function(dfbg, B = 2000) {
  reps <- sort(unique(dfbg$replicate))
  if (length(reps) < 2) stop("Need >=2 replicates for bootstrap.")
  
  # MLE on original grouped data
  mle <- fit_ou_group(dfbg, n_restarts = 25, seed = 123)
  
  # Bootstrap over replicates
  boot <- matrix(NA_real_, nrow = B, ncol = 3)
  colnames(boot) <- c("mu", "theta", "sigma")
  
  for (b in 1:B) {
    draw <- sample(reps, size = length(reps), replace = TRUE)
    
    # duplicate replicate trajectories with unique replicate IDs
    boot_df <- do.call(rbind, lapply(seq_along(draw), function(k) {
      sub <- dfbg[dfbg$replicate == draw[k], ]
      sub$replicate <- 1e7 + b*1e3 + k
      sub
    }))
    
    fitb <- fit_ou_group(boot_df, n_restarts = 10, seed = 1000 + b)
    boot[b, ] <- c(fitb$mu, fitb$theta, fitb$sigma)
  }
  
  boot_df <- as.data.frame(boot)
  
  # CIs
  qs <- apply(boot_df, 2, quantile, probs = c(0.025, 0.5, 0.975), na.rm = TRUE)
  
  list(
    mle = mle,
    boot = boot_df,
    ci = qs,
    corr_logtheta_logsigma = cor(log(boot_df$theta), log(boot_df$sigma), use="complete.obs")
  )
}

# Run for each lineage
res_wt   <- bootstrap_grouped(df_wt,   B = 2000)
res_priA <- bootstrap_grouped(df_priA, B = 2000)
res_recG <- bootstrap_grouped(df_recG, B = 2000)

# Build summary table
make_row <- function(name, res) {
  data.frame(
    genotype = name,
    bootstrap_method = "nonparam_replicate_grouped",
    n_boot_kept = nrow(res$boot),
    mu_mle = res$mle$mu,
    mu_ci025 = res$ci["2.5%", "mu"],
    mu_ci975 = res$ci["97.5%", "mu"],
    theta_mle = res$mle$theta,
    theta_ci025 = res$ci["2.5%", "theta"],
    theta_ci975 = res$ci["97.5%", "theta"],
    sigma_mle = res$mle$sigma,
    sigma_ci025 = res$ci["2.5%", "sigma"],
    sigma_ci975 = res$ci["97.5%", "sigma"],
    corr_logtheta_logsigma = res$corr_logtheta_logsigma
  )
}

tab1 <- rbind(
  make_row("wt", res_wt),
  make_row("priA", res_priA),
  make_row("recG", res_recG)
)

write.csv(tab1, "Table1_grouped_bootstrap_summary.csv", row.names = FALSE)
write.csv(
  rbind(
    transform(res_wt$boot, genotype="wt"),
    transform(res_priA$boot, genotype="priA"),
    transform(res_recG$boot, genotype="recG")
  ),
  "Table1_grouped_bootstrap_samples.csv",
  row.names = FALSE
)

# ------------------------------
# Enhanced Table 1: add derived quantities from bootstrap
# ------------------------------
summ_ci <- function(x) {
  q <- quantile(x, probs = c(0.025, 0.975), na.rm = TRUE)
  c(ci025 = unname(q[1]), ci975 = unname(q[2]))
}

make_row2 <- function(name, res) {
  boot <- res$boot %>%
    mutate(
      eq_freq = 10^mu,                        # equilibrium on original scale
      stat_var = (sigma^2) / (2*theta),       # stationary variance on log10 scale
      half_life = log(2) / theta              # mean-reversion half-life (time units of t)
    )
  
  mle_eq <- 10^(res$mle$mu)
  mle_statvar <- (res$mle$sigma^2) / (2*res$mle$theta)
  mle_halflife <- log(2) / res$mle$theta
  
  c_mu <- summ_ci(boot$mu)
  c_th <- summ_ci(boot$theta)
  c_sg <- summ_ci(boot$sigma)
  c_eq <- summ_ci(boot$eq_freq)
  c_sv <- summ_ci(boot$stat_var)
  c_hl <- summ_ci(boot$half_life)
  
  data.frame(
    genotype = name,
    bootstrap_method = "nonparam_replicate_grouped",
    n_boot_kept = nrow(boot),
    
    mu_mle = res$mle$mu, mu_ci025 = c_mu["ci025"], mu_ci975 = c_mu["ci975"],
    theta_mle = res$mle$theta, theta_ci025 = c_th["ci025"], theta_ci975 = c_th["ci975"],
    sigma_mle = res$mle$sigma, sigma_ci025 = c_sg["ci025"], sigma_ci975 = c_sg["ci975"],
    
    eq_freq_mle = mle_eq, eq_freq_ci025 = c_eq["ci025"], eq_freq_ci975 = c_eq["ci975"],
    stat_var_mle = mle_statvar, stat_var_ci025 = c_sv["ci025"], stat_var_ci975 = c_sv["ci975"],
    half_life_mle = mle_halflife, half_life_ci025 = c_hl["ci025"], half_life_ci975 = c_hl["ci975"],
    
    corr_logtheta_logsigma = res$corr_logtheta_logsigma
  )
}

tab1_new <- rbind(
  make_row2("wt", res_wt),
  make_row2("priA", res_priA),
  make_row2("recG", res_recG)
)

write.csv(tab1_new, "Table1_grouped_bootstrap_summary.csv", row.names = FALSE)
message("Updated: Table1_grouped_bootstrap_summary.csv")
print(tab1_new)

