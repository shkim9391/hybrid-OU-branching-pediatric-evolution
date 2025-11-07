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
df_raw <- read_csv("mut_freq_data.csv", show_col_types = FALSE)
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

message("Auto-mapping used -> background: ", colnames(df_raw)[i_background],
        ", replicate: ", colnames(df_raw)[i_replicate],
        ", t: ", colnames(df_raw)[i_t],
        ", x: ", colnames(df_raw)[i_x])
message("Backgrounds found: ", paste(sort(unique(df$background)), collapse = ", "))

# ------------------------------
# 1) OU exact-transition negative log-likelihood
# ------------------------------
nll_ou <- function(par, t, x) {
  mu     <- par[1]
  theta  <- exp(par[2])
  sigma  <- exp(par[3])
  
  ord <- order(t)
  t <- as.numeric(t[ord]); x <- as.numeric(x[ord])
  n <- length(x)
  if (n < 2) return(1e12)
  
  ll <- 0
  for (i in 2:n) {
    dt <- t[i] - t[i-1]
    if (!is.finite(dt) || dt <= 0) next
    e <- exp(-theta * dt)
    m <- mu + (x[i-1] - mu) * e
    v <- (sigma^2 / (2 * theta)) * (1 - e^2)
    v <- max(v, 1e-12)
    ll <- ll + 0.5 * (log(2*pi*v) + (x[i] - m)^2 / v)
  }
  ll
}

fit_ou_group <- function(dfbg) {
  groups <- split(dplyr::arrange(dfbg, t), dfbg$replicate)
  start  <- c(mu = mean(dfbg$x, na.rm = TRUE),
              logtheta = log(0.1),
              logsigma = log(sd(dfbg$x, na.rm = TRUE) + 1e-8))
  fn <- function(par) sum(vapply(groups, function(g) nll_ou(par, g$t, g$x), 0.0))
  opt <- optim(start, fn, method = "L-BFGS-B")
  list(mu = opt$par[1], theta = exp(opt$par[2]), sigma = exp(opt$par[3]), nll = opt$value)
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
# 3) Profile likelihood surfaces (θ, σ) with μ fixed
# ------------------------------
profile_surface_grouped <- function(dfbg, mu_fix, theta_grid, sigma_grid) {
  groups <- split(dplyr::arrange(dfbg, t), dfbg$replicate)
  Z <- matrix(NA_real_, nrow = length(theta_grid), ncol = length(sigma_grid))
  for (i in seq_along(theta_grid)) {
    for (j in seq_along(sigma_grid)) {
      par <- c(mu_fix, log(theta_grid[i]), log(sigma_grid[j]))
      Z[i, j] <- sum(vapply(groups, function(g) nll_ou(par, g$t, g$x), 0.0))
    }
  }
  Z - min(Z, na.rm = TRUE)  # convert to ΔNLL
}

make_grids <- function(fit, span_logs = 3, n = 80) {
  theta <- exp(seq(log(fit$theta) - span_logs, log(fit$theta) + span_logs, length.out = n))
  sigma <- exp(seq(log(fit$sigma) - span_logs, log(fit$sigma) + span_logs, length.out = n))
  list(theta = theta, sigma = sigma)
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
    labs(title = title, x = expression(theta), y = expression(sigma)) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"),
          legend.position = "right")
}

p_wt   <- plot_surface(surf_wt,   fit_wt,   "WT: profile ΔNLL(θ, σ)",   "navy")
p_priA <- plot_surface(surf_priA, fit_priA, "priA: profile ΔNLL(θ, σ)", "darkred")
p_recG <- plot_surface(surf_recG, fit_recG, "recG: profile ΔNLL(θ, σ)", "darkgreen")

fig_vertical <- cowplot::plot_grid(p_wt, p_priA, p_recG, ncol = 1, rel_heights = c(1,1,1))
ggsave("figure4_profile_likelihoods_wt_priA_recG_vertical.png",
       fig_vertical, width = 6.6, height = 12.8, dpi = 1500)

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
message("\nSaved: figure4_profile_likelihoods_wt_priA_recG_vertical.png and AIC_comparison_Fig3_R.csv")

# ------------------------------
# (Optional) also save a horizontal 1×3 layout
# ------------------------------
fig_horizontal <- cowplot::plot_grid(p_wt, p_priA, p_recG, ncol = 3, rel_widths = c(1,1,1))
ggsave("figure4_profile_likelihoods_wt_priA_recG_horizontal.png",
       fig_horizontal, width = 14.0, height = 4.8, dpi = 1500)