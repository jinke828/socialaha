library(mediation)
library(broom)

df <- read.csv("/Users/jinke/Desktop/socialaha_github/results/IS-RSA/mediation.csv")

roi_cols <- paste0("brain_sim_", 0:115)

results <- list()        # store full mediation objects (optional)
results_tbl <- data.frame()  # store tidy summary

for (roi in roi_cols) {
  
  vars <- c("thought_before", "thought_after", roi, "scene", "pair", "condition")
  df_use <- df[complete.cases(df[, vars]), vars]
  
  # (Recommended) make categorical controls factors if they are IDs / discrete groups
  df_use$scene <- factor(df_use$scene)
  df_use$pair <- factor(df_use$pair)
  df_use$condition <- factor(df_use$condition)
  
  # mediator model: M ~ T + controls
  m_model <- lm(reformulate(c("thought_before", "scene", "pair", "condition"), response = roi),
                data = df_use)
  
  # outcome model: Y ~ T + M + controls
  y_model <- lm(as.formula(paste0("thought_after ~ thought_before + ", roi,
                                  " + scene + pair + condition")),
                data = df_use)
  
  # Mediation
  med_out <- mediate(
    model.m = m_model,
    model.y = y_model,
    treat = "thought_before",
    mediator = roi,
    boot = TRUE,
    sims = 1000
  )
  
  # ---- Print summary (as requested) ----
  cat("\n==============================\n")
  cat("ROI:", roi, "\n")
  cat("==============================\n")
  print(summary(med_out))
  
  # ---- Save full object (optional but recommended) ----
  results[[roi]] <- med_out
  
  # ---- Extract tidy stats ----
  results_tbl <- rbind(
    results_tbl,
    data.frame(
      roi = roi,
      ACME = med_out$d0,
      ACME_p = med_out$d0.p,
      ACME_ci_low = med_out$d0.ci[1],
      ACME_ci_high = med_out$d0.ci[2],
      ADE = med_out$z0,
      ADE_p = med_out$z0.p,
      Total = med_out$tau.coef,
      Total_p = med_out$tau.p,
      Prop_Mediated = med_out$n0,
      Prop_Mediated_p = med_out$n0.p,
      N = nrow(df_use)
    )
  )
}

write.csv(
  results_tbl,
  "/Users/jinke/Downloads/mediation_results_all_rois.csv",
  row.names = FALSE
)

saveRDS(
  results,
  "/Users/jinke/Downloads/mediation_objects_all_rois.rds"
)
