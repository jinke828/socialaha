# Load required package
library(lme4)
library(ggplot2)
library(dplyr)

# Load the CSV
df <- read.csv("/Users/jinke/Desktop/socialaha_github/results/impression_updates/df.csv")

# Convert grouping variables to factors
df$group <- as.factor(df$group)
df$character <- as.factor(df$character)
df$sub <- as.factor(df$sub)

# Remove rows with missing similarity values
df <- na.omit(df)

# Nested mixed-effects model:
# similarity ~ distance + (1 | group/sub) + (1 | character)
# where sub is nested within group, and character is an additional random effect
df <- df %>%
  mutate(
    distance_z = scale(distance),
    similarity_z = scale(similarity)
  )

model <- lmer(similarity_z ~ distance_z + (1 | group/sub) + (1 | character), data = df)
summary(model)

# Summarize results
summary(model)


# plot
# Make distance discrete on the x-axis (keeps numeric order)
df <- df %>% mutate(distance_f = factor(distance, levels = sort(unique(distance))))
df_summary <- df %>%
  group_by(distance) %>%
  summarise(
    mean_sim = mean(similarity, na.rm = TRUE),
    se_sim = sd(similarity, na.rm = TRUE)/sqrt(n())
  )

ggplot(df_summary, aes(x = distance, y = mean_sim)) +
  geom_point(size = 4) +
  geom_line(size = 1.5) +
  geom_errorbar(aes(ymin = mean_sim - se_sim, ymax = mean_sim + se_sim),size = 1, width = 0.2) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid = element_blank(),      # remove grid
    axis.line = element_line(size = 1, colour = "black")  # add axes
  ) +
  labs(x = "Distance", y = "Mean Similarity")
