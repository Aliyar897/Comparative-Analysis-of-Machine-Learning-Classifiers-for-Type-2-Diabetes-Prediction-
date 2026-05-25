# COM747 Data Science & Machine Learning  Component 1
# Integrated Group Project Code, All Four Members


install.packages(c("tidyverse", "caret", "randomForest", "e1071",
                    "pROC", "ROSE", "cluster", "factoextra", "gridExtra",
                    "scales", "Metrics", "corrplot","gridExtra", "nortest", "car",
                    "reshape2"))


library(gridExtra)
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(ROSE)
library(cluster)
library(factoextra)
library(gridExtra)
library(scales)
library(Metrics)
library(corrplot)
library(nortest)
library(car)
library(reshape2)
library(ggplot2)



# MEMBER 1 Dataset Loading, Cleaning, EDA & Visualisations

setwd("C:/Users/USER/OneDrive - Ulster University/DSM Group Project")
getwd()

#  Load dataset
diabetes <- read.csv("diabetes.csv", stringsAsFactors = FALSE)

# Initial inspection
head(diabetes)
str(diabetes)
dim(diabetes)
summary(diabetes)

#Convert column Outcome to factor
diabetes$Outcome <- as.factor(diabetes$Outcome)
levels(diabetes$Outcome) <- c("No_Diabetes", "Diabetes")
table(diabetes$Outcome)
prop.table(table(diabetes$Outcome))

# Data cleaning and preprocessing
# Replaced zero 0 values with NA (missing values),
# then use median imputation to fill in missing data.
# Median is used because it is less affected by outliers.
diabetes$Glucose[diabetes$Glucose == 0] <- NA
diabetes$BloodPressure[diabetes$BloodPressure == 0] <- NA
diabetes$SkinThickness[diabetes$SkinThickness == 0] <- NA
diabetes$Insulin[diabetes$Insulin == 0] <- NA
diabetes$BMI[diabetes$BMI == 0] <- NA
colSums(is.na(diabetes))

#Using median imputation to replace the missing values(NA)
diabetes$Glucose[is.na(diabetes$Glucose)]<- median(diabetes$Glucose, na.rm = TRUE)
diabetes$BloodPressure[is.na(diabetes$BloodPressure)] <- median(diabetes$BloodPressure, na.rm = TRUE)
diabetes$SkinThickness[is.na(diabetes$SkinThickness)] <- median(diabetes$SkinThickness, na.rm = TRUE)
diabetes$Insulin[is.na(diabetes$Insulin)]<- median(diabetes$Insulin, na.rm = TRUE)
diabetes$BMI[is.na(diabetes$BMI)]<- median(diabetes$BMI, na.rm = TRUE)

colSums(is.na(diabetes))
dim(diabetes)
summary(diabetes)

# Descriptive statistics
# Glucose
mean(diabetes$Glucose)
sd(diabetes$Glucose)
range(diabetes$Glucose)
quantile(diabetes$Glucose)

# BMI
mean(diabetes$BMI)
sd(diabetes$BMI)
range(diabetes$BMI)
quantile(diabetes$BMI)

# Age
mean(diabetes$Age)
sd(diabetes$Age)
range(diabetes$Age)
quantile(diabetes$Age)

# Blood Pressure
mean(diabetes$BloodPressure)
sd(diabetes$BloodPressure)
range(diabetes$BloodPressure)

# Group means by Outcome
aggregate(Glucose ~ Outcome, data = diabetes, mean)
aggregate(BMI ~ Outcome, data = diabetes, mean)
aggregate(Age ~ Outcome, data = diabetes, mean)
aggregate(BloodPressure ~ Outcome, data = diabetes, mean)

# EDA Visualisations
# Histograms
hist(diabetes$Glucose,main = "Histogram of Glucose",xlab = "Glucose")
hist(diabetes$BMI,    main = "Histogram of BMI",    xlab = "BMI")
hist(diabetes$Age,    main = "Histogram of Age",    xlab = "Age")

# Boxplots (overall)
boxplot(diabetes$Glucose,main = "Boxplot of Glucose")
boxplot(diabetes$BMI,   main = "Boxplot of BMI")
boxplot(diabetes$Age,   main = "Boxplot of Age")

# Boxplots by Outcome
boxplot(Glucose ~ Outcome,data = diabetes, main = "Glucose by Outcome")
boxplot(BMI ~ Outcome,    data = diabetes, main = "BMI by Outcome")
boxplot(Age ~ Outcome,    data = diabetes, main = "Age by Outcome")

# Scatterplots
plot(diabetes$Glucose, diabetes$BMI,
     col  = diabetes$Outcome,
     main = "Glucose vs BMI",
     xlab = "Glucose", ylab = "BMI")

plot(diabetes$Age, diabetes$Glucose,
     col  = diabetes$Outcome,
     main = "Age vs Glucose",
     xlab = "Age", ylab = "Glucose")

# Correlation analysis & heatmap
numericData <- diabetes[, c("Pregnancies", "Glucose", "BloodPressure",
                             "SkinThickness", "Insulin", "BMI",
                             "DiabetesPedigreeFunction", "Age")]
cor_matrix_m1 <- cor(numericData)
print(cor_matrix_m1)

cor_melt <- melt(cor_matrix_m1)

ggplot(cor_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1)) +
  labs(title = "Correlation Heatmap", x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# MEMBER 2 — Feature Engineering, Statistical Analysis & Data Splitting


# NOTE: Data is already loaded and cleaned by Member 1.
# This section adds new features, selects predictors, splits and scales.

# Feature engineering
diabetes$BMI_Category <- cut(diabetes$BMI,
                              breaks = c(-Inf, 18.5, 24.9, 29.9, Inf),
                              labels = c("Underweight", "Normal", "Overweight", "Obese"),
                              include.lowest = TRUE)

diabetes$Age_Group <- cut(diabetes$Age,
                           breaks = c(20, 30, 40, 50, Inf),
                           labels = c("20-30", "31-40", "41-50", "50+"),
                           include.lowest = TRUE)

diabetes$Glucose_Category <- cut(diabetes$Glucose,
                                  breaks = c(-Inf, 99, 125, Inf),
                                  labels = c("Normal", "Pre-diabetic", "Diabetic"),
                                  include.lowest = TRUE)

diabetes$Insulin_Glucose_Ratio <- diabetes$Insulin / (diabetes$Glucose + 1)
diabetes$High_Pregnancy_Risk   <- ifelse(diabetes$Pregnancies >= 4, 1, 0)

# Fix any Inf / NaN from ratio
diabetes$Insulin_Glucose_Ratio[!is.finite(diabetes$Insulin_Glucose_Ratio)] <- NA
diabetes$Insulin_Glucose_Ratio[is.na(diabetes$Insulin_Glucose_Ratio)] <-
  median(diabetes$Insulin_Glucose_Ratio, na.rm = TRUE)

# Numeric column set
numeric_cols <- names(diabetes)[sapply(diabetes, is.numeric)]
numeric_cols <- setdiff(numeric_cols, "Outcome")

# One-hot encoding
df_encoded <- diabetes

df_encoded$BMI_Category     <- addNA(df_encoded$BMI_Category)
df_encoded$Age_Group        <- addNA(df_encoded$Age_Group)
df_encoded$Glucose_Category <- addNA(df_encoded$Glucose_Category)

dummies <- model.matrix(~ BMI_Category + Age_Group + Glucose_Category - 1,
                         data = df_encoded)
stopifnot(nrow(dummies) == nrow(df_encoded))

df_onehot <- cbind(
  df_encoded[numeric_cols],
  dummies,
  df_encoded[c("Insulin_Glucose_Ratio", "High_Pregnancy_Risk")],
  Outcome = df_encoded$Outcome
)

# Correlation & RFE feature selection
cor_matrix <- cor(diabetes[numeric_cols], method = "pearson", use = "complete.obs")
high_cor   <- findCorrelation(cor_matrix, cutoff = 0.85, names = TRUE)
cat("Highly correlated features (>0.85):", paste(high_cor, collapse = ", "), "\n")

# Clean numeric data for RFE
numeric_df <- diabetes[, c(numeric_cols, "Outcome")]
numeric_df[numeric_cols] <- lapply(numeric_df[numeric_cols], function(x) {
  x[!is.finite(x)] <- NA
  x[is.na(x)]      <- median(x, na.rm = TRUE)
  x
})
stopifnot(sum(is.na(numeric_df[numeric_cols])) == 0)

# RFE uses its own control object (rfe_ctrl) — does not conflict with Member 3
rfe_ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)

set.seed(42)
rfe_result <- rfe(
  x        = numeric_df[, numeric_cols],
  y        = numeric_df$Outcome,
  sizes    = c(3, 5, 7, 8),
  rfeControl = rfe_ctrl
)

# Create 'output' folder if it doesn't exist
if (!dir.exists("output")) {
  dir.create("output")
}
# Save the plot
png("output/rfe_feature_selection.png", width = 700, height = 500)
plot(rfe_result, type = c("g", "o"))
dev.off()

# Build model dataset from RFE-selected features
selected_features <- predictors(rfe_result)
cat("RFE selected features:", paste(selected_features, collapse = ", "), "\n")

df_model <- diabetes[, c(selected_features, "Outcome")]
df_model$Insulin_Glucose_Ratio <- diabetes$Insulin_Glucose_Ratio
df_model$High_Pregnancy_Risk   <- diabetes$High_Pregnancy_Risk

# Statistical analysis hypothesis tests
cat("\n--- Welch t-tests: Diabetes vs No_Diabetes ---\n")
for (col in selected_features) {
  if (is.numeric(diabetes[[col]])) {
    g1 <- diabetes[[col]][diabetes$Outcome == "Diabetes"]
    g2 <- diabetes[[col]][diabetes$Outcome == "No_Diabetes"]
    tt <- t.test(g1, g2)
    cat(sprintf("  %-28s p = %.4f %s\n", col, tt$p.value,
                ifelse(tt$p.value < 0.05, "***", "")))
  }
}

# Train / Validation / Test split (60 / 20 / 20) 
set.seed(42)

train_index <- createDataPartition(df_model$Outcome, p = 0.80, list = FALSE)
train_data  <- df_model[ train_index, ]
test_data   <- df_model[-train_index, ]

val_index  <- createDataPartition(train_data$Outcome, p = 0.80, list = FALSE)
val_data   <- train_data[-val_index, ]
train_data <- train_data[ val_index, ]

cat(sprintf("\nSplit — Train: %d | Val: %d | Test: %d\n",
            nrow(train_data), nrow(val_data), nrow(test_data)))

# Z-score scaling fit on train only — no data leakage
num_pred <- setdiff(names(df_model), "Outcome")

preproc      <- preProcess(train_data[, num_pred], method = c("center", "scale"))
train_scaled <- predict(preproc, train_data)
val_scaled   <- predict(preproc, val_data)
test_scaled  <- predict(preproc, test_data)

saveRDS(preproc, "preprocessor.rds")


# Create 'data' folder if it doesn't exist
if (!dir.exists("data")) {
  dir.create("data")
}
# Save files inside the data folder
write.csv(train_scaled, "data/train_data.csv",      row.names = FALSE)
write.csv(val_scaled,   "data/validation_data.csv", row.names = FALSE)
write.csv(test_scaled,  "data/test_data.csv",       row.names = FALSE)

cat("\nSaved: train_data.csv | validation_data.csv | test_data.csv | preprocessor.rds\n")


# Member 2 visualisations
# Feature distributions by outcome (single pivot_longer — no duplicate)
numeric_long <- tidyr::pivot_longer(
  diabetes,
  cols      = any_of(numeric_cols),
  names_to  = "Feature",
  values_to = "Value"
)

p_box <- ggplot(numeric_long, aes(x = Outcome, y = Value, fill = Outcome)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~ Feature, scales = "free_y") +
  scale_fill_manual(values = c("No_Diabetes" = "#2196F3", "Diabetes" = "#F44336")) +
  labs(title = "Feature Distributions by Diabetes Outcome",
       x = "Outcome", y = "Value") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

print(p_box)
ggsave("output/feature_distributions_by_outcome.png", p_box, width = 12, height = 8, dpi = 150)

p_glucose <- ggplot(diabetes, aes(x = Glucose_Category, fill = Outcome)) +
  geom_bar(position = "fill", alpha = 0.85) +
  scale_fill_manual(values = c("No_Diabetes" = "#42A5F5", "Diabetes" = "#EF5350")) +
  labs(title = "Glucose Category vs Diabetes Outcome",
       x = "Glucose Category", y = "Proportion", fill = "Outcome") +
  theme_minimal(base_size = 12)

print(p_glucose)
ggsave("output/glucose_category_vs_outcome.png", p_glucose, width = 8, height = 5, dpi = 150)

p_bmi <- ggplot(diabetes, aes(x = BMI_Category, fill = Outcome)) +
  geom_bar(position = "fill", alpha = 0.85) +
  scale_fill_manual(values = c("No_Diabetes" = "#66BB6A", "Diabetes" = "#FF7043")) +
  labs(title = "BMI Category vs Diabetes Outcome",
       x = "BMI Category", y = "Proportion", fill = "Outcome") +
  theme_minimal(base_size = 12)

print(p_bmi)
ggsave("output/bmi_category_vs_outcome.png", p_bmi, width = 8, height = 5, dpi = 150)

cat("\n=== Member 2 complete ===\n")



# MEMBER 3 — ML Models, Hyperparameter Tuning, Evaluation & Benchmarking

# NOTE: train_scaled / val_scaled / test_scaled are already in memory from
#       Member 2. No need to re-read the CSV files.

# Outcome factor — enforce consistent two-level encoding
CLF_POS    <- "Diabetes"
CLF_NEG    <- "No_Diabetes"
CLF_LEVELS <- c(CLF_NEG, CLF_POS)

fix_outcome <- function(df) {
  df$Outcome <- factor(as.character(df$Outcome), levels = CLF_LEVELS)
  df
}


train_scaled <- fix_outcome(train_scaled)
val_scaled   <- fix_outcome(val_scaled)
test_scaled  <- fix_outcome(test_scaled)


stopifnot(
  nlevels(train_scaled$Outcome) == 2,
  nlevels(test_scaled$Outcome)  == 2
)

cat("Train class distribution:\n")
print(round(prop.table(table(train_scaled$Outcome)) * 100, 1))
cat("Test class distribution:\n")
print(round(prop.table(table(test_scaled$Outcome)) * 100, 1))

# Feature / label matrices
X_train <- train_scaled %>% select(-Outcome)
y_train <- train_scaled$Outcome

X_test  <- test_scaled  %>% select(-Outcome)
y_test  <- test_scaled$Outcome
y_test
# Class imbalance check — apply ROSE only if needed
ratio <- min(table(y_train)) / max(table(y_train))
cat(sprintf("Train minority/majority ratio: %.3f\n", ratio))

## ROSE Implementation is through LLM but the logic flow is mine
if (ratio < 0.70) {
  cat("  → Applying ROSE oversampling to training set.\n")
  set.seed(42)
  train_balanced    <- ROSE(Outcome ~ ., data = train_scaled, seed = 42)$data
  train_balanced$Outcome <- factor(as.character(train_balanced$Outcome),
                                    levels = CLF_LEVELS)
  X_train <- train_balanced %>% select(-Outcome)
  y_train <- train_balanced$Outcome
  cat("  Balanced distribution:\n")
  print(round(prop.table(table(y_train)) * 100, 1))
} else {
  cat("  → Class balance acceptable. No oversampling needed.\n")
}

# Cross-validation control clf_ctrl — distinct from rfe_ctrl
clf_ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter     = FALSE
)

# Train five classification models
set.seed(42)

## LLms are used only for selecting best hyperparamterters
models <- list(
  
  Logistic = train(X_train, y_train,
                   method    = "glm",
                   family    = "binomial",
                   metric    = "ROC",
                   trControl = clf_ctrl),
  
  RandomForest = train(X_train, y_train,
                       method     = "rf",
                       metric     = "ROC",
                       tuneGrid   = data.frame(mtry = c(2, 3, 4, 5)),
                       ntree      = 300,
                       importance = TRUE,
                       trControl  = clf_ctrl),
  
  SVM = train(X_train, y_train,
              method    = "svmRadial",
              metric    = "ROC",
              tuneGrid  = expand.grid(
                C     = c(0.1, 0.5, 1, 2, 5),
                sigma = c(0.01, 0.05, 0.1, 0.5)),
              trControl = clf_ctrl),
  
  KNN = train(X_train, y_train,
              method    = "knn",
              metric    = "ROC",
              tuneGrid  = data.frame(k = seq(3, 25, by = 2)),
              trControl = clf_ctrl)
)

# Evaluation function
## a bit of code is written and debugged using LLM
evaluate_model <- function(model, X, y, name) {
  y     <- factor(as.character(y), levels = CLF_LEVELS)
  preds <- factor(as.character(predict(model, X)), levels = CLF_LEVELS)

  prob_df  <- predict(model, X, type = "prob")
  if (!CLF_POS %in% colnames(prob_df))
    stop(paste("Missing probability column:", CLF_POS))
  prob_pos <- prob_df[[CLF_POS]]

  cm      <- confusionMatrix(preds, y, positive = CLF_POS)
  roc_obj <- pROC::roc(response  = as.numeric(y == CLF_POS),
                        predictor = prob_pos,
                        quiet     = TRUE)

  # Compute Specificity from confusion table
  tbl         <- cm$table
  TN          <- tbl[1, 1]; FP <- tbl[1, 2]
  specificity <- TN / (TN + FP)

  list(
    name        = name,
    cm          = cm,
    roc         = roc_obj,
    accuracy    = as.numeric(cm$overall["Accuracy"]),
    precision   = as.numeric(cm$byClass["Precision"]),
    recall      = as.numeric(cm$byClass["Recall"]),
    specificity = specificity,
    f1          = as.numeric(cm$byClass["F1"]),
    auc         = as.numeric(pROC::auc(roc_obj))
  )
}

# Evaluate all models on test set
results <- list()

for (m in names(models)) {
  cat("\nEvaluating:", m, "\n")
  results[[m]] <- tryCatch(
    evaluate_model(models[[m]], X_test, y_test, m),
    error = function(e) {
      cat("FAILED:", m, "->", e$message, "\n")
      NULL
    }
  )
}

results
results <- results[!sapply(results, is.null)]
cat(sprintf("\n%d / %d models evaluated successfully.\n",
            length(results), length(models)))

# Classification benchmarking table
clf_benchmark <- bind_rows(lapply(results, function(r) {
  data.frame(
    Model       = r$name,
    Accuracy    = round(r$accuracy,    4),
    Precision   = round(r$precision,   4),
    Recall      = round(r$recall,      4),
    Specificity = round(r$specificity, 4),
    F1_Score    = round(r$f1,          4),
    AUC         = round(r$auc,         4)
  )
}))

cat("   CLASSIFICATION BENCHMARKING TABLE  (Test Set)     \n")
print(clf_benchmark, row.names = FALSE)

best_f1_idx  <- which.max(clf_benchmark$F1_Score)
best_auc_idx <- which.max(clf_benchmark$AUC)
cat(sprintf("\n  Best by F1  : %s (F1 = %.4f)\n",
            clf_benchmark$Model[best_f1_idx],  clf_benchmark$F1_Score[best_f1_idx]))
cat(sprintf("  Best by AUC : %s (AUC = %.4f)\n",
            clf_benchmark$Model[best_auc_idx], clf_benchmark$AUC[best_auc_idx]))

# Benchmarking bar chart
bench_long <- clf_benchmark %>%
  select(Model, Accuracy, Precision, Recall, Specificity, F1_Score, AUC) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Score")

bench_long$Model <- factor(bench_long$Model,
                            levels = rev(clf_benchmark$Model[order(clf_benchmark$F1_Score)]))
## took help for creating graphs
p_bench <- ggplot(bench_long, aes(x = Model, y = Score, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7, alpha = 0.88) +
  geom_hline(yintercept = 0.5, linetype = "dashed",
             colour = "grey40", linewidth = 0.6) +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.05)) +
  coord_flip() +
  labs(title    = "Classification Model Benchmarking — All Metrics (Test Set)",
       subtitle = "Models ranked by F1 Score (top = best)",
       x = NULL, y = "Score", fill = "Metric") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

print(p_bench)
# Force white background
p_bench_white <- p_bench +
  theme(
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA)
  )

# Save with white background
ggsave(
  filename = "output/benchmark_plot.png",
  plot     = p_bench_white,
  width    = 8,
  height   = 5,
  dpi      = 300,
  bg       = "white"
)

# Precision–Recall trade-off table
cat("\n--- Precision–Recall Trade-off ---\n")
cat("Note: HIGH RECALL is prioritised — missed diagnosis is clinically more costly.\n\n")

pr_table <- clf_benchmark %>%
  select(Model, Precision, Recall, F1_Score) %>%
  mutate(Assessment = case_when(
    Recall >= 0.75 & Precision >= 0.70 ~ "Balanced",
    Recall >= 0.75                      ~ "High Recall / Low Precision",
    Precision >= 0.70                   ~ "High Precision / Low Recall",
    TRUE                                ~ "Needs Improvement"
  ))
print(pr_table, row.names = FALSE)

# Regression models
# Target: predict continuous Glucose (mg/dL) from other features.
# We use the already-cleaned `diabetes` object — no need to re-read the CSV.

cat("\n=== Regression: predicting Glucose (mg/dL) ===\n")

reg_features <- setdiff(names(diabetes), c("Glucose", "Outcome",
                                             "BMI_Category", "Age_Group",
                                             "Glucose_Category"))
df_reg <- diabetes[, c(reg_features, "Glucose")]

set.seed(42)
reg_idx      <- createDataPartition(df_reg$Glucose, p = 0.80, list = FALSE)
df_reg_train <- df_reg[ reg_idx, ]
df_reg_test  <- df_reg[-reg_idx, ]

X_reg_train <- df_reg_train %>% select(-Glucose)
y_reg_train <- df_reg_train$Glucose
X_reg_test  <- df_reg_test  %>% select(-Glucose)
y_reg_test  <- df_reg_test$Glucose

ctrl_reg <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

# Linear Regression
set.seed(42)
model_lm <- train(x = X_reg_train, y = y_reg_train,
                  method = "lm", trControl = ctrl_reg)

pred_lm <- predict(model_lm, X_reg_test)
rmse_lm <- round(rmse(y_reg_test, pred_lm), 3)
mae_lm  <- round(mae(y_reg_test,  pred_lm), 3)
r2_lm   <- round(cor(y_reg_test,  pred_lm)^2, 4)
cat(sprintf("Linear Regression  — RMSE: %.3f | MAE: %.3f | R²: %.4f\n",
            rmse_lm, mae_lm, r2_lm))
cat("Coefficients:\n")
print(summary(model_lm$finalModel)$coefficients)

# Random Forest Regressor
set.seed(42)
model_rf_reg <- train(
  x = X_reg_train, y = y_reg_train,
  method    = "rf",
  tuneGrid  = expand.grid(mtry = c(2, 3, 4, 5)),
  trControl = ctrl_reg,
  ntree     = 300
)
pred_rf_reg  <- predict(model_rf_reg, X_reg_test)
rmse_rf_reg  <- round(rmse(y_reg_test, pred_rf_reg), 3)
mae_rf_reg   <- round(mae(y_reg_test,  pred_rf_reg), 3)
r2_rf_reg    <- round(cor(y_reg_test,  pred_rf_reg)^2, 4)
cat(sprintf("Random Forest Reg. — Best mtry: %d | RMSE: %.3f | MAE: %.3f | R²: %.4f\n",
            model_rf_reg$bestTune$mtry, rmse_rf_reg, mae_rf_reg, r2_rf_reg))

# Regression benchmark
reg_benchmark <- data.frame(
  Model     = c("Linear Regression", "Random Forest Regressor"),
  RMSE      = c(rmse_lm,  rmse_rf_reg),
  MAE       = c(mae_lm,   mae_rf_reg),
  R_Squared = c(r2_lm,    r2_rf_reg)
)
cat("\n--- Regression Benchmarking Table ---\n")
print(reg_benchmark, row.names = FALSE)
cat(sprintf("Best (lowest RMSE): %s\n",
            reg_benchmark$Model[which.min(reg_benchmark$RMSE)]))

# Regression visualisations
reg_plot_df <- rbind(
  data.frame(Actual = y_reg_test, Predicted = pred_lm,     Model = "Linear Regression"),
  data.frame(Actual = y_reg_test, Predicted = pred_rf_reg, Model = "Random Forest Regressor")
)

reg_ann <- data.frame(
  Model     = c("Linear Regression", "Random Forest Regressor"),
  label     = c(sprintf("R²=%.4f\nRMSE=%.3f", r2_lm,     rmse_lm),
                sprintf("R²=%.4f\nRMSE=%.3f", r2_rf_reg, rmse_rf_reg)),
  Actual    = rep(min(y_reg_test) + 3, 2),
  Predicted = rep(max(c(pred_lm, pred_rf_reg)) - 5, 2)
)

p_reg_scatter <- ggplot(reg_plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, size = 2, colour = "#4C72B0") +
  geom_abline(slope = 1, intercept = 0,
              colour = "#E41A1C", linetype = "dashed", linewidth = 1) +
  geom_text(data = reg_ann, aes(label = label),
            hjust = 0, vjust = 1, size = 3.5, colour = "grey30") +
  facet_wrap(~Model) +
  labs(title    = "Regression — Actual vs Predicted Glucose (mg/dL)",
       subtitle = "Red dashed line = perfect prediction",
       x = "Actual Glucose (mg/dL)", y = "Predicted Glucose (mg/dL)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"))

print(p_reg_scatter)
# Ensure output folder exists
if (!dir.exists("output")) {
  dir.create("output")
}

# Save with white background
ggsave(
  filename = "output/regression_scatter.png",
  plot     = p_reg_scatter,
  width    = 8,
  height   = 5,
  dpi      = 300,
  bg       = "white"
)

reg_long <- reg_benchmark %>%
  pivot_longer(cols = c(RMSE, MAE), names_to = "Metric", values_to = "Value")

p_reg_bench <- ggplot(reg_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.6), width = 0.5, alpha = 0.85) +
  scale_fill_manual(values = c("MAE" = "#F4A582", "RMSE" = "#92C5DE")) +
  labs(title    = "Regression Model Benchmarking (RMSE & MAE)",
       subtitle = "Lower is better | Predicting Glucose (mg/dL)",
       x = NULL, y = "Error (mg/dL)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

print(p_reg_bench)
# Ensure output folder exists
if (!dir.exists("output")) {
  dir.create("output")
}

# Save with white background
ggsave(
  filename = "output/p_reg_bench.png",
  plot     = p_reg_bench,
  width    = 8,
  height   = 5,
  dpi      = 300,
  bg       = "white"
)



# Member 3 final summary

cat(" MEMBER 3 — FINAL SUMMARY ")


cat(" Classification Benchmark ")
print(clf_benchmark, row.names = FALSE)

cat("Regression Benchmark")
print(reg_benchmark, row.names = FALSE)

best_clf <- clf_benchmark$Model[best_f1_idx]
best_reg <- reg_benchmark$Model[which.min(reg_benchmark$RMSE)]

cat(sprintf("
RECOMMENDED CLASSIFICATION MODEL : %s
  F1 Score     : %.4f
  AUC          : %.4f
  Recall       : %.4f  (diabetics correctly identified)
  Precision    : %.4f
  Accuracy     : %.4f

RECOMMENDED REGRESSION MODEL     : %s
  RMSE         : %.3f mg/dL
  R-Squared    : %.4f
\n",
  best_clf,
  clf_benchmark$F1_Score[best_f1_idx],
  clf_benchmark$AUC[best_f1_idx],
  clf_benchmark$Recall[best_f1_idx],
  clf_benchmark$Precision[best_f1_idx],
  clf_benchmark$Accuracy[best_f1_idx],
  best_reg,
  reg_benchmark$RMSE[which.min(reg_benchmark$RMSE)],
  reg_benchmark$R_Squared[which.min(reg_benchmark$RMSE)]
))

cat(" Member 3 complete ")


# MEMBER 4 — Code Review, ROC Curves, Feature Importance, Confusion Matrices

# NOTE: All plots here use the results and models objects produced by Member 3.
#       clf_benchmark is already built above — not rebuilt here.

# ROC curves — all models
roc_df <- bind_rows(lapply(results, function(r) {
  
  data.frame(
    FPR   = 1 - r$roc$specificities,
    TPR   = r$roc$sensitivities,
    Model = r$name
  )
  
}))

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR, colour = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = "grey60") +
  labs(
    title = "ROC Curves (Test Set)",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

print(p_roc)
# Ensure output folder exists
if (!dir.exists("output")) {
  dir.create("output")
}

# Save ROC plot with white background
ggsave(
  filename = "output/roc_curve.png",
  plot     = p_roc,
  width    = 8,
  height   = 5,
  dpi      = 300,
  bg       = "white"
)

# Random Forest feature importance
rf_imp <- varImp(models$RandomForest)$importance
rf_imp$Feature <- rownames(rf_imp)

# Create Overall if missing
if (!"Overall" %in% names(rf_imp)) {
  rf_imp$Overall <- rowMeans(rf_imp[, sapply(rf_imp, is.numeric)])
}

p_rf_imp <- ggplot(
  rf_imp,
  aes(x = reorder(Feature, Overall), y = Overall)
) +
  geom_col(fill = "#377EB8") +
  coord_flip() +
  labs(
    title = "Random Forest Feature Importance",
    x = NULL,
    y = "Importance"
  ) +
  theme_minimal(base_size = 13)

print(p_rf_imp)
# Ensure output folder exists
if (!dir.exists("output")) {
  dir.create("output")
}

# Save ROC plot with white background
ggsave(
  filename = "output/feature_importance.png",
  plot     = p_rf_imp,
  width    = 8,
  height   = 5,
  dpi      = 300,
  bg       = "white"
)

# Logistic Regression coefficients
log_model <- models$Logistic$finalModel

coef_df <- data.frame(
  Feature     = names(coef(log_model)),
  Coefficient = as.numeric(coef(log_model))
) %>%
  filter(Feature != "(Intercept)")

p_log <- ggplot(
  coef_df,
  aes(x = reorder(Feature, Coefficient), y = Coefficient)
) +
  geom_col(fill = "#E41A1C") +
  coord_flip() +
  labs(
    title = "Logistic Regression Coefficients",
    x = NULL,
    y = "Coefficient"
  ) +
  theme_minimal(base_size = 13)

print(p_log)


# Confusion matrices — one per model

plot_cm <- function(r) {
  
  df <- as.data.frame(r$cm$table)
  
  ggplot(df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 5) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = paste("Confusion Matrix —", r$name)) +
    theme_minimal(base_size = 12) +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA)
    )
}

cm_plots <- lapply(results, plot_cm)
invisible(lapply(cm_plots, print))

png("output/all_confusion_matrices.png", width = 1200, height = 800)

grid.arrange(
  grobs = cm_plots,
  ncol = 2
)
dev.off()

# Precision–Recall scatter plot
p_pr <- ggplot(
  clf_benchmark,
  aes(x = Recall, y = Precision, label = Model)
) +
  geom_point(size = 3, colour = "#4C72B0") +
  geom_text(vjust = -0.6) +
  labs(
    title = "Precision–Recall Tradeoff",
    x = "Recall (Sensitivity)",
    y = "Precision"
  ) +
  theme_minimal(base_size = 13)

print(p_pr)


# Summary panel
grid.arrange(
  p_roc,
  p_rf_imp,
  p_log,
  p_pr,
  ncol = 2
)

cat("Section 4: Additional Visualisations COMPLETE")
cat("Full pipeline COMPLETE")


