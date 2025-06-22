# =============================================================================
# File:         marketing_affinity_score_model.r
# Author:       Dr. Arif Doenmez
# Contact:      https://arifdoenmez.github.io/
# Date Created: 2025-04-21
#
# Project:      Marketing Affinity Score Model for Car Insurance
#
# Description:  This script performs an end-to-end analysis to build a predictive
#               model for customer affinity towards a car insurance product. It includes
#               data loading and merging, exploratory data analysis (EDA),
#               feature engineering, and the training and evaluation of a
#               Random Forest model using the 'caret' and 'ranger' packages.
#
# Libraries:    data.table, cli, ggplot2, lubridate, caret, ranger, pROC, e1071
# =============================================================================

library(data.table)
library(cli)
library(ggplot2)
library(lubridate)
library(caret)
library(ranger)      # Faster implementation of random forest
library(pROC)        # For ROC curve analysis
library(e1071)

cli_h3("Loading Data")  # load  -------

dt_alter_geschlecht <- fread('alter_geschlecht.csv')
dt_rest <- fread('rest.csv', sep = ';') # Specify separator
dt_interesse <- fread('interesse.csv')


cli_h3("Merging Data")  # merge  -------

# Merge demographics and other features
dt_merged <- merge(dt_alter_geschlecht, dt_rest, by = "id", all = FALSE) # Inner join by default

# Merge with the target variable
dt_final <- merge(dt_merged, dt_interesse, by = "id", all = FALSE) # Inner join

# Check for duplicate IDs after merge - should ideally be 0
cli_alert_warning(paste("Duplicate IDs after merging:", sum(duplicated(dt_final$id))))

cli_alert_success("Final (Merged) Data Table")
print(tibble::as_tibble(dt_final))


cli_h3("Exploratory Data Analysis")  # EDA -----

cli_alert_info("Converting 'Jahresbeitrag' to numeric...")
dt_final[, Jahresbeitrag := as.numeric(Jahresbeitrag)]

# Check Missing Values (% per column)
cli_alert_warning("Missing Values (%):")
missing_perc <- sapply(dt_final, function(x) sum(is.na(x)) / length(x) * 100)
print(sort(missing_perc[missing_perc > 0], decreasing = TRUE))


cli_alert_info("Target Variable 'Interesse' Distribution:")  # Target Variable  ----.
dt_final[, Interesse := factor(Interesse, levels = c(0, 1), labels = c("No", "Yes"))]
target_counts <- dt_final[, .N, by = Interesse][, Pct := round(N / sum(N) * 100, 2)][]  # Calculate counts and percentages
print(tibble::as_tibble(target_counts))

print(ggplot(dt_final, aes(x = Interesse)) +
        geom_bar(fill = "steelblue") +
        geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
        labs(title = "Distribution of Target Variable (Interesse)", x = "Interesse (0=No, 1=Yes)", y = "Count") +
        theme_minimal())


# Observation: The target variable is imbalanced 




cli_h3("Feature Engineering")

# Convert GebDat to Age
cli_alert_info("Calculating Age...")
reference_date <- as.Date('2024-01-01')
dt_final[, GebDat := as.Date(GebDat, format = "%Y-%m-%d")] # Ensure correct date format
dt_final[, Age := as.numeric(difftime(reference_date, GebDat, units = "days") / 365.25)]



options(tibble.width = 400)  # want to see more cols 


cli_alert_info("Age Summary:")
print(summary(dt_final$Age))
# Median imputation if Age is NA
age_na_count <- sum(is.na(dt_final$Age))
if (age_na_count > 0) {
    cli_alert_warning(paste("Warning: Found", age_na_count, "missing values in Age. Imputing with median age."))
    median_age <- median(dt_final$Age, na.rm = TRUE)
    dt_final[is.na(Age), Age := median_age]
}

cli_text()
jahresbeitrag_na_count <- sum(is.na(dt_final$Jahresbeitrag))
if (jahresbeitrag_na_count > 0) {
    cli_alert_warning(paste("Warning: Found", jahresbeitrag_na_count, "missing values in Jahresbeitrag. Imputing with median Jahresbeitrag."))
    median_jahresbeitrag <- median(dt_final$Jahresbeitrag, na.rm = TRUE)
    dt_final[is.na(Jahresbeitrag), Jahresbeitrag := median_jahresbeitrag]
}


cli_alert_info("Mapping 'Vorschaden' Yes/No to factor 1/0...")
# Using fifelse for efficiency in data.table
dt_final[, Vorschaden := factor(fifelse(Vorschaden == "Yes", 1, 0), levels = c(0, 1))]
# Handle potential NAs created if source had other values/NAs
if (anyNA(dt_final$Vorschaden)) {
    print("Warning: 'Vorschaden' contains NAs after mapping. Filling with mode (0 - No).")
    # Calculate mode (most frequent level)
    mode_val <- dt_final[, .N, by = Vorschaden][order(-N)][1, Vorschaden]
    dt_final[is.na(Vorschaden), Vorschaden := mode_val]
}


cli_alert_info("Ensure other binary columns are factors with levels 0, 1")
dt_final[, Fahrerlaubnis := factor(Fahrerlaubnis, levels = c(0, 1))]
dt_final[, Vorversicherung := factor(Vorversicherung, levels = c(0, 1))]

character_cols <- names(dt_final)[sapply(dt_final, is.character)]
cli_alert_info(paste("Converting character columns to factors:", paste(character_cols, collapse=", ")))
for (col in character_cols) {
    dt_final[, (col) := as.factor(get(col))]
}


# Handle Ordinal Factor: Alter_Fzg
alter_fzg_order <- c('< 1 Year', '1-2 Year', '> 2 Years') # Define the order
cli_alert_info("Setting ordered levels for 'Alter_Fzg'...")
dt_final[, Alter_Fzg := factor(Alter_Fzg, levels = alter_fzg_order, ordered = TRUE)]

# Prepare final data for modeling: drop ID and original date
dt_model_data <- dt_final[, !c("id", "GebDat")]


cli_h3("Data Splitting")
cli_alert_info("Splitting data into training and testing sets (80/20)...")
set.seed(42) # for reproducibility
# Use createDataPartition from caret for stratified sampling
train_index <- createDataPartition(y = dt_model_data$Interesse, p = 0.8, list = FALSE)
train_data <- dt_model_data[train_index, ]
test_data <- dt_model_data[-train_index, ]


if(FALSE){



cli_h3("Modellbuilding - Random Forest")

results_list <- list()

# Define the model formula (predict Interesse using all other variables)
# model_formula <- Interesse ~ .

# Define training control: 5-fold CV, calculate AUC (using twoClassSummary)
# Using `twoClassSummary` requires class probabilities
ctrl <- trainControl(method = "cv",
                     number = 5,
                     summaryFunction = twoClassSummary, # Calculates ROC (AUC), Sens, Spec
                     classProbs = TRUE,         # MUST be TRUE for twoClassSummary
                     verboseIter = FALSE,       # Suppress progress updates during CV
                     savePredictions = "final") # Save predictions from CV folds


tuneGrid_rf <- expand.grid(.mtry = floor(sqrt(ncol(train_data)-1)), # Basic tuning grid for ranger
                            .splitrule = "gini",
                            .min.node.size = 1)


cli_alert_info("Training Random Forest ...")
set.seed(42) # Reset seed for each model training for reproducibility

tryCatch({

    model_fit <- train(Interesse ~ .,
                       data = train_data,
                       method = "ranger",
                       trControl = ctrl,
                       metric = "ROC",  # Optimize for AUC
                       preProcess = c("medianImpute", "center", "scale"), # Impute missing, center, scale numerical
                       tuneGrid = tuneGrid_rf,
                       importance = "permutation"
                    )

    results_list[["Random Forest"]] <- model_fit
    cli_alert_info("CV Results (AUC):")
    cli_alert_info(model_fit$results[which.max(model_fit$results$ROC), c("ROC", "Sens", "Spec")]) # Show best tuning result

}, error = function(e) {
    cli_alert_danger(paste("Error training Random Forest:", e$message))
})


selected_model_name <- "Random Forest" # Or choose based on results_list comparison
final_model <- results_list[[selected_model_name]]

cli_alert_success(paste("Model ready:", selected_model_name))
print(final_model)

# saveRDS(final_model, 'model_affinity.rds')  # save model for further usages
# cli_alert_success("Model saved.")



cli_h3("Evaluating Final Model on Test Set")  # Evaluation on Test Set

# Predict probabilities for the positive class ('Yes')
pred_prob <- predict(final_model, newdata = test_data, type = "prob")

# Predict class labels ('Yes'/'No')
pred_class <- predict(final_model, newdata = test_data, type = "raw") # 'raw' gives class labels

# Calculate AUC
roc_obj <- roc(response = test_data$Interesse, predictor = pred_prob$Yes, levels = c("No", "Yes")) # Specify levels explicitly
auc_score <- auc(roc_obj)
cli_alert_info(paste("Test Set AUC Score:", round(auc_score, 4)))


cli_alert_info("Confusion Matrix and Statistics:") # Classification Report / Confusion Matrix
# Ensure pred_class and test_data$Interesse are factors with same levels
conf_matrix <- confusionMatrix(data = pred_class,
                               reference = test_data$Interesse,
                               positive = "Yes") # Specify the positive class
print(conf_matrix)

# Plot ROC Curve
cli_alert_info("Plotting ROC Curve...")
plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc_score, 4), ")"), print.auc = TRUE)



cli_h3("Feature Importances") # Feature Importances
tryCatch({
    imp <- varImp(final_model, scale = TRUE) # Use scale=TRUE for comparable importances
    cli_alert_info("Top 20 Features:")
    print(imp) # Print the importance object
    # Plot top N feature importances
    plot(imp, top = 20, main = "Top 20 Feature Importances")
}, error = function(e){
    cli_alert_danger(paste("Could not extract feature importances:", e$message))
})

}