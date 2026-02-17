# --- 1. SETUP ---
if(!require(randomForest)) install.packages("randomForest")
if(!require(dplyr)) install.packages("dplyr")
if(!require(ggplot2)) install.packages("ggplot2") # For the curve plot

library(randomForest)
library(dplyr)
library(ggplot2)

# Load Data (Assuming file is in folder)
data <- read.csv("enhanced_data_v2.csv", stringsAsFactors = FALSE)

# Helper to clean numbers
clean_num <- function(x) { as.numeric(gsub(",", "", as.character(x))) }
data$Estimated_Violence_Fatalities <- clean_num(data$Estimated_Violence_Fatalities)
data$GCC_Total_Imports <- clean_num(data$GCC_Total_Imports)
data$Modeled_Tire_Imports_4011.20_Units <- clean_num(data$Modeled_Tire_Imports_4011.20_Units)
data$Brent_Crude_Price_USD_Barrel <- clean_num(data$Brent_Crude_Price_USD_Barrel)
data$Gold_Price_USD_oz <- clean_num(data$Gold_Price_USD_oz)
data$Global_Rubber_Price_Index <- clean_num(data$Global_Rubber_Price_Index)

# Feature Engineering (Lags)
LAG_MONTHS <- 4
model_data <- data %>%
  mutate(
    Target_Violence = Estimated_Violence_Fatalities,
    Toyota_Lag4 = dplyr::lag(GCC_Total_Imports, n = LAG_MONTHS),
    Tires_Lag4  = dplyr::lag(Modeled_Tire_Imports_4011.20_Units, n = LAG_MONTHS),
    Oil_Lag4    = dplyr::lag(Brent_Crude_Price_USD_Barrel, n = LAG_MONTHS),
    Gold_Lag4   = dplyr::lag(Gold_Price_USD_oz, n = LAG_MONTHS),
    Rubber_Lag4 = dplyr::lag(Global_Rubber_Price_Index, n = LAG_MONTHS)
  ) %>%
  select(Target_Violence, Toyota_Lag4, Tires_Lag4, Oil_Lag4, Gold_Lag4, Rubber_Lag4) %>%
  na.omit()

# --- 2. THE OVERFITTING CHECK (80/20 Split) ---
# We split by TIME. Train on the first 80%, Test on the future 20%.
split_idx <- floor(0.8 * nrow(model_data))
train_set <- model_data[1:split_idx, ]
test_set  <- model_data[(split_idx + 1):nrow(model_data), ]

print(paste("Training on first", nrow(train_set), "months. Testing on next", nrow(test_set), "months."))

set.seed(123)
rf_split <- randomForest(Target_Violence ~ ., data = train_set, ntree = 500)

# Calculate Errors (RMSE)
rmse <- function(actual, pred) { sqrt(mean((actual - pred)^2)) }

# 1. How well did it memorize the past? (Train Error)
pred_train <- predict(rf_split, train_set)
error_train <- rmse(train_set$Target_Violence, pred_train)

# 2. How well does it predict the future? (Test Error)
pred_test <- predict(rf_split, test_set)
error_test <- rmse(test_set$Target_Violence, pred_test)

print("--- OVERFITTING REPORT ---")
print(paste("Training Error (RMSE):", round(error_train)))
print(paste("Test/Future Error (RMSE):", round(error_test)))

ratio <- error_test / error_train
print(paste("Overfitting Ratio:", round(ratio, 2)))

if(ratio > 2.0) {
  print("WARNING: High Overfitting detected. Model is memorizing the past but failing on the future.")
} else {
  print("STATUS: GOOD. The model generalizes well.")
}

# --- 3. GENERATE LEARNING CURVE ---
# We train the model on increasing chunks of data (20 rows, 30 rows...)
# and see if the Test Error drops.

print("Generating Learning Curve...")
results <- data.frame()

# Start with 20 months of data, add 5 months at a time
subset_sizes <- seq(20, nrow(train_set), by = 5)

for(size in subset_sizes) {
  # Train on specific subset size
  sub_train <- train_set[1:size, ]
  
  # Train Model
  set.seed(123)
  m <- randomForest(Target_Violence ~ ., data = sub_train, ntree = 500)
  
  # Predict on the FIXED Test Set (The Future)
  p_train <- predict(m, sub_train)
  p_test  <- predict(m, test_set)
  
  # Record RMSE
  results <- rbind(results, data.frame(
    Size = size,
    Train_Error = rmse(sub_train$Target_Violence, p_train),
    Test_Error  = rmse(test_set$Target_Violence, p_test)
  ))
}

# --- 4. PLOT THE CURVE ---
ggplot(results, aes(x = Size)) +
  geom_line(aes(y = Train_Error, color = "Train Error"), size = 1) +
  geom_line(aes(y = Test_Error, color = "Test (Future) Error"), size = 1) +
  labs(title = "Learning Curve: Is the model learning?",
       subtitle = "If lines converge (get closer), more data helps. If wide gap, overfitting.",
       x = "Number of Months in Training Data",
       y = "Error (RMSE - Lower is Better)",
       color = "Metric") +
  theme_minimal()