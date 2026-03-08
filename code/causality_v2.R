# --- 1. LIBRARIES ---
if(!require(randomForest)) install.packages("randomForest")
if(!require(dplyr)) install.packages("dplyr")
if(!require(lubridate)) install.packages("lubridate")

library(randomForest)
library(dplyr)
library(lubridate)

# --- 2. LOAD & FORCE-CLEAN DATA ---
# We assume the file is named 'enhanced_data_v2.csv'
data <- read.csv("enhanced_data_v2.csv", stringsAsFactors = FALSE)

# DATA TYPE FIXER (The most important part)
# We force every relevant column to be numeric, removing commas if they exist
clean_num <- function(x) { as.numeric(gsub(",", "", as.character(x))) }

data$GCC_Total_Imports             <- clean_num(data$GCC_Total_Imports)
data$Estimated_Violence_Fatalities <- clean_num(data$Estimated_Violence_Fatalities)
data$Brent_Crude_Price_USD_Barrel  <- clean_num(data$Brent_Crude_Price_USD_Barrel)
data$Gold_Price_USD_oz             <- clean_num(data$Gold_Price_USD_oz)
data$Global_Rubber_Price_Index     <- clean_num(data$Global_Rubber_Price_Index)
data$Modeled_Tire_Imports_4011.20_Units <- clean_num(data$Modeled_Tire_Imports_4011.20_Units)

# Fix Dates
clean_months <- substr(data$Month, 1, 3) 
month_numbers <- match(clean_months, month.abb)
data$date_obj <- make_date(year = data$Year, month = month_numbers, day = 1)
data <- data[order(data$date_obj),] # Sort by date

# --- 3. CREATE LAGS ---
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
  na.omit() # Drops the first 4 rows (Lags)

# --- 4. RUN MODEL ---
print(paste("Running model on", nrow(model_data), "rows..."))

set.seed(123)
rf_model <- randomForest(
  Target_Violence ~ .,
  data = model_data,
  ntree = 500,
  importance = TRUE
)

# --- 5. SAFE PLOTTING (Manual Code) ---
# We extract values manually to prevent the 'xlim' crash
imp_matrix <- importance(rf_model)

# Sanity Check: Print the raw numbers
print("--- Importance Values ---")
print(imp_matrix)

# If values are NaN (Not a Number), replace with 0
imp_matrix[is.na(imp_matrix)] <- 0

# Sort by importance (%IncMSE is column 1)
sorted_imp <- imp_matrix[order(imp_matrix[,1], decreasing = FALSE), ]

# Create the Chart
dotchart(sorted_imp[, 1], 
         main = "Predictive Power (Fixed)",
         xlab = "% Increase in Accuracy (Importance)",
         pch = 19,
         col = "blue")

# --- 6. PREDICTION TEST ---
# Let's verify: What does the model predict for the very last row of data?
last_row <- tail(model_data, 1)
predicted_val <- predict(rf_model, last_row)
actual_val <- last_row$Target_Violence

print("--- REALITY CHECK (Last Data Point) ---")
print(paste("Actual Fatalities:", actual_val))
print(paste("Predicted Fatalities:", round(predicted_val)))
print(paste("Error:", round(abs(predicted_val - actual_val))))

# --- THE SMOKING GUN PLOT ---
# We normalize the data (scale 0-1) so we can plot them on the same chart
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }

plot_data <- model_data
plot_data$Norm_Violence <- normalize(plot_data$Target_Violence)
plot_data$Norm_Toyota   <- normalize(plot_data$Toyota_Lag4)

# Create the plot
plot(plot_data$Norm_Violence, type = "l", col = "red", lwd = 3,
     main = "Toyota Imports (Lagged) vs. Violence",
     xlab = "Time", ylab = "Normalized Intensity (0-1)")

# Add the Toyota Line
lines(plot_data$Norm_Toyota, col = "blue", lwd = 2, lty = 2)

# Add Legend
legend("topleft", legend = c("Conflict Fatalities", "Toyota Imports (4 Months Prior)"),
       col = c("red", "blue"), lty = c(1, 2), lwd = c(3, 2))


# --- FORECASTING THE FUTURE ---
# We want to predict Jan 2026.
# The inputs come from 4 months ago (Sep 2025).

# 1. Get the data from Sep 2025
sep_2025_data <- data[data$Month == "Sep" & data$Year == 2025, ]

# 2. Build the input row (matching the 'model_data' structure)
future_input <- data.frame(
  Toyota_Lag4 = sep_2025_data$GCC_Total_Imports,
  Tires_Lag4  = sep_2025_data$Modeled_Tire_Imports_4011.20_Units,
  Oil_Lag4    = sep_2025_data$Brent_Crude_Price_USD_Barrel,
  Gold_Lag4   = sep_2025_data$Gold_Price_USD_oz,
  Rubber_Lag4 = sep_2025_data$Global_Rubber_Price_Index
)

# 3. Ask the Model
prediction <- predict(rf_model, future_input)

print("--- INTELLIGENCE FORECAST: JAN 2026 ---")
print(paste("Based on Sep 2025 Logistics, we predict:", round(prediction), "Fatalities."))