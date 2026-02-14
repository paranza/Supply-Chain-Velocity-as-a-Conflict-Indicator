# --- 1. LIBRARIES ---
if(!require(vars)) install.packages("vars")
if(!require(lubridate)) install.packages("lubridate")
if(!require(dplyr)) install.packages("dplyr")

library(vars)
library(lubridate)
library(dplyr)

# --- 2. ROBUST DATE FIXING ---
# Convert months to Date object
# We strip the first 3 letters to handle "Sep" vs "Sept" vs "September"
clean_months <- substr(data$Month, 1, 3) 
month_numbers <- match(clean_months, month.abb)

# Create the date object
data$date_obj <- make_date(year = data$Year, month = month_numbers, day = 1)
data <- data[order(data$date_obj),]

# --- 3. SAFE CLEANING ---
# We use column INDICES (3, 4, 5) to avoid name errors.
# We also turn NAs into 0 to prevent crashes.
clean_num <- function(x) { 
  val <- as.numeric(gsub(",", "", x))
  val[is.na(val)] <- 0
  return(val)
}

# WARNING: Ensure your columns are in this order: Month, Year, IMPORTS, CONFLICT, OIL
# If your order is different, change these numbers!
imports_clean  <- clean_num(data[,3])
conflict_clean <- clean_num(data[,4])
oil_clean      <- clean_num(data[,5])

# --- 4. PREPARE TIME SERIES (With Zero-Protection) ---
# We add +1 inside log() to handle months with 0 sales/events
# This prevents "-Inf" errors.
ts_matrix <- data.frame(
  Conflict = diff(log(conflict_clean + 1)),
  Imports  = diff(log(imports_clean + 1)), 
  Oil      = diff(log(oil_clean + 1))
)

# Remove any infinite or NA values created by the math
ts_matrix <- ts_matrix[is.finite(rowSums(ts_matrix)),]

# Check if we have enough data left
print(paste("Rows of usable data:", nrow(ts_matrix)))

if(nrow(ts_matrix) < 10) {
  stop("ERROR: You have fewer than 10 rows of clean data. You cannot run Granger Causality on such a small dataset.")
}

# --- 5. RUN CAUSALITY TEST ---
# A. Optimal Lag (Safeguarded)
# If data is short, lag.max=4 might fail. We use a tryCatch block.
optimal_lag <- tryCatch({
  lag_selection <- VARselect(ts_matrix, lag.max = 4, type = "const")
  as.numeric(lag_selection$selection["AIC"])
}, error = function(e) { 
  return(1) # Fallback to 1 if it crashes
})
 
# If optimal_lag comes back NA, force it to 1
if(is.na(optimal_lag)) { optimal_lag <- 3}

print(paste("Using Lag:", optimal_lag))

# B. Run Model
var_model <- VAR(ts_matrix, p = optimal_lag, type = "const")

# C. Test: Does Imports (Col 3) predict Conflict (Col 4)?
test_result <- causality(var_model, cause = "Imports")

print("---------------- RESULTS ----------------")
print(test_result$Granger)
print("-----------------------------------------")
print("INTERPRETATION:")
print("p-value < 0.05 : Imports PREDICT Conflict (Significant)")