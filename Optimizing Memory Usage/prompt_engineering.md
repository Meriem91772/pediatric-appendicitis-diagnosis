# Task: Memory Optimization Function
# 1️⃣ Initial Human Attempt (Partially Incorrect Idea)
# At first, I attempted to manually convert numerical data types to save memory. My idea was to check each column type and directly convert float64 to float32 and int64 to int32.
def optimize_memory(cleaned_dt):
for col in cleaned_dt.columns:
if cleaned_dt[col].dtype == 'float64':
cleaned_dt[col] = cleaned_dt[col].astype('float32')
elif cleaned_dt[col].dtype == 'int64':
cleaned_dt[col] = cleaned_dt[col].astype('int32')
return cleaned_dt
# Issues with My Approach:

# No Memory Comparison: I didn’t measure memory usage before and after optimization.
# Missed object Columns: Some categorical columns stored as object could be converted to category for further optimization.
# No Safety Checks: Some conversions could have caused data loss if not handled properly.
# 2️⃣ AI-Corrected Version (ChatGPT Assistance)
# I asked AI for a better version that also calculates memory usage before and after optimization.

# 🔹 Prompt Used:
# "Improve my memory optimization function for Pandas DataFrame. Ensure it prints memory usage reduction and converts object columns to category."

# 🔹 AI’s Correction (40% Contribution)
# ChatGPT provided the following enhanced function :
import pandas as pd
import numpy as np

def optimize_memory(cleaned_dt):
before_mem = cleaned_dt.memory_usage(deep=True).sum() / 1024**2  # Memory in MB

    for col in cleaned_dt.columns:
        col_type = cleaned_dt[col].dtype
        
        if col_type == 'int64':
            cleaned_dt[col] = cleaned_dt[col].astype('int32')
        elif col_type == 'float64':
            cleaned_dt[col] = cleaned_dt[col].astype('float32')
        elif col_type == 'object':  # Convert text-based columns to category
            cleaned_dt[col] = cleaned_dt[col].astype('category')
    
    after_mem = cleaned_dt.memory_usage(deep=True).sum() / 1024**2  # Memory in MB
    print(f"Memory usage reduced from {before_mem:.2f} MB to {after_mem:.2f} MB ({((before_mem - after_mem) / before_mem) * 100:.2f}% reduction).")
    return cleaned_dt 