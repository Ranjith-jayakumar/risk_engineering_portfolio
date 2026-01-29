import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
rows = 300

# Flood and earthquake scores sampled from {0.3, 0.6, 0.9}
flood_scores = np.random.choice([0.3, 0.6, 0.9], rows)
earthquake_scores = np.random.choice([0.3, 0.6, 0.9], rows)

# Wind scores sampled from uniform distribution between 0 and 1
wind_scores = np.random.uniform(0, 1, rows)

# NatCat score calculation
natcat_scores = np.round(0.5*flood_scores + 0.3*earthquake_scores + 0.2*wind_scores, 2)

# Create DataFrame
df = pd.DataFrame({
    'flood_score': flood_scores,
    'earthquake_score': earthquake_scores,
    'wind_score': wind_scores,
    'natcat_score': natcat_scores
})

# Save to CSV
df.to_csv(r"..\Risk_Engineering\src\ml_algo\natcat\data\natcat_dataset.csv", index=False)

print("CSV file 'natcat_dataset.csv' has been created with 300 rows.")
