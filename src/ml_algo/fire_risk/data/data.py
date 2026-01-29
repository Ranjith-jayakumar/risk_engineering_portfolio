import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
rows = 300

# Generate synthetic values
sprinkler_scores = np.random.uniform(0, 1, rows)   # coverage percentage normalized
fire_brigade_scores = np.random.uniform(0, 1, rows) # proximity score
detection_scores = np.random.choice([0, 1], rows)   # binary detection system
housekeeping_scores = np.random.uniform(0, 1, rows) # housekeeping score normalized

# Fire risk score calculation
fire_risk_scores = np.round(
    0.4*sprinkler_scores + 
    0.3*fire_brigade_scores + 
    0.2*detection_scores + 
    0.1*housekeeping_scores, 2
)

# Create DataFrame
df = pd.DataFrame({
    'sprinkler_score': sprinkler_scores,
    'fire_brigade_score': fire_brigade_scores,
    'detection_score': detection_scores,
    'housekeeping_score': housekeeping_scores,
    'fire_risk_score': fire_risk_scores
})

# Save to CSV

df.to_csv(r"..\Risk_Engineering\src\ml_algo\fire_risk\data\fire_risk_dataset.csv", index=False)

print("CSV file 'fire_risk_dataset.csv' created with 300 rows.")

