import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
rows = 300

# Generate synthetic values
single_site_flags = np.random.choice([True, False], rows)   # True = single site, False = multiple
critical_processes = np.random.choice([0, 1], rows)         # 1 if critical process
bi_sum_insured = np.random.uniform(0, 2e8, rows)            # random BI sum insured up to 200M
normalized_bi = np.minimum(bi_sum_insured/1e8, 1)           # normalize to max 1
loss_flags = np.random.choice([True, False], rows)          # loss history flag

# Operational risk score calculation
operational_risk_scores = np.round(
    0.3*(1 - single_site_flags.astype(int)) +
    0.3*critical_processes +
    0.2*normalized_bi +
    0.2*(1 - loss_flags.astype(int)), 2
)

# Create DataFrame
df = pd.DataFrame({
    'single_site': single_site_flags.astype(int),   # store as 0/1
    'critical_process': critical_processes,
    'bi_sum_insured': bi_sum_insured,
    'normalized_bi': normalized_bi,
    'loss_flag': loss_flags.astype(int),            # store as 0/1
    'operational_risk_score': operational_risk_scores
})

# Save to CSV
df.to_csv(r"..\Risk_Engineering\src\ml_algo\operational_risk\data\operational_risk_dataset.csv", index=False)

print("CSV file 'operational_risk_dataset.csv' created with 300 rows.")

