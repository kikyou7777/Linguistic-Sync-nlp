import pandas as pd

# Read the CSV file
df = pd.read_csv('paired_file.csv')

# Count distinct receivers per sender for each condition
sender_receiver_counts = df.groupby(['condition', 'sender_id'])['receiver_id'].nunique().reset_index()

# Compute network density: actual connections / max possible connections (9)
sender_receiver_counts['density'] = sender_receiver_counts['receiver_id'] / 9

# Compute mean and standard deviation of distinct receiver counts per condition
density_stats = sender_receiver_counts.groupby('condition')['density'].agg(['mean', 'std']).reset_index()

# Save results to CSV
sender_receiver_counts.to_csv('sender_receiver_counts.csv', index=False)
density_stats.to_csv('network_density_stats.csv', index=False)

# Print summary
print("Network Density Stats by Condition:")
print(density_stats)
import pandas as pd
from scipy.stats import ttest_ind

# Read the computed network density data
density_data = pd.read_csv('sender_receiver_counts.csv')

# Separate data by condition
exploit_density = density_data[density_data['condition'] == 'exploit']['density']
explore_density = density_data[density_data['condition'] == 'explore']['density']

# Perform an independent t-test
t_stat, p_value = ttest_ind(exploit_density, explore_density, equal_var=False)  # Welch's t-test
print("exploit density is",exploit_density, "and explore is", explore_density)

# Degrees of freedom (approximate for Welch's test)
df = len(exploit_density) + len(explore_density) - 2

# Print results
print(f"Network density significantly differed between conditions (t = {t_stat:.2f}, p = {p_value:.3f}, df = {df})")

# Interpretation
if p_value < 0.05:
    print("The difference is statistically significant.")
else:
    print("No significant difference between conditions.")


# Load your paired data
df = pd.read_csv('paired_file.csv')

# Count unique receiver_id per sender_id within each session
density_df = df.groupby(['chat_id', 'condition', 'sender_id'])['receiver_id'].nunique().reset_index()

# Normalize density to a 0-1 scale (max 9 connections per sender)
density_df['density'] = density_df['receiver_id'] / 9

# Aggregate by session to get the average network density per session
session_density = density_df.groupby(['chat_id', 'condition'])['density'].mean().reset_index()

# Ensure there are exactly 7 sessions per condition
print(session_density['condition'].value_counts())  # Should print 7 for each condition

# Separate conditions
explore_density = session_density[session_density['condition'] == 'explore']['density']
exploit_density = session_density[session_density['condition'] == 'exploit']['density']
control_density = session_density[session_density['condition'] == 'control']['density']

# Perform an independent t-test
t_stat, p_value = ttest_ind(explore_density, exploit_density, equal_var=True)  # Assume equal variance

# Degrees of freedom = (7 + 7 - 2) = 12
df = len(explore_density) + len(exploit_density) + len(control_density)- 2

print(f"Network density significantly differed between conditions (t = {t_stat:.2f}, p = {p_value:.3f}, df = {df})")
