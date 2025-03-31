import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# ============================================================
# 1. Load Dyadic Data and Preprocessing
# ============================================================

# Load Dyadic STS Scores
dyadic_df = pd.read_csv("dyadic_sts_scores.csv")  # Replace with your actual dyadic file path

# Remove rows where sts_score == 1
dyadic_df = dyadic_df[dyadic_df['sts_score'] != 1]

dyadic_df['message1_word_count'] = dyadic_df['message1'].astype(str).apply(lambda x: len(x.split()))

# Sum the word counts for all rows
total_words = dyadic_df['message1_word_count'].sum()

print("Total words in 'message1' column:", total_words)
# Calculate mean STS scores for users by grouping by sender_id (or receiver_id if needed)
user_mean_sts = dyadic_df.groupby('sender_id')['sts_score'].mean().reset_index()

# Rename columns to match the expected format in user_mean_sts_scores.csv
user_mean_sts.rename(columns={"sender_id": "SubID", "sts_score": "mean_sts_score"}, inplace=True)

# ============================================================
# 2. Load Pre-Survey Data with Group Labels
# ============================================================
pre_files = ["preExperiment_control_6231.csv", "preExperiment_explore_6231.csv", "preExperiment_exploit_6231.csv"]
dfs = []
for f in pre_files:
    df_temp = pd.read_csv(f)
    # Assign group based on file name
    if "control" in f:
        df_temp['Group'] = 'control'
    elif "explore" in f:
        df_temp['Group'] = 'explore'
    elif "exploit" in f:
        df_temp['Group'] = 'exploit'
    dfs.append(df_temp)
pre_survey_df = pd.concat(dfs, ignore_index=True)
# Save group info (SubID and Group) for later merge
group_info = pre_survey_df[['SubID', 'Group']].drop_duplicates()

# ============================================================
# 3. Load Post-Survey Data
# ============================================================
post_survey_df = pd.read_csv("postExperiment_6231.csv")

# Select only the SPANE columns (there are 12 items)
spane_columns = ['SPANE_1', 'SPANE_2', 'SPANE_3', 'SPANE_4', 'SPANE_5', 'SPANE_6',
                 'SPANE_7', 'SPANE_8', 'SPANE_9', 'SPANE_10', 'SPANE_11', 'SPANE_12']
pre_survey_df = pre_survey_df[['SubID', 'Group'] + spane_columns]
post_survey_df = post_survey_df[['SubID'] + spane_columns]

# ============================================================
# 4. Compute SPANE Scores (Regrouping into Positive and Negative)
# ============================================================
def compute_spane(df):
    """
    Computes SPANE scores by grouping items into:
      - Positive items: SPANE_1, SPANE_3, SPANE_5, SPANE_7, SPANE_10, SPANE_12
      - Negative items: SPANE_2, SPANE_4, SPANE_6, SPANE_8, SPANE_9, SPANE_11
    """
    positive_items = ['SPANE_1', 'SPANE_3', 'SPANE_5', 'SPANE_7', 'SPANE_10', 'SPANE_12']
    negative_items = ['SPANE_2', 'SPANE_4', 'SPANE_6', 'SPANE_8', 'SPANE_9', 'SPANE_11']

    # Convert responses to numeric; non-numeric become NaN
    df[positive_items] = df[positive_items].apply(pd.to_numeric, errors='coerce')
    df[negative_items] = df[negative_items].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing data in any SPANE item
    df.dropna(subset=positive_items + negative_items, inplace=True)

    # Sum the responses for each grouping
    df['SPANE_P'] = df[positive_items].sum(axis=1)
    df['SPANE_N'] = df[negative_items].sum(axis=1)
    df['SPANE_Balance'] = df['SPANE_P'] - df['SPANE_N']

    return df[['SubID', 'SPANE_P', 'SPANE_N', 'SPANE_Balance']]


# Compute scores for pre- and post-surveys
pre_spane = compute_spane(pre_survey_df.copy())
post_spane = compute_spane(post_survey_df.copy())

# ============================================================
# 5. Merge Pre & Post Surveys and Compute Change Variables
# ============================================================
# Merge by SubID; suffixes indicate pre and post measurements
spane_change = pd.merge(pre_spane, post_spane, on="SubID", suffixes=("_pre", "_post"))

# Compute changes separately for positive and negative scores, and also for overall balance
spane_change['SPANE_P_change'] = spane_change['SPANE_P_post'] - spane_change['SPANE_P_pre']
spane_change['SPANE_N_change'] = spane_change['SPANE_N_post'] - spane_change['SPANE_N_pre']
spane_change['SPANE_Balance_change'] = spane_change['SPANE_Balance_post'] - spane_change['SPANE_Balance_pre']

# Merge group information (from pre-survey)
spane_change = pd.merge(spane_change, group_info, on="SubID", how="left")

# ============================================================
# 6. Merge with STS Scores and Save Final DataFrame
# ============================================================
user_mean_sts['SubID'] = pd.to_numeric(user_mean_sts['SubID'], errors='coerce')
spane_change['SubID'] = pd.to_numeric(spane_change['SubID'], errors='coerce')
final_df = pd.merge(user_mean_sts, spane_change, on="SubID")

# Save merged dataset for future reference
final_df.to_csv("spane_analysis_baseline.csv", index=False)
print("Saved: spane_analysis_baseline.csv")

# ============================================================
# 7. Overall Analysis: STS Score vs. SPANE Changes (Positive & Negative separately)
# ============================================================

print("\n--- Overall Relationship: SPANE Positive Change ---")
corr_P, pval_P = stats.pearsonr(final_df['mean_sts_score'], final_df['SPANE_P_change'])
print(f"Pearson Correlation (Positive): {corr_P:.3f}, p-value: {pval_P:.3f}")
X_pos = sm.add_constant(final_df['mean_sts_score'])
model_pos = sm.OLS(final_df['SPANE_P_change'], X_pos).fit()
print(model_pos.summary())

print("\n--- Overall Relationship: SPANE Negative Change ---")
corr_N, pval_N = stats.pearsonr(final_df['mean_sts_score'], final_df['SPANE_N_change'])
print(f"Pearson Correlation (Negative): {corr_N:.3f}, p-value: {pval_N:.3f}")
X_neg = sm.add_constant(final_df['mean_sts_score'])
model_neg = sm.OLS(final_df['SPANE_N_change'], X_neg).fit()
print(model_neg.summary())

# ============================================================
# 8. Subgroup Analysis: Relationships by Group (with Interaction)
# ============================================================
print("\n--- Subgroup Analysis: With Interaction ---")
groups = final_df['Group'].unique()
for grp in groups:
    print(f"\nGroup: {grp}")
    grp_df = final_df[final_df['Group'] == grp]
    if grp_df.shape[0] < 10:
        print("  Not enough data to perform reliable analysis.")
        continue

    # Interaction between Group and mean_sts_score for Positive Change
    print("  SPANE Positive Change (with Interaction):")
    interaction_model_pos = smf.ols('SPANE_P_change ~ mean_sts_score * Group', data=grp_df).fit()
    print(interaction_model_pos.summary())

    # Interaction between Group and mean_sts_score for Negative Change
    print("  SPANE Negative Change (with Interaction):")
    interaction_model_neg = smf.ols('SPANE_N_change ~ mean_sts_score * Group', data=grp_df).fit()
    print(interaction_model_neg.summary())

# ============================================================
# 9. Visualizations for Overall Relationships
# ============================================================
plt.figure(figsize=(8, 6))
sns.regplot(x='mean_sts_score', y='SPANE_P_change', data=final_df, scatter_kws={'alpha': 0.5})
plt.title("Mean STS Score vs. SPANE Positive Change")
plt.xlabel("Mean STS Score")
plt.ylabel("SPANE Positive Change")
plt.show()

plt.figure(figsize=(8, 6))
sns.regplot(x='mean_sts_score', y='SPANE_N_change', data=final_df, scatter_kws={'alpha': 0.5})
plt.title("Mean STS Score vs. SPANE Negative Change")
plt.xlabel("Mean STS Score")
plt.ylabel("SPANE Negative Change")
plt.show()
