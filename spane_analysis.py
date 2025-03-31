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
# 1. Data Loading and Preprocessing
# ============================================================

# Load Precomputed STS Scores
user_mean_sts = pd.read_csv("user_mean_sts_scores.csv")
user_mean_sts.rename(columns={"sender_id": "SubID"}, inplace=True)

# Load Pre-Survey Data with Group Labels
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

# Load Post-Survey Data
post_survey_df = pd.read_csv("postExperiment_6231.csv")

# Select only the SPANE columns (there are 12 items)
spane_columns = ['SPANE_1', 'SPANE_2', 'SPANE_3', 'SPANE_4', 'SPANE_5', 'SPANE_6',
                 'SPANE_7', 'SPANE_8', 'SPANE_9', 'SPANE_10', 'SPANE_11', 'SPANE_12']
pre_survey_df = pre_survey_df[['SubID', 'Group'] + spane_columns]
post_survey_df = post_survey_df[['SubID'] + spane_columns]


# ============================================================
# 2. Compute SPANE Scores (Regrouping into Positive and Negative)
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
# 3. Merge Pre & Post Surveys and Compute Change Variables
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
# 4. Merge with STS Scores and Save Final DataFrame
# ============================================================
user_mean_sts['SubID'] = pd.to_numeric(user_mean_sts['SubID'], errors='coerce')
spane_change['SubID'] = pd.to_numeric(spane_change['SubID'], errors='coerce')
final_df = pd.merge(user_mean_sts, spane_change, on="SubID")

# Save merged dataset for future reference
final_df.to_csv("sts_spane_analysis.csv", index=False)
print("Saved: sts_spane_analysis.csv")

# ============================================================
# 5. Overall Analysis: STS Score vs. SPANE Changes (Positive & Negative separately)
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
# 6. Subgroup Analysis: Relationships by Group
# ============================================================
# Group: explore
# Positive Change:
# Correlation: 0.074 (p = 0.547)
# Regression Coefficient: 24.04 (p = 0.547)
# Interpretation: There is no significant relationship between mean STS score and positive affect change in the explore group.
# Negative Change:
# Correlation: –0.316 (p = 0.008)
# Regression Coefficient: –77.03 (p = 0.008)
# Interpretation: In the explore group, higher mean STS scores are significantly associated with a greater reduction in negative affect. A one-unit increase in mean STS score is associated with a drop of roughly 77 units in negative affect change, suggesting that in this subgroup, higher STS may help decrease negative feelings.

# Group: exploit
# Positive Change:
# Correlation: –0.116 (p = 0.340)
# Regression Coefficient: –36.71 (p = 0.340)
# Interpretation: There is no statistically significant relationship between mean STS score and positive affect change in the exploit group.
# Negative Change:
# Correlation: 0.146 (p = 0.228)
# Regression Coefficient: 35.74 (p = 0.228)
# Interpretation: Although the sign is reversed compared to the explore group, the relationship between STS score and negative affect change in the exploit group is not statistically significant.

# Group: control
# Positive Change:
# Correlation: –0.029 (p = 0.811)
# Regression Coefficient: –12.03 (p = 0.811)
# Interpretation: No significant relationship exists between mean STS score and positive affect change in the control group.
# Negative Change:
# Correlation: –0.114 (p = 0.349)
# Regression Coefficient: –28.40 (p = 0.349)
# Interpretation: There is also no significant relationship between mean STS score and negative affect change in the control group.
# groups = final_df['Group'].unique()
print("\n--- Subgroup Analysis ---")
for grp in groups:
    print(f"\nGroup: {grp}")
    grp_df = final_df[final_df['Group'] == grp]
    if grp_df.shape[0] < 10:
        print("  Not enough data to perform reliable analysis.")
        continue

    print("  SPANE Positive Change:")
    corr_grp_P, pval_grp_P = stats.pearsonr(grp_df['mean_sts_score'], grp_df['SPANE_P_change'])
    print(f"    Pearson Correlation: {corr_grp_P:.3f}, p-value: {pval_grp_P:.3f}")
    X_grp_P = sm.add_constant(grp_df['mean_sts_score'])
    model_grp_P = sm.OLS(grp_df['SPANE_P_change'], X_grp_P).fit()
    print(model_grp_P.summary())

    print("  SPANE Negative Change:")
    corr_grp_N, pval_grp_N = stats.pearsonr(grp_df['mean_sts_score'], grp_df['SPANE_N_change'])
    print(f"    Pearson Correlation: {corr_grp_N:.3f}, p-value: {pval_grp_N:.3f}")
    X_grp_N = sm.add_constant(grp_df['mean_sts_score'])
    model_grp_N = sm.OLS(grp_df['SPANE_N_change'], X_grp_N).fit()
    print(model_grp_N.summary())

# ============================================================
# 7. Visualizations for Overall Relationships
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

# ============================================================
# 8. Multiple Regression: Controlling for Pre-Survey Scores
# ============================================================
# For Positive Change: Predict SPANE_P_change from mean_sts_score and pre-survey positive score (SPANE_P_pre)
predictors_pos = ['mean_sts_score', 'SPANE_P_pre']
X_multi_pos = final_df[predictors_pos]
y_multi_pos = final_df['SPANE_P_change']
X_multi_pos = sm.add_constant(X_multi_pos)
multi_model_pos = sm.OLS(y_multi_pos, X_multi_pos).fit()
print("\nMultiple Linear Regression Results for SPANE Positive Change:")
print(multi_model_pos.summary())

# For Negative Change: Predict SPANE_N_change from mean_sts_score and pre-survey negative score (SPANE_N_pre)
predictors_neg = ['mean_sts_score', 'SPANE_N_pre']
X_multi_neg = final_df[predictors_neg]
y_multi_neg = final_df['SPANE_N_change']
X_multi_neg = sm.add_constant(X_multi_neg)
multi_model_neg = sm.OLS(y_multi_neg, X_multi_neg).fit()
print("\nMultiple Linear Regression Results for SPANE Negative Change:")
print(multi_model_neg.summary())
