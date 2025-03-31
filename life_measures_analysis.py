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

# Calculate mean STS scores for users by grouping by sender_id (or receiver_id if needed)
user_mean_sts = dyadic_df.groupby('sender_id')['sts_score'].mean().reset_index()

# Rename columns to match the expected format in user_mean_sts_scores.csv
user_mean_sts.rename(columns={"sender_id": "SubID", "sts_score": "mean_sts_score"}, inplace=True)

# ============================================================
# 2. Load Post-Survey Data
# ============================================================
post_survey_df = pd.read_csv("postExperiment_6231.csv")

# Define new three measures categories
social_connectedness = ["Q69_1", "Q69_2", "Q69_3", "Q69_4", "Q69_5", "Q69_6"]
satisfaction_life = ["Q68_6", "Q68_7", "Q68_9", "Q68_10", "Q68_15"]
psychological_richness = ["Q68_1", "Q68_3", "Q68_5", "Q68_12", "Q68_14"]

# Keep only relevant columns from post-survey
post_survey_df = post_survey_df[['SubID'] + social_connectedness + satisfaction_life + psychological_richness]

# ============================================================
# 3. Compute Scores for the Three Measures
# ============================================================
def compute_measure(df, measure_columns, measure_name):
    df[measure_columns] = df[measure_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=measure_columns, inplace=True)
    df[measure_name] = df[measure_columns].mean(axis=1)
    return df[['SubID', measure_name]]

# Compute scores for post-surveys only
post_social = compute_measure(post_survey_df.copy(), social_connectedness, 'social_connectedness')
post_satisfaction = compute_measure(post_survey_df.copy(), satisfaction_life, 'satisfaction_life')
post_psychological = compute_measure(post_survey_df.copy(), psychological_richness, 'psychological_richness')

# ============================================================
# 4. Merge Post-Survey Measures
# ============================================================
total_change = post_social.merge(post_satisfaction, on='SubID').merge(post_psychological, on='SubID')

# ============================================================
# 5. Merge with STS Scores and Save Final DataFrame
# ============================================================
# Ensure 'SubID' is numeric for both dataframes
user_mean_sts['SubID'] = pd.to_numeric(user_mean_sts['SubID'], errors='coerce')
total_change['SubID'] = pd.to_numeric(total_change['SubID'], errors='coerce')

# Merge STS scores with post-survey data
final_df = pd.merge(user_mean_sts, total_change, on="SubID")

# ============================================================
# 6. Merge with Dyadic Data Condition Column (Already existing as 'condition')
# ============================================================
# Ensure 'condition' is used directly from dyadic data (already exists)
final_df = pd.merge(final_df, dyadic_df[['sender_id', 'condition']], left_on='SubID', right_on='sender_id', how='left')
final_df.drop(columns='sender_id', inplace=True)  # Remove 'sender_id' after merging

# ============================================================
# 7. Save merged dataset
# ============================================================
final_df.to_csv("three_measures_analysis_baseline.csv", index=False)
print("Saved: three_measures_analysis_baseline.csv")

# ============================================================
# 8. Compute SPANE Scores (SPANE_P and SPANE_N)
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
# 5. Merge SPANE Scores with Final DataFrame
# ============================================================
# Merge pre and post SPANE scores with group info
# Ensure 'SubID' columns are of the same type (convert to int64)
final_df['SubID'] = pd.to_numeric(final_df['SubID'], errors='coerce', downcast='integer')
pre_spane['SubID'] = pd.to_numeric(pre_spane['SubID'], errors='coerce', downcast='integer')

# Merge pre and post SPANE scores with group info
final_df = pd.merge(final_df, pre_spane, on='SubID', how='left', suffixes=('_pre', '_post'))

# ============================================================
# 9. Analysis: Relationship Between SPANE (P and N) and Post-Survey Measures
# ============================================================

# Correlation between SPANE (SPANE_P, SPANE_N) and the three post-survey measures
correlations_spane = final_df[['SPANE_P', 'SPANE_N', 'social_connectedness', 'satisfaction_life', 'psychological_richness']].corr()
print("Correlations between SPANE (P, N) and Post-Survey Measures:")
print(correlations_spane)

# ============================================================
# 10. Regression Analysis: SPANE_P and SPANE_N as predictors of Post-survey measures
# ============================================================

# SPANE_P and SPANE_N predicting social connectedness
model_social_spane = smf.ols('social_connectedness ~ SPANE_P + SPANE_N', data=final_df).fit()
print("\nRegression: SPANE_P and SPANE_N -> Social Connectedness")
print(model_social_spane.summary())

# SPANE_P and SPANE_N predicting satisfaction life
model_satisfaction_spane = smf.ols('satisfaction_life ~ SPANE_P + SPANE_N', data=final_df).fit()
print("\nRegression: SPANE_P and SPANE_N -> Satisfaction Life")
print(model_satisfaction_spane.summary())

# SPANE_P and SPANE_N predicting psychological richness
model_psychological_spane = smf.ols('psychological_richness ~ SPANE_P + SPANE_N', data=final_df).fit()
print("\nRegression: SPANE_P and SPANE_N -> Psychological Richness")
print(model_psychological_spane.summary())

# ============================================================
# 11. Moderation Analysis: Check for Moderation Effects (Interaction term)
# ============================================================

# Testing moderation effect between SPANE and Post-survey Measures (e.g., social connectedness)
moderation_model = smf.ols('social_connectedness ~ SPANE_P + SPANE_N + condition', data=final_df).fit()
print("\nModeration Analysis: SPANE_P and SPANE_N * condition on Social Connectedness")
print(moderation_model.summary())


# ============================================================
# 11. Analysis: Three Measures Across Conditions
# ============================================================

# Summary Statistics by Condition
summary_stats = final_df.groupby('condition')[['social_connectedness', 'satisfaction_life', 'psychological_richness']].describe()
print("Summary Statistics by Condition:\n", summary_stats)

# One-way ANOVA to test for significant differences across conditions
for measure in ['social_connectedness', 'satisfaction_life', 'psychological_richness']:
    anova_model = smf.ols(f"{measure} ~ C(condition)", data=final_df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    print(f"\nANOVA for {measure} across conditions:")
    print(anova_table)

# ============================================================
# 12. OLS the Three Measures Across Conditions
# ============================================================

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Run regression for each outcome variable
satisfaction_model = smf.ols("satisfaction_life ~ C(condition)", data=final_df).fit()
richness_model = smf.ols("psychological_richness ~ C(condition)", data=final_df).fit()
connectedness_model = smf.ols("social_connectedness ~ C(condition)", data=final_df).fit()


def extract_stats(model, outcome_name):
    """Extracts key statistics and formats them."""
    results = model.params
    t_values = model.tvalues
    p_values = model.pvalues

    for condition in results.index[1:]:  # Skipping the intercept
        beta = results[condition]
        t_stat = t_values[condition]
        p_val = p_values[condition]
        p_text = "p < .01" if p_val < 0.05 else f"p = {p_val:.2f}"

        print(f"{outcome_name}: The {condition} condition was perceived as more {outcome_name} "
              f"(β = {beta:.2f}, t = {t_stat:.2f}, {p_val},{p_text}).")


# Format results for each model
extract_stats(satisfaction_model, "satisfaction")
extract_stats(richness_model, "psychological richness")
extract_stats(connectedness_model, "connectedness")
