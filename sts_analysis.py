import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
#from scipy.stats import pearsonr
from matplotlib import pyplot as plt

# curl - L - O https: // github.com / conda - forge / miniforge / releases / latest / download / Miniforge3 - MacOSX - arm64.sh
# bash Miniforge3 - MacOSX - arm64.sh
# conda create --name tf_env python=3.10
# conda activate tf_env
# ----------------------------
# STEP 1: Load the Restructured Chat Data
# ----------------------------
# Update the filename if necessary
DATA_FILE = "paired_file.csv"
df = pd.read_csv(DATA_FILE)
print("Loaded conversation data with shape:", df.shape)

# ----------------------------
# STEP 3: Load Sentence Embedding Model
# ----------------------------
print("Loading Universal Sentence Encoder from TF Hub...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("Model loaded.")

# ----------------------------
# STEP 4: Compute STS Scores for Dyadic Pairs
# ----------------------------
def compute_sts_scores(sent_1_list, sent_2_list):
    # Compute embeddings and normalize them.
    embeddings1 = tf.nn.l2_normalize(embed(sent_1_list), axis=1)
    embeddings2 = tf.nn.l2_normalize(embed(sent_2_list), axis=1)
    # Compute cosine similarities.
    cosine_similarities = tf.reduce_sum(tf.multiply(embeddings1, embeddings2), axis=1)
    # Clip cosine similarities to ensure they lie between -1 and 1.
    clipped = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    # Convert cosine similarity to STS score.
    scores = 1.0 - tf.acos(clipped) / math.pi
    return scores.numpy()

# Process dyadic pairs in batches.
batch_size = 32
all_scores = []
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    scores = compute_sts_scores(batch['sent_1'].tolist(), batch['sent_2'].tolist())
    all_scores.extend(scores)

df['sts_score'] = all_scores

# ----------------------------
# STEP 5: Aggregate STS Scores by Condition
# ----------------------------
avg_sts = df.groupby('condition')['sts_score'].mean().reset_index()
print("Average STS scores by condition:")
print(avg_sts)

# Optionally, you can also compute the Pearson correlation with some gold labels
# if you have them. For now, we only compute the mean scores.

# ----------------------------
# STEP 6: Save and Visualize Results
# ----------------------------
# Save the dyadic pairs with their STS scores.
df.to_csv("dyadic_sts_scores.csv", index=False)
print("Dyadic pairs with STS scores saved to 'dyadic_sts_scores.csv'.")

# # Simple visualization of average STS scores per condition.
# plt.figure(figsize=(8, 6))
# plt.bar(avg_sts['condition'], avg_sts['sts_score'], color=['blue', 'orange', 'green'])
# plt.xlabel("Condition")
# plt.ylabel("Average STS Score")
# plt.title("Average STS Scores by Condition")
# plt.ylim([0, 1])
# plt.show()
