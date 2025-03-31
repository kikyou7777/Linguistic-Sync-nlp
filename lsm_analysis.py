import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from statsmodels.formula.api import ols

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Lowercase, remove punctuation, tokenize, and remove stopwords."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    filtered = [token for token in tokens if token not in stop_words]
    return " ".join(filtered)

# List of chat CSV files with their corresponding condition labels
chat_files = [
    ("controlChatroom_6231.csv", "control"),
    ("exploreChatroom_6231.csv", "explore"),
    ("exploitChatroom_6231.csv", "exploit")
]

chat_dfs = []
for file, condition in chat_files:
    # Read CSV with semicolon delimiter
    df = pd.read_csv(file, delimiter=";", encoding="utf-8")
    df["condition"] = condition
    chat_dfs.append(df)

# Combine all chat data
chat_data = pd.concat(chat_dfs, ignore_index=True)
print(f"Combined chat data shape: {chat_data.shape}")

# Use the "ID" column as the merge key; rename it to "SubID" to match survey data
chat_data.rename(columns={"ID": "SubID", "Message": "message"}, inplace=True)

# Preprocess chat messages
chat_data["processed_text"] = chat_data["message"].fillna("").apply(preprocess)

# Function to compute LSM (average cosine similarity) for a list of messages from one participant
def compute_lsm(messages):
    if len(messages) < 2:
        return np.nan  # Not enough messages to compute similarity
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(messages)
    dense = X.toarray()
    scores = []
    for i in range(1, len(dense)):
        # Skip if either vector is all zeros
        if np.all(dense[i] == 0) or np.all(dense[i-1] == 0):
            continue
        sim = 1 - cosine(dense[i], dense[i-1])
        scores.append(sim)
    return np.nanmean(scores) if scores else np.nan

# Compute LSM score for each participant (grouped by SubID)
lsm_scores = chat_data.groupby("SubID")["processed_text"].apply(lambda texts: compute_lsm(list(texts))).reset_index()
lsm_scores.rename(columns={"processed_text": "lsm_score"}, inplace=True)
print(f"LSM scores computed for {lsm_scores.shape[0]} participants.")

# Load survey data (postExperiment file)
survey_data = pd.read_csv("postExperiment_6231.csv", encoding="utf-8")
print(f"Survey data shape: {survey_data.shape}")

# Debug: Print unique merge keys in both datasets
print("Unique SubIDs in chat data:", chat_data["SubID"].unique())
print("Unique SubIDs in survey data:", survey_data["SubID"].unique())

# Convert merge key columns to string type
lsm_scores["SubID"] = lsm_scores["SubID"].astype(str)
survey_data["SubID"] = survey_data["SubID"].astype(str)

# Compute overall SPANE score from SPANE_1 to SPANE_12 columns (convert to numeric)
spane_cols = [col for col in survey_data.columns if col.startswith("SPANE_")]
if not spane_cols:
    raise ValueError("No SPANE columns found in survey data.")
survey_data[spane_cols] = survey_data[spane_cols].apply(pd.to_numeric, errors='coerce')
survey_data["spane_score"] = survey_data[spane_cols].mean(axis=1)

# Merge the LSM scores with survey data on "SubID"
merged_data = pd.merge(lsm_scores, survey_data, on="SubID", how="inner")
print(f"Merged data shape: {merged_data.shape}")

# Drop rows with missing values in lsm_score or spane_score
merged_data.dropna(subset=["lsm_score", "spane_score"], inplace=True)
print(f"Merged data shape after dropping missing values: {merged_data.shape}")

if merged_data.empty:
    raise ValueError("Merged data is empty. Check merge keys and file contents.")

# Regression analysis: Does LSM predict well-being (spane_score)?
model = ols("spane_score ~ lsm_score", data=merged_data).fit()
print(model.summary())
