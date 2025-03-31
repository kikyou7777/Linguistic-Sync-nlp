import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nrclex import NRCLex
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt


# ----------------------------
# STEP 1: Load & Reorder CSV Data
# ----------------------------
def load_data(filename, condition):
    # Read CSV file using comma as delimiter.
    df = pd.read_csv(
        filename,
        delimiter=",",
        quotechar='"',
        header=0,
        on_bad_lines="skip",
        dtype=str  # Read all data as strings
    )

    # Clean up column names.
    df.columns = [col.strip() for col in df.columns if col.strip() != '']

    # We expect the original CSV to have 8 columns:
    #   Column 0: Id
    #   Column 1: Channel Identifier
    #   Column 2: Abs. Time
    #   Column 3: Sender
    #   Column 4: Message
    #   Column 5: ID        -> This contains the composite receiver info (e.g. """P.1_P.4""")
    #   Column 6: (numeric) -> Sender ID
    #   Column 7: Avatar    -> Sender Avatar
    if len(df.columns) < 8:
        raise ValueError("Unexpected number of columns in the CSV. Expected at least 8 columns.")

    # Reorder and rename columns to the desired order:
    # Desired order:
    #   chat_id, channel_identifier, abs_time, sender_number, message,
    #   receiver_info, sender_id, sender_avatar, condition
    #
    # Mapping:
    #   - chat_id: from original "Channel Identifier" (column 1)
    #   - channel_identifier: from original "Id" (column 0)
    #   - abs_time: from original "Abs. Time" (column 2)
    #   - sender_number: from original "Sender" (column 3)
    #   - message: from original "Message" (column 4)
    #   - receiver_info: from original "ID" (column 5)
    #   - sender_id: from original column 6 (numeric sender id)
    #   - sender_avatar: from original column 7 (sender avatar image file)
    df = df.iloc[:, :8]  # take only the first 8 columns
    new_order = {
        'chat_id': df.iloc[:, 1],  # Column 1: Channel Identifier
        'channel_identifier': df.iloc[:, 0],  # Column 0: Id
        'abs_time': df.iloc[:, 2],  # Column 2: Abs. Time
        'sender_number': df.iloc[:, 3],  # Column 3: Sender
        'message': df.iloc[:, 4],  # Column 4: Message
        'receiver_info': df.iloc[:, 5],  # Column 5: ID (composite)
        'sender_id': df.iloc[:, 6],  # Column 6: Sender ID
        'sender_avatar': df.iloc[:, 7]  # Column 7: Avatar
    }
    df = pd.DataFrame(new_order)

    # Add the condition column from the provided parameter.
    df["condition"] = condition
    return df


# ----------------------------
# STEP 2: Combine Data from Different Conditions
# ----------------------------
control = load_data("controlChatroom_6231.csv", "control")
exploit = load_data("exploitChatroom_6231.csv", "exploit")
explore = load_data("exploreChatroom_6231.csv", "explore")

# Combine the datasets into one DataFrame.
df = pd.concat([control, exploit, explore], ignore_index=True)
print("Initial df shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Sample of data:\n", df.head())

# ----------------------------
# STEP 3: Process Receiver Info and Add New Columns
# ----------------------------
# Clean the receiver_info field by removing extra quotes.
# E.g., convert '"""P.1_P.4"""' to 'P.1_P.4'
df['receiver_info'] = df['receiver_info'].str.replace('"', '', regex=False).str.strip()

# Extract receiver_number from receiver_info.
# We assume the format is "P.X_P.Y" where P.Y is the receiver.
df['receiver_number'] = df['receiver_info'].str.split('_').str[1]

# Optionally, extract the sender from receiver_info to verify it matches sender_number.
df['extracted_sender'] = df['receiver_info'].str.split('_').str[0]

# ----------------------------
# STEP 4: Convert Timestamps
# ----------------------------
# Convert abs_time to numeric (assumed to be in milliseconds)
df['abs_time'] = pd.to_numeric(df['abs_time'], errors='coerce')
df = df.dropna(subset=['abs_time'])
print("Shape after dropping invalid timestamps:", df.shape)


# ----------------------------
# STEP 5: Analyze Emotions in Messages Using NRCLex
# ----------------------------
def analyze_emotions(text):
    emotion = NRCLex(text)
    return emotion.affect_frequencies


# Ensure 'message' column is of string type
df['message'] = df['message'].astype(str)

# Apply the analyze_emotions function to each message
df['emotion_scores'] = df['message'].apply(analyze_emotions)

# Convert emotion_scores dictionaries into a DataFrame
emotion_df = pd.json_normalize(df['emotion_scores'])

# Merge the emotion DataFrame back into the original DataFrame
df = pd.concat([df, emotion_df], axis=1)

# Drop the original emotion_scores column
df = df.drop(columns=['emotion_scores'])
print("Sample emotion scores:\n", df.head())

# ----------------------------
# STEP 6: Calculate Session Time Within Each Chat Session by Condition
# ----------------------------
# We now group by both 'chat_id' and 'condition' so that sessions are computed per chat per condition.
df['timestamp'] = df['abs_time']  # Rename for clarity (timestamp in ms)
df['session_time'] = df.groupby(['chat_id', 'condition'])['timestamp'].transform(
    lambda x: (x - x.min()) / (1000 * 60)  # Convert ms to minutes
)
# Also calculate the total session duration for each chat/condition.
df['total_session_time'] = df.groupby(['chat_id', 'condition'])['session_time'].transform('max')
print("Sample session times:\n",
      df[['chat_id', 'condition', 'session_time', 'total_session_time']].drop_duplicates().head())

# ----------------------------
# STEP 7: Create Time Segments
# ----------------------------
# Bin session_time into segments (early, middle, late).
df['segment'] = pd.cut(df['session_time'], bins=[0, 10, 20, np.inf],
                       labels=['early', 'middle', 'late'],
                       include_lowest=True)
print("Segment value counts:\n", df['segment'].value_counts())

# ----------------------------
# STEP 8: Map Receiver Information
# ----------------------------
# Identify whether a message is directed to a single person or a group
df['is_group_message'] = df['receiver_number'].str.contains('_', na=False)

# Count unique receivers per message
df['num_receivers'] = df['receiver_info'].apply(lambda x: len(str(x).split('_')))

# ----------------------------
# STEP 9: Statistical Analysis
# ----------------------------
# Perform a simple linear regression to check if session time influences emotion scores
emotion_cols = ['fear', 'anger', 'anticip', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy']
for emotion in emotion_cols:
    formula = f"{emotion} ~ session_time"
    model = smf.ols(formula=formula, data=df).fit()
    print(f"Regression results for {emotion}:")
    print(model.summary())

# ----------------------------
# STEP 10: Visualizations
# ----------------------------
# Plot average emotion scores over session time
plt.figure(figsize=(12, 6))
for emotion in emotion_cols:
    df.groupby('session_time')[emotion].mean().plot(label=emotion)

plt.xlabel('Session Time (Minutes)')
plt.ylabel('Average Emotion Score')
plt.title('Emotion Scores Over Time')
plt.legend()
plt.show()

# ----------------------------
# STEP 11: Save Processed Data
# ----------------------------
df.to_csv("processed_chat_data.csv", index=False)
print("Processed data saved successfully.")
