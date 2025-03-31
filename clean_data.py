import pandas as pd
import numpy as np


# ----------------------------
# STEP 1: Load & Reorder CSV Data
# ----------------------------
def load_data(filename, condition):
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

    if len(df.columns) < 8:
        raise ValueError("Unexpected number of columns in the CSV. Expected at least 8 columns.")

    # Reorder and rename columns
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
def combine_data():
    control = load_data("controlChatroom_6231.csv", "control")
    exploit = load_data("exploitChatroom_6231.csv", "exploit")
    explore = load_data("exploreChatroom_6231.csv", "explore")

    # Combine the datasets into one DataFrame.
    df = pd.concat([control, exploit, explore], ignore_index=True)
    return df


# ----------------------------
# STEP 3: Process Receiver Info and Add New Columns
# ----------------------------
def process_receiver_info(df):
    # Clean the receiver_info field by removing extra quotes.
    df['receiver_info'] = df['receiver_info'].str.replace('"', '', regex=False).str.strip()

    # Extract sender and receiver information
    def extract_sender_receiver(group):
        # Split the receiver_info by '_'
        split_info = group['receiver_info'].str.split('_')

        # Extract sender and receiver from the split parts
        group['extracted_sender'] = split_info.str[0]
        group['receiver_number'] = split_info.str[1]

        # If the sender_number is not in extracted_sender, swap sender/receiver
        condition = group['sender_number'] != group['extracted_sender']
        group.loc[condition, ['extracted_sender', 'receiver_number']] = group.loc[
            condition, ['receiver_number', 'extracted_sender']].values

        return group

    # Apply the extraction to each group (grouped by chat_id and condition)
    df = df.groupby(['chat_id', 'condition'], group_keys=False).apply(extract_sender_receiver)

    # ----------------------------
    # Now map the receiver_number to receiver_id based on sender_number
    def map_receiver_id(group):
        # Create a mapping for sender_number to sender_id within the group
        unique_sender_mapping = group.drop_duplicates(subset=['sender_number'])[['sender_number', 'sender_id']]

        # Create a dictionary to map sender_number to sender_id within the group
        sender_to_receiver_id_map = unique_sender_mapping.set_index('sender_number')['sender_id'].to_dict()

        # Apply the mapping within the group
        group['receiver_id'] = group['receiver_number'].map(sender_to_receiver_id_map)
        group['receiver_id'] = group['receiver_id'].fillna('Unknown')  # Handle missing values if necessary

        return group

    # Apply the mapping function to each group (grouped by chat_id and condition)
    df = df.groupby(['chat_id', 'condition'], group_keys=False).apply(map_receiver_id)

    return df


# ----------------------------
# STEP 4: Convert Timestamps
# ----------------------------
def convert_timestamps(df):
    # Convert abs_time to numeric (assumed to be in milliseconds)
    df['abs_time'] = pd.to_numeric(df['abs_time'], errors='coerce')
    df = df.dropna(subset=['abs_time'])
    return df


# ----------------------------
# STEP 5: Calculate Session Time Within Each Chat Session by Condition
# ----------------------------
def calculate_session_time(df):
    df['timestamp'] = df['abs_time']  # Rename for clarity (timestamp in ms)
    df['session_time'] = df.groupby(['chat_id', 'condition'])['timestamp'].transform(
        lambda x: (x - x.min()) / (1000 * 60)  # Convert ms to minutes
    )
    # Calculate the total session duration for each chat/condition.
    df['total_session_time'] = df.groupby(['chat_id', 'condition'])['session_time'].transform('max')
    return df


# ----------------------------
# STEP 6: Create Time Segments
# ----------------------------
def create_time_segments(df):
    # Bin session_time into segments (early, middle, late).
    df['segment'] = pd.cut(df['session_time'], bins=[0, 10, 20, np.inf],
                           labels=['early', 'middle', 'late'],
                           include_lowest=True)
    return df


# ----------------------------
# STEP 7: Final Cleanup and Save Processed Data with New Column Order
# ----------------------------
def save_processed_data(df):
    # Reorder the columns to match the required structure
    df = df[['chat_id', 'channel_identifier', 'condition', 'message',
             'sender_id', 'receiver_id',
             'session_time', 'segment']]

    # Save to a new CSV file
    df.to_csv("restructured_chat_data.csv", index=False)
    print("Processed data saved successfully with new structure.")


# Combine all the data cleaning steps
def clean_data():
    df = combine_data()
    df = process_receiver_info(df)
    df = convert_timestamps(df)
    df = calculate_session_time(df)
    df = create_time_segments(df)
    save_processed_data(df)
    return df


clean_data()
