import pandas as pd
# Read the CSV file into a DataFrame
df = pd.read_csv('restructured_chat_data.csv')

# Read the CSV file into a DataFrame

# Initialize a list to store new pairs
new_pairs = []

# Group the DataFrame by chat_id and condition to process each conversation separately
grouped = df.groupby(['chat_id', 'condition'])

# Process each chat group
for (chat_id, condition), group in grouped:
    # Sort the messages by session_time (it should already be sorted, but to be sure)
    group = group.sort_values(by=['session_time'])

    # Iterate over each message in the group
    for i in range(len(group)):
        message1 = group.iloc[i]

        # Loop through the messages that come after the current message (message1)
        for j in range(i + 1, len(group)):
            message2 = group.iloc[j]

            # Check if the sender of message2 is the receiver of message1 and vice versa
            if message1['receiver_id'] == message2['sender_id'] and message1['sender_id'] == message2['receiver_id']:
                # Add the pair to the new_pairs list
                new_pairs.append([
                    message1['chat_id'],
                    message1['condition'],
                    message1['message'],
                    message2['message'],
                    message1['sender_id'],
                    message1['receiver_id'],
                    message2['session_time'],  # Use message2's session_time as the pair's time
                    message2['segment']
                ])
                break  # Stop looking for a reply for message1 after finding the first valid pair

# Create a new DataFrame with the new pairs
new_df = pd.DataFrame(new_pairs, columns=[
    'chat_id', 'condition', 'message1', 'message2', 'sender_id', 'receiver_id', 'session_time', 'segment'
])

# Save the new DataFrame to a CSV
new_df.to_csv('paired_file.csv', index=False)