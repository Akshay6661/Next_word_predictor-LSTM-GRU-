import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')


##new

# ── df — use EST date and time columns ────────────────────────────────────────
df["match_dt"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["time"].astype(str)
                 )

# ── df_tracker — strip day name and parse 12hr format ────────────────────────
df_tracker["time_clean"] = df_tracker["received time"].str.replace(
                                r"^[A-Za-z]{3}\s+", "", regex=True
                           ).str.strip()

df_tracker["match_dt"]   = pd.to_datetime(
                                df_tracker["received date"].astype(str) + " " + 
                                df_tracker["time_clean"].astype(str),
                                format="%Y-%m-%d %I:%M %p"
                            )

# ── Quick check both are now EST and aligned ──────────────────────────────────
print("df match_dt         :", df["match_dt"].head(3).tolist())
print("df_tracker match_dt :", df_tracker["match_dt"].head(3).tolist())


def find_comment(row, df_tracker, time_tolerance_mins=5):
    
    # Exact subject match first
    subject_match = df_tracker[df_tracker["subject"] == row["subject"]]
    
    if subject_match.empty:
        return None
    
    # Then time difference check
    for _, t_row in subject_match.iterrows():
        time_diff = abs((row["match_dt"] - t_row["match_dt"]).total_seconds() / 60)
        if time_diff <= time_tolerance_mins:
            return t_row["comments"]
    
    return None

df["comment"] = df.apply(
    lambda row: find_comment(row, df_tracker, time_tolerance_mins=5), axis=1
)

# ── Match rate ────────────────────────────────────────────────────────────────
total     = len(df)
matched   = df["comment"].notna().sum()
unmatched = df["comment"].isna().sum()

print(f"✅ Total emails : {total}")
print(f"✅ Matched      : {matched}   ({round(matched/total*100, 1)}%)")
print(f"⚠️  Unmatched    : {unmatched}  ({round(unmatched/total*100, 1)}%)")

df[["subject", "date", "time", "comment"]].head(10)

# Pick one unmatched email and see why it didn't match
sample = df[df["comment"].isna()].iloc[0]
print("Subject    :", sample["subject"])
print("match_dt   :", sample["match_dt"])

# Check if subject exists in tracker at all
print("\nTracker matches on subject:")
print(df_tracker[df_tracker["subject"] == sample["subject"]][["subject", "received date", "received time"]])



# ── df_tracker — strip day name then let pandas infer format ──────────────────
df_tracker["time_clean"] = df_tracker["received time"].str.replace(
                                r"^[A-Za-z]{3}\s+", "", regex=True
                           ).str.strip()

# ✅ Remove format= and use infer_datetime_format instead
df_tracker["match_dt"]   = pd.to_datetime(
                                df_tracker["received date"].astype(str) + " " + 
                                df_tracker["time_clean"].astype(str),
                                infer_datetime_format=True
                            )

# ── df — same ─────────────────────────────────────────────────────────────────
df["match_dt"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["time"].astype(str),
                    infer_datetime_format=True
                 )

# ── Quick check ───────────────────────────────────────────────────────────────
print("df match_dt         :", df["match_dt"].head(3).tolist())
print("df_tracker match_dt :", df_tracker["match_dt"].head(3).tolist())
