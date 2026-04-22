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



# Check duplicates in tracker side
print("Tracker duplicate subjects:")
print(df_tracker[df_tracker.duplicated(subset=["subject"], keep=False)][["subject", "received date", "time_clean"]].head(10))

# Check how many df rows matched same comment
print("\nComment value counts:")
print(df["comment"].value_counts().head(10))



def find_comment(row, df_tracker, time_tolerance_mins=5, matched_indices=set()):
    
    subject_match = df_tracker[df_tracker["subject"] == row["subject"]]
    
    if subject_match.empty:
        return None
    
    for idx, t_row in subject_match.iterrows():
        
        # ✅ Skip already used tracker rows
        if idx in matched_indices:
            continue
        
        time_diff = abs((row["match_dt"] - t_row["match_dt"]).total_seconds() / 60)
        if time_diff <= time_tolerance_mins:
            matched_indices.add(idx)    # ✅ mark as used
            return t_row["comments"]
    
    return None

# ── Reset matched set and re-run ──────────────────────────────────────────────
matched_indices = set()

df["comment"] = df.apply(
    lambda row: find_comment(row, df_tracker, time_tolerance_mins=5, matched_indices=matched_indices),
    axis=1
)

# ── Match rate ────────────────────────────────────────────────────────────────
total     = len(df)
matched   = df["comment"].notna().sum()
unmatched = df["comment"].isna().sum()

print(f"✅ Total emails    : {total}")
print(f"✅ Matched         : {matched}   ({round(matched/total*100, 1)}%)")
print(f"⚠️  Unmatched       : {unmatched}  ({round(unmatched/total*100, 1)}%)")
print(f"📊 Tracker rows    : {len(df_tracker)}")
print(f"📊 Unique matched  : {len(matched_indices)}")




# ── Class distribution ────────────────────────────────────────────────────────
class_counts = df["comment"].value_counts(dropna=False)

print(f"Total Classes    : {df['comment'].nunique()}")
print(f"Total Classified : {df['comment'].notna().sum()}")
print(f"Unclassified     : {df['comment'].isna().sum()}")
print(f"\n── Class Counts ──────────────────────────────")
print(class_counts.to_string())



# ── Separate each class ───────────────────────────────────────────────────────
df_dsd     = df[df["comment"] == "DSD   Acknowledgement"].copy()
df_followup= df[df["comment"] == "For Follow up"].copy()
df_argus   = df[df["comment"] == "Argus ID"].copy()

print(f"DSD Acknowledgement : {len(df_dsd)}")
print(f"For Follow up       : {len(df_followup)}")
print(f"Argus ID            : {len(df_argus)}")

from collections import Counter
import re

def get_top_words(df_class, col="bodyPreview", top_n=30):
    
    # Combine all text
    all_text = " ".join(df_class[col].dropna().tolist()).lower()
    
    # Remove common stop words
    stop_words = {
        "the","is","in","it","of","and","to","a","an","that","this",
        "for","on","are","was","with","as","at","be","by","from",
        "have","has","had","not","but","or","you","we","i","re",
        "your","our","please","thank","thanks","dear","hi","hello",
        "regards","mail","email","will","would","could","should"
    }
    
    words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
    words = [w for w in words if w not in stop_words]
    
    return Counter(words).most_common(top_n)

print("\n── DSD Acknowledgement Top Words ────────────────")
print(get_top_words(df_dsd))

print("\n── For Follow Up Top Words ──────────────────────")
print(get_top_words(df_followup))

print("\n── Argus ID Top Words ───────────────────────────")
print(get_top_words(df_argus))

def get_top_subjects(df_class, top_n=10):
    return df_class["subject"].value_counts().head(top_n)

print("\n── DSD Acknowledgement Top Subjects ─────────────")
print(get_top_subjects(df_dsd))

print("\n── For Follow Up Top Subjects ────────────────────")
print(get_top_subjects(df_followup))

print("\n── Argus ID Top Subjects ─────────────────────────")
print(get_top_subjects(df_argus))



def show_samples(df_class, col="bodyPreview", n=5):
    samples = df_class[col].dropna().head(n).tolist()
    for i, s in enumerate(samples):
        print(f"\n── Sample {i+1} ───────────────────────────────────")
        print(s[:300])

print("\n═══ DSD Acknowledgement Samples ═════════════════")
show_samples(df_dsd)

print("\n═══ For Follow Up Samples ════════════════════════")
show_samples(df_followup)

print("\n═══ Argus ID Samples ═════════════════════════════")
show_samples(df_argus)



print("── DSD Acknowledgement Top Senders ──────────────")
print(df_dsd["sender_email"].value_counts().head(10))

print("\n── For Follow Up Top Senders ─────────────────────")
print(df_followup["sender_email"].value_counts().head(10))

print("\n── Argus ID Top Senders ──────────────────────────")
print(df_argus["sender_email"].value_counts().head(10))





# ── Run this entire cell at once ──────────────────────────────────────────────
from collections import Counter
import re

def get_word_counts(df_class, col="bodyPreview"):
    all_text = " ".join(df_class[col].dropna().tolist()).lower()
    
    stop_words = {
        "the","is","in","it","of","and","to","a","an","that","this",
        "for","on","are","was","with","as","at","be","by","from",
        "have","has","had","not","but","or","you","we","i","re",
        "your","our","please","thank","thanks","dear","hi","hello",
        "regards","mail","email","will","would","could","should",
        "just","also","get","can","one","all","any","been","when",
        "they","them","their","there","here","which","more","than"
    }
    
    words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
    words = [w for w in words if w not in stop_words]
    return Counter(words)

# ── Step 1: Build counters ────────────────────────────────────────────────────
counter_dsd      = get_word_counts(df_dsd)
counter_followup = get_word_counts(df_followup)
counter_argus    = get_word_counts(df_argus)

print("✅ Counters built")
print(f"   DSD unique words     : {len(counter_dsd)}")
print(f"   Follow Up unique words: {len(counter_followup)}")
print(f"   Argus unique words   : {len(counter_argus)}")

# ── Step 2: Common words across all 3 ────────────────────────────────────────
common_across_all = set(counter_dsd.keys()) & set(counter_followup.keys()) & set(counter_argus.keys())
print(f"\n✅ Common words ignored : {len(common_across_all)}")

# ── Step 3: Filter function ───────────────────────────────────────────────────
def get_unique_words(counter, df_class, common_words, threshold_pct=0.3, top_n=20):
    class_size    = len(df_class)
    min_frequency = class_size * threshold_pct

    filtered = {
        word: count
        for word, count in counter.items()
        if word not in common_words
        and count >= min_frequency
    }
    
    return sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]

# ── Step 4: Print results ─────────────────────────────────────────────────────
print(f"\n── DSD Acknowledgement ({len(df_dsd)} emails) ────────────────")
print(f"   Min frequency : {int(len(df_dsd) * 0.3)}")
for word, count in get_unique_words(counter_dsd, df_dsd, common_across_all):
    print(f"   {word:<20} {count:>5}  ({round(count/len(df_dsd)*100, 1)}%)")

print(f"\n── For Follow Up ({len(df_followup)} emails) ─────────────────")
print(f"   Min frequency : {int(len(df_followup) * 0.3)}")
for word, count in get_unique_words(counter_followup, df_followup, common_across_all):
    print(f"   {word:<20} {count:>5}  ({round(count/len(df_followup)*100, 1)}%)")

print(f"\n── Argus ID ({len(df_argus)} emails) ────────────────────────")
print(f"   Min frequency : {int(len(df_argus) * 0.3)}")
for word, count in get_unique_words(counter_argus, df_argus, common_across_all):
    print(f"   {word:<20} {count:>5}  ({round(count/len(df_argus)*100, 1)}%)")






# ── Check 1: Is bodyPreview populated ─────────────────────────────────────────
print("DSD bodyPreview null count  :", df_dsd["bodyPreview"].isna().sum())
print("DSD bodyPreview sample      :", df_dsd["bodyPreview"].iloc[0])

# ── Check 2: Are counters actually populated ──────────────────────────────────
print("\nTop 10 DSD raw counter:")
print(counter_dsd.most_common(10))

# ── Check 3: How many words survive common filter ─────────────────────────────
dsd_after_common = {w: c for w, c in counter_dsd.items() if w not in common_across_all}
print(f"\nDSD words after removing common : {len(dsd_after_common)}")
print("Top 10 after common filter:")
print(sorted(dsd_after_common.items(), key=lambda x: x[1], reverse=True)[:10])

# ── Check 4: What is the min frequency cutting off ────────────────────────────
min_freq = len(df_dsd) * 0.3
print(f"\nDSD min frequency threshold : {min_freq}")
print(f"Max word count in DSD       : {max(dsd_after_common.values()) if dsd_after_common else 0}")

# ── Check 5: Lower threshold to 0 to see anything ────────────────────────────
print("\nDSD top words with 0 threshold:")
for word, count in get_unique_words(counter_dsd, df_dsd, common_across_all, threshold_pct=0.0):
    print(f"   {word:<20} {count:>5}  ({round(count/len(df_dsd)*100, 1)}%)")
