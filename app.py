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


# ── Sort by conversation and reply position ───────────────────────────────────
df_body = df.sort_values(["conversationId", "reply_position"]).reset_index(drop=True).copy()

def extract_new_content(df_input):
    
    df_output = df_input.copy()
    df_output["actual_body"] = ""
    
    for conv_id, group in df_output.groupby("conversationId"):
        
        group    = group.sort_values("reply_position")
        prev_body = ""
        
        for idx, row in group.iterrows():
            
            current_body = row["body_full"] if pd.notna(row["body_full"]) else ""
            
            if row["reply_position"] == 1:
                df_output.at[idx, "actual_body"] = current_body
                
            else:
                if prev_body:
                    overlap_pos = current_body.lower().find(prev_body[:100].lower())
                    if overlap_pos > 0:
                        df_output.at[idx, "actual_body"] = current_body[:overlap_pos].strip()
                    else:
                        df_output.at[idx, "actual_body"] = current_body
                else:
                    df_output.at[idx, "actual_body"] = current_body
            
            prev_body = current_body
    
    return df_output

df_actual_body = extract_new_content(df_body)

print(f"✅ Saved as df_actual_body")
print(f"   Shape          : {df_actual_body.shape}")
print(f"   Empty bodies   : {(df_actual_body['actual_body'] == '').sum()}")
print(f"   Filled bodies  : {(df_actual_body['actual_body'] != '').sum()}")





import html
import re

def clean_actual_body(text):
    if not text or pd.isna(text):
        return ""
    
    # ── Step 1: Decode HTML entities ──────────────────────────────────────────
    text = html.unescape(text)                        # &nbsp; &gt; &amp; → actual chars
    
    # ── Step 2: Remove remaining HTML tags ────────────────────────────────────
    text = re.sub(r"<[^>]+>", " ", text)
    
    # ── Step 3: Remove specific HTML entities that survived ───────────────────
    text = re.sub(r"&[a-zA-Z]+;",  " ", text)        # &nbsp; &gt; &lt; &amp;
    text = re.sub(r"&#[0-9]+;",    " ", text)        # &#160; &#43;
    text = re.sub(r"&[#a-zA-Z0-9]+;", " ", text)    # catch anything remaining
    
    # ── Step 4: Remove special characters and garbage ─────────────────────────
    text = re.sub(r"[;:<>{}\[\]|\\]", " ", text)    # ; : < > { } [ ]
    text = re.sub(r"[^\x00-\x7F]+",   " ", text)    # non ASCII characters
    text = re.sub(r"http\S+",          " ", text)    # URLs
    text = re.sub(r"\S+@\S+",          " ", text)    # email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)  # emails
    
    # ── Step 5: Remove leftover single characters and numbers only tokens ──────
    text = re.sub(r"\b[a-zA-Z]{1,2}\b", " ", text)  # single/double char words
    text = re.sub(r"\b[0-9]+\b",        " ", text)  # standalone numbers
    
    # ── Step 6: Final whitespace cleanup ──────────────────────────────────────
    text = re.sub(r"[\t\r\n]+", " ", text)           # tabs and newlines
    text = re.sub(r" {2,}",     " ", text)           # multiple spaces
    text = text.strip()
    
    return text

# ── Apply to df_actual_body ───────────────────────────────────────────────────
df_actual_body["actual_body"] = df_actual_body["actual_body"].apply(clean_actual_body)

# ── Quick check ───────────────────────────────────────────────────────────────
print(f"✅ Cleaned actual_body")
print(f"   Empty bodies  : {(df_actual_body['actual_body'] == '').sum()}")
print(f"   Filled bodies : {(df_actual_body['actual_body'] != '').sum()}")

# ── Sample check ──────────────────────────────────────────────────────────────
print("\n── Sample Cleaned Body ──────────────────────────────")
print(df_actual_body["actual_body"].iloc[0][:300])
print("─────────────────────────────────────────────────────")




CASE_NUMBER_PATTERN = r"[A-Za-z]{5}\d{2}-\d{4,5}"

def extract_pure_body(text):
    if not text or pd.isna(text):
        return ""
    
    # ── Step 1: Remove CAUTION banner ─────────────────────────────────────────
    text = re.sub(
        r"CAUTION.*?safe\.?",
        " ", text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # ── Step 2: Cut at signature indicators ───────────────────────────────────
    signature_patterns = [
        r"(regards|best regards|warm regards|kind regards)",
        r"(thanks and regards|thank you and regards)",
        r"(sincerely|yours sincerely|yours faithfully)",
        r"(thanks|thank you)\s*,?\s*\n",
        r"(cheers|cordially|respectfully)",
    ]
    for pattern in signature_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
            break

    # ── Step 3: Remove From/To/CC/Sent header lines ───────────────────────────
    text = re.sub(r"from\s*:.*",          "", text, flags=re.IGNORECASE)
    text = re.sub(r"to\s*:.*",            "", text, flags=re.IGNORECASE)
    text = re.sub(r"cc\s*:.*",            "", text, flags=re.IGNORECASE)
    text = re.sub(r"sent\s*:.*",          "", text, flags=re.IGNORECASE)
    text = re.sub(r"subject\s*:.*",       "", text, flags=re.IGNORECASE)
    text = re.sub(r"date\s*:.*",          "", text, flags=re.IGNORECASE)

    # ── Step 4: Remove signature lines ────────────────────────────────────────
    text = re.sub(r"phone\s*:.*",         "", text, flags=re.IGNORECASE)
    text = re.sub(r"tel\s*:.*",           "", text, flags=re.IGNORECASE)
    text = re.sub(r"mob\s*:.*",           "", text, flags=re.IGNORECASE)
    text = re.sub(r"fax\s*:.*",           "", text, flags=re.IGNORECASE)
    text = re.sub(r"email\s*:.*",         "", text, flags=re.IGNORECASE)
    text = re.sub(r"website\s*:.*",       "", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.\S+",            "", text, flags=re.IGNORECASE)

    # ── Step 5: Remove phone numbers ──────────────────────────────────────────
    text = re.sub(r"\+?[\d\s\-\(\)]{7,}", " ", text)

    # ── Step 6: Remove pipe separated signature lines ─────────────────────────
    text = re.sub(r"[^.!?]*\|[^.!?]*",   " ", text)

    # ── Step 7: Remove disclaimer blocks ──────────────────────────────────────
    disclaimer_patterns = [
        r"this email.*?confidential.*",
        r"this message.*?intended.*",
        r"disclaimer.*",
        r"caution.*?external email.*",
    ]
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # ── Step 8: Final cleanup ─────────────────────────────────────────────────
    text = re.sub(r"\b[a-zA-Z]{1,2}\b",  " ", text)
    text = re.sub(r" {2,}",               " ", text)
    text = re.sub(r"\n{2,}",             "\n", text)
    text = text.strip()

    return text


# ── Apply pure body ───────────────────────────────────────────────────────────
df_actual_body["pure_body"] = df_actual_body["actual_body"].apply(extract_pure_body)

# ── Extract case number from subject only ─────────────────────────────────────
df_actual_body["case_number"] = df_actual_body["subject"].apply(
    lambda x: re.search(CASE_NUMBER_PATTERN, x).group() 
              if pd.notna(x) and re.search(CASE_NUMBER_PATTERN, x) 
              else None
)

# ── Stats ─────────────────────────────────────────────────────────────────────
print(f"✅ Pure body filled    : {(df_actual_body['pure_body'] != '').sum()}")
print(f"✅ With case number    : {df_actual_body['case_number'].notna().sum()}")
print(f"⚠️  Without case number: {df_actual_body['case_number'].isna().sum()}")

df_actual_body[["subject", "case_number", "pure_body"]].head(5)
# ── Extract case number from subject ──────────────────────────────────────────
CASE_NUMBER_PATTERN = r"[A-Za-z]{5}\d{2}-\d{4,5}"

df_actual_body["case_number"] = df_actual_body["subject"].apply(
    lambda x: re.search(CASE_NUMBER_PATTERN, x).group() if pd.notna(x) and re.search(CASE_NUMBER_PATTERN, x) else None
)

# ── Verify ────────────────────────────────────────────────────────────────────
has_case    = df_actual_body["case_number"].notna().sum()
no_case     = df_actual_body["case_number"].isna().sum()

print(f"✅ Emails with case number    : {has_case}")
print(f"⚠️  Emails without case number: {no_case}")

# ── Sample check ──────────────────────────────────────────────────────────────
print("\n── Sample subject vs extracted case number ──────────")
df_actual_body[df_actual_body["case_number"].notna()][["subject", "case_number"]].head(10)



#Rebuild Class DataFrames from Clean Data

# ── Use only classified emails ─────────────────────────────────────────────
df_classified = df_actual_body[df_actual_body["comment"].notna()].copy()

print(f"Total classified emails : {len(df_classified)}")
print(f"\nClass distribution:")
print(df_classified["comment"].value_counts())

# ── Separate by class ──────────────────────────────────────────────────────
df_dsd      = df_classified[df_classified["comment"] == "DSD   Acknowledgement"].copy()
df_followup = df_classified[df_classified["comment"] == "For Follow up"].copy()
df_argus    = df_classified[df_classified["comment"] == "Argus ID"].copy()

print(f"\nDSD          : {len(df_dsd)}")
print(f"Follow Up    : {len(df_followup)}")
print(f"Argus ID     : {len(df_argus)}")


#Build Word Baskets Using Both Subject + Pure Body
from collections import Counter
import re

stop_words = {
    "the","is","in","it","of","and","to","a","an","that","this",
    "for","on","are","was","with","as","at","be","by","from",
    "have","has","had","not","but","or","you","we","i","re",
    "your","our","please","thank","thanks","dear","hi","hello",
    "regards","mail","email","will","would","could","should",
    "just","also","get","can","one","all","any","been","when",
    "they","them","their","there","here","which","more","than",
    "per","yes","no","ok","sure","noted"
}

def get_word_counts(df_class, col="pure_body"):
    all_text = " ".join(
        (df_class["subject"].fillna("") + " " + df_class[col].fillna("")).tolist()
    ).lower()
    words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
    words = [w for w in words if w not in stop_words]
    return Counter(words)



# ── Build counters ─────────────────────────────────────────────────────────────
counter_dsd      = get_word_counts(df_dsd)
counter_followup = get_word_counts(df_followup)
counter_argus    = get_word_counts(df_argus)

# ── Build basket — no common word removal, just threshold ─────────────────────
def build_word_basket(counter, df_class, threshold_pct=0.1, top_n=30):
    """
    threshold_pct=0.1 → word must appear in at least 10% of class emails
    No elimination of cross-class words — let overlap show naturally
    """
    min_freq = len(df_class) * threshold_pct
    
    filtered = {
        word: count
        for word, count in counter.items()
        if count >= min_freq
    }
    return sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]

basket_dsd      = build_word_basket(counter_dsd,      df_dsd,      threshold_pct=0.1)
basket_followup = build_word_basket(counter_followup, df_followup, threshold_pct=0.1)
basket_argus    = build_word_basket(counter_argus,    df_argus,    threshold_pct=0.1)

# ── Get word sets for overlap detection ───────────────────────────────────────
words_dsd      = set(w for w, c in basket_dsd)
words_followup = set(w for w, c in basket_followup)
words_argus    = set(w for w, c in basket_argus)

def get_overlap(word):
    classes = []
    if word in words_dsd:      classes.append("DSD")
    if word in words_followup: classes.append("Followup")
    if word in words_argus:    classes.append("Argus")
    return ", ".join(classes) if len(classes) > 1 else "unique"

# ── Print baskets with overlap info ───────────────────────────────────────────
print(f"\n── DSD Acknowledgement Basket ({len(df_dsd)} emails) ────────────")
print(f"   {'Word':<25} {'Count':>6}  {'%':>6}  {'Overlap'}")
print(f"   {'─'*60}")
for word, count in basket_dsd:
    overlap = get_overlap(word)
    flag    = "⚠️" if overlap != "unique" else "✅"
    print(f"   {word:<25} {count:>6}  {round(count/len(df_dsd)*100, 1):>5}%  {flag} {overlap}")

print(f"\n── For Follow Up Basket ({len(df_followup)} emails) ──────────────")
print(f"   {'Word':<25} {'Count':>6}  {'%':>6}  {'Overlap'}")
print(f"   {'─'*60}")
for word, count in basket_followup:
    overlap = get_overlap(word)
    flag    = "⚠️" if overlap != "unique" else "✅"
    print(f"   {word:<25} {count:>6}  {round(count/len(df_followup)*100, 1):>5}%  {flag} {overlap}")

print(f"\n── Argus ID Basket ({len(df_argus)} emails) ──────────────────────")
print(f"   {'Word':<25} {'Count':>6}  {'%':>6}  {'Overlap'}")
print(f"   {'─'*60}")
for word, count in basket_argus:
    overlap = get_overlap(word)
    flag    = "⚠️" if overlap != "unique" else "✅"
    print(f"   {word:<25} {count:>6}  {round(count/len(df_argus)*100, 1):>5}%  {flag} {overlap}")



##this line is to fetch the data from jan 

def fetch_all_emails(start_date=None, end_date=None):
    
    # ── Build filter based on dates provided ──────────────────────────────────
    if start_date and end_date:
        start     = f"{start_date}T00:00:00Z"
        end       = f"{end_date}T23:59:59Z"
        date_filter = f"&$filter=receivedDateTime ge {start} and receivedDateTime le {end}"
    elif start_date:
        start       = f"{start_date}T00:00:00Z"
        date_filter = f"&$filter=receivedDateTime ge {start}"
    elif end_date:
        end         = f"{end_date}T23:59:59Z"
        date_filter = f"&$filter=receivedDateTime le {end}"
    else:
        date_filter = ""   # no filter — fetch all emails

    url = (
        f"https://graph.microsoft.com/v1.0/users/{USER_EMAIL}/messages"
        f"?$select=id,subject,body,bodyPreview,from,toRecipients,receivedDateTime,"
        f"hasAttachments,conversationId"
        f"&$top=1000"
        f"&$orderby=receivedDateTime desc"
        f"{date_filter}"
    )
    
    emails  = []
    t_start = time.time()

    while url:
        resp = requests.get(url, headers=HEADERS, verify=False).json()
        if "error" in resp:
            print("❌ API Error:", resp["error"]["message"])
            break
        batch = resp.get("value", [])
        emails.extend(batch)
        url = resp.get("@odata.nextLink")

    elapsed = round(time.time() - t_start, 1)
    print(f"✅ Fetched {len(emails)} emails in {elapsed}s")
    
    df = pd.DataFrame(emails)
    if df.empty:
        print("⚠️ No emails found")
        return df

    df["sender_name"]   = df["from"].apply(lambda x: x["emailAddress"]["name"])
    df["sender_email"]  = df["from"].apply(lambda x: x["emailAddress"]["address"])
    df["body_full"]     = df["body"].apply(extract_actual_body)
    df["to_recipients"] = df["toRecipients"].apply(
                            lambda x: ", ".join([r["emailAddress"]["address"] for r in x])
                          )

    cols_to_drop = ["@odata.etag", "@odata.type", "from", "body", "toRecipients"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    df = df.sort_values(["conversationId", "receivedDateTime"]).reset_index(drop=True)

    df["reply_count"]          = df.groupby("conversationId")["id"].transform("count")
    df["reply_position"]       = df.groupby("conversationId")["receivedDateTime"].rank(method="first").astype(int)
    df["is_thread"]            = df["reply_count"] > 1
    df["is_original_email"]    = df["reply_position"] == 1
    df["thread_started_at"]    = df.groupby("conversationId")["receivedDateTime"].transform("min")
    df["thread_last_reply_at"] = df.groupby("conversationId")["receivedDateTime"].transform("max")
    df["all_participants"]     = df.groupby("conversationId")["sender_email"].transform(
                                    lambda x: ", ".join(x.unique())
                                 )

    desired_cols = [
        "conversationId", "is_thread", "reply_count", "reply_position",
        "is_original_email", "thread_started_at", "thread_last_reply_at",
        "all_participants", "id", "receivedDateTime", "sender_name",
        "sender_email", "to_recipients", "subject", "bodyPreview",
        "body_full", "hasAttachments"
    ]
    df = df[[col for col in desired_cols if col in df.columns]]

    print(f"✅ Shape: {df.shape}")
    return df

# ── Specific date range ────────────────────────────────────────────────────────
df_all = fetch_all_emails(start_date="2026-01-01", end_date="2026-04-20")

# ── Only start date ───────────────────────────────────────────────────────────
df_all = fetch_all_emails(start_date="2026-01-01")

# ── Only end date ─────────────────────────────────────────────────────────────
df_all = fetch_all_emails(end_date="2026-04-20")

# ── No filter — fetch everything ──
────────────────────────────────────────────
df_all = fetch_all_emails()


df_all = fetch_all_emails(start_date="2026-01-01", end_date="2026-04-20")
print(f"✅ Total emails : {len(df_all)}")


import re
import html

def extract_actual_body(body_dict):
    if not body_dict:
        return ""
    
    content = body_dict.get("content", "")
    
    # ── Step 1: Remove HTML comments ──────────────────────────────────────────
    content = re.sub(r"<!--.*?-->", " ", content, flags=re.DOTALL)
    
    # ── Step 2: Remove style and script blocks ────────────────────────────────
    content = re.sub(r"<style.*?>.*?</style>", " ", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<script.*?>.*?</script>", " ", content, flags=re.DOTALL | re.IGNORECASE)
    
    # ── Step 3: Replace block tags with newlines ──────────────────────────────
    content = re.sub(r"<br\s*/?>", "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</p>",      "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</div>",    "\n", content, flags=re.IGNORECASE)
    
    # ── Step 4: Strip all remaining HTML tags ─────────────────────────────────
    content = re.sub(r"<[^>]+>", "", content)
    
    # ── Step 5: Decode HTML entities ──────────────────────────────────────────
    content = html.unescape(content)
    content = re.sub(r"&[a-zA-Z]+;", " ", content)
    content = re.sub(r"&#\d+;",      " ", content)
    
    # ── Step 6: Cut at thread dividers ────────────────────────────────────────
    thread_dividers = [
        r"From\s*:\s*.+?Sent\s*:\s*.+?To\s*:",
        r"On\s+.+?wrote\s*:",
        r"-{3,}.*?Original Message.*?-{3,}",
        r"_{3,}",
        r"-{5,}",
        r"Sent from my (iPhone|iPad|Outlook|Mail)",
        r"Get Outlook for (iOS|Android)",
        r"CAUTION\s*:",
        r"DISCLAIMER\s*:",
        r"This email and any attachments",
        r"This message contains confidential",
    ]
    for pattern in thread_dividers:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            content = content[:match.start()].strip()
            break

    # ── Step 7: Clean garbage characters ──────────────────────────────────────
    content = re.sub(r"[^\x00-\x7F]+", " ", content)
    content = re.sub(r"[\t\r]+",        " ", content)
    content = re.sub(r"\n{3,}",        "\n\n", content)
    content = re.sub(r" {2,}",          " ", content)
    content = re.sub(r"[;:]{2,}",       "",  content)
    content = content.strip()
    
    return content if content else ""


# ── Now call fetch ─────────────────────────────────────────────────────────────
df_all = fetch_all_emails(start_date="2026-01-01", end_date="2026-04-20")
print(f"✅ Total emails : {len(df_all)}")



#Dynamic Word Basket Scoring Engine

from collections import Counter
import re

stop_words = {
    "the","is","in","it","of","and","to","a","an","that","this",
    "for","on","are","was","with","as","at","be","by","from",
    "have","has","had","not","but","or","you","we","i","re",
    "your","our","please","thank","thanks","dear","hi","hello",
    "regards","mail","email","will","would","could","should",
    "just","also","get","can","one","all","any","been","when",
    "they","them","their","there","here","which","more","than",
    "per","yes","no","ok","sure","noted","use","used","using"
}

# ── Step 1: Build classified dataset ──────────────────────────────────────────
df_classified = df[df["comment"].notna()].copy()
print(f"Total classified emails : {len(df_classified)}")
print(df_classified["comment"].value_counts())

# ── Step 2: Separate classes ───────────────────────────────────────────────────
df_dsd      = df_classified[df_classified["comment"].str.contains("DSD",    case=False, na=False)]
df_followup = df_classified[df_classified["comment"].str.contains("Follow", case=False, na=False)]
df_argus    = df_classified[df_classified["comment"].str.contains("Argus",  case=False, na=False)]

# ── Step 3: Word counter per class ────────────────────────────────────────────
def get_word_counts(df_class, col="body_clean"):
    all_text = " ".join(
        (df_class["subject"].fillna("") + " " + df_class[col].fillna("")).tolist()
    ).lower()
    words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
    words = [w for w in words if w not in stop_words]
    return Counter(words)

counter_dsd      = get_word_counts(df_dsd)
counter_followup = get_word_counts(df_followup)
counter_argus    = get_word_counts(df_argus)

print(f"\n✅ Counters built")
print(f"   DSD unique words      : {len(counter_dsd)}")
print(f"   Follow Up unique words : {len(counter_followup)}")
print(f"   Argus unique words     : {len(counter_argus)}")

# ── Step 4: Build weighted basket ─────────────────────────────────────────────
def build_weighted_basket(counters, class_sizes, threshold_pct=0.05):
    """
    counters    : dict of {class_name: Counter}
    class_sizes : dict of {class_name: int}
    threshold   : min frequency % to include word
    
    Weight logic:
    - Word in 1 class only  → weight = 1.0  (unique)
    - Word in 2 classes     → weight = 0.5  (shared)
    - Word in 3 classes     → weight = 0.2  (common)
    - Multiplied by frequency % for final weight
    """
    
    all_classes = list(counters.keys())
    basket      = {}   # {class: {word: weight}}
    
    # Collect all unique words across all classes
    all_words = set()
    for counter in counters.values():
        all_words.update(counter.keys())
    
    for word in all_words:
        
        # How many classes contain this word above threshold
        classes_with_word = []
        for class_name, counter in counters.items():
            freq_pct = counter[word] / class_sizes[class_name]
            if freq_pct >= threshold_pct:
                classes_with_word.append((class_name, freq_pct))
        
        if not classes_with_word:
            continue
        
        # Overlap penalty
        overlap_count = len(classes_with_word)
        if overlap_count == 1:
            overlap_weight = 1.0    # unique   → full weight
        elif overlap_count == 2:
            overlap_weight = 0.5    # shared   → half weight
        else:
            overlap_weight = 0.2    # common   → low weight
        
        # Assign weighted score to each class
        for class_name, freq_pct in classes_with_word:
            if class_name not in basket:
                basket[class_name] = {}
            
            # Final weight = overlap weight × frequency %
            final_weight = round(overlap_weight * freq_pct, 4)
            basket[class_name][word] = final_weight
    
    return basket

# ── Define classes and sizes ───────────────────────────────────────────────────
counters = {
    "DSD Acknowledgement" : counter_dsd,
    "For Follow Up"       : counter_followup,
    "Argus ID"            : counter_argus
}

class_sizes = {
    "DSD Acknowledgement" : len(df_dsd),
    "For Follow Up"       : len(df_followup),
    "Argus ID"            : len(df_argus)
}

weighted_basket = build_weighted_basket(counters, class_sizes, threshold_pct=0.05)

# ── Print basket summary ───────────────────────────────────────────────────────
for class_name, words in weighted_basket.items():
    print(f"\n── {class_name} ({'─'*30})")
    print(f"   Total words in basket : {len(words)}")
    
    # Show top 20 by weight
    top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"   {'Word':<25} {'Weight':>8}")
    print(f"   {'─'*35}")
    for word, weight in top_words:
        unique = "✅" if weight >= 0.5 else "⚠️"
        print(f"   {word:<25} {weight:>8.4f}  {unique}")


#Step 5 — Scoring Engine

def score_email(text, subject, weighted_basket):
    """
    Score an email against each class basket
    Returns predicted class, scores and confidence
    """
    if not text:
        text = ""
    
    combined = f"{subject} {text}".lower()
    words    = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined))
    words    = {w for w in words if w not in stop_words}
    
    scores = {}
    matched_words = {}
    
    for class_name, basket in weighted_basket.items():
        score = 0
        hits  = []
        
        for word in words:
            if word in basket:
                score += basket[word]
                hits.append((word, basket[word]))
        
        scores[class_name]        = round(score, 4)
        matched_words[class_name] = sorted(hits, key=lambda x: x[1], reverse=True)[:5]
    
    # ── Determine predicted class ──────────────────────────────────────────────
    if all(s == 0 for s in scores.values()):
        return "Unclassified", 0.0, scores, {}
    
    sorted_scores  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_class      = sorted_scores[0][0]
    top_score      = sorted_scores[0][1]
    second_score   = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    
    # Confidence = gap between top and second class
    total          = sum(scores.values())
    confidence     = round(top_score / total, 4) if total > 0 else 0
    
    return top_class, confidence, scores, matched_words[top_class]


# ── Step 6: Apply to unmatched emails ─────────────────────────────────────────
def classify_row(row):
    predicted, confidence, scores, keywords = score_email(
        row["body_clean"], row["subject"], weighted_basket
    )
    return pd.Series({
        "predicted_class" : predicted,
        "confidence"      : confidence,
        "score_dsd"       : scores.get("DSD Acknowledgement", 0),
        "score_followup"  : scores.get("For Follow Up", 0),
        "score_argus"     : scores.get("Argus ID", 0),
        "matched_keywords": str([w for w, s in keywords])
    })

df_unmatched[["predicted_class", "confidence",
              "score_dsd", "score_followup", 
              "score_argus", "matched_keywords"]] = df_unmatched.apply(classify_row, axis=1)

# ── Step 7: Results ───────────────────────────────────────────────────────────
print(f"\n✅ Classification Results")
print(f"{'─'*40}")
print(df_unmatched["predicted_class"].value_counts())

print(f"\n── Confidence Distribution ──────────────────")
print(f"High   (>0.7) : {(df_unmatched['confidence'] > 0.7).sum()}")
print(f"Medium (0.4-0.7): {((df_unmatched['confidence'] >= 0.4) & (df_unmatched['confidence'] <= 0.7)).sum()}")
print(f"Low    (<0.4) : {(df_unmatched['confidence'] < 0.4).sum()}")

df_unmatched.to_excel("df_classified_scored.xlsx", index=False)
print(f"\n✅ Saved to df_classified_scored.xlsx")


#classification fucntion based on rule 

# ── Word baskets based on your analysis ───────────────────────────────────────

# Argus ID — single word trigger
ARGUS_TRIGGER = ["argus"]

# DSD Acknowledgement — single word trigger  
DSD_TRIGGER = ["acknowledge", "acknowledged", "acknowledgement", "acknowledgment"]

# Follow Up — needs multiple unique word matches
FOLLOWUP_UNIQUE_WORDS = [
    "investigation", "batch", "sample", "kindly", "team",
    "observed", "provide", "patient", "information", "discrepancy",
    "found", "were"
]

FOLLOWUP_MIN_MATCHES = 2   # at least 2 unique words must match

# ── Overlap words — used as tiebreaker only, not primary signal ───────────────
OVERLAP_WORDS = ["colleague", "below", "find", "case", "greetings", "receipt"]


def classify_email(row, weighted_basket):
    
    subject   = str(row["subject"]).lower()   if pd.notna(row["subject"])    else ""
    body      = str(row["body_clean"]).lower() if pd.notna(row["body_clean"]) else ""
    combined  = f"{subject} {body}"
    words     = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined))

    # ── Rule 1: Argus ID — highest priority ───────────────────────────────────
    if any(trigger in words for trigger in ARGUS_TRIGGER):
        return pd.Series({
            "predicted_class" : "Argus ID",
            "confidence"      : 0.97,
            "rule_triggered"  : "argus_trigger",
            "matched_keywords": str([w for w in ARGUS_TRIGGER if w in words])
        })

    # ── Rule 2: DSD Acknowledgement ───────────────────────────────────────────
    if any(trigger in words for trigger in DSD_TRIGGER):
        return pd.Series({
            "predicted_class" : "DSD Acknowledgement",
            "confidence"      : 0.97,
            "rule_triggered"  : "dsd_trigger",
            "matched_keywords": str([w for w in DSD_TRIGGER if w in words])
        })

    # ── Rule 3: Follow Up — needs multiple unique word matches ────────────────
    followup_hits = [w for w in FOLLOWUP_UNIQUE_WORDS if w in words]
    
    if len(followup_hits) >= FOLLOWUP_MIN_MATCHES:
        
        # Confidence scales with number of matches
        # 2 matches = 0.60, 3 = 0.70, 4 = 0.80, 5+ = 0.90+
        confidence = min(0.50 + (len(followup_hits) * 0.10), 0.99)
        
        return pd.Series({
            "predicted_class" : "For Follow Up",
            "confidence"      : round(confidence, 2),
            "rule_triggered"  : f"followup_{len(followup_hits)}_words_matched",
            "matched_keywords": str(followup_hits)
        })

    # ── Rule 4: Weak Follow Up signal — 1 unique word + overlap words ─────────
    overlap_hits  = [w for w in OVERLAP_WORDS if w in words]
    
    if len(followup_hits) == 1 and len(overlap_hits) >= 2:
        return pd.Series({
            "predicted_class" : "For Follow Up",
            "confidence"      : 0.45,
            "rule_triggered"  : "followup_weak_signal",
            "matched_keywords": str(followup_hits + overlap_hits)
        })

    # ── Rule 5: Score based fallback using weighted basket ────────────────────
    scores        = {}
    matched_words = {}
    
    for class_name, basket in weighted_basket.items():
        score = sum(basket.get(w, 0) for w in words)
        hits  = [w for w in words if w in basket]
        scores[class_name]        = round(score, 4)
        matched_words[class_name] = hits

    if all(s == 0 for s in scores.values()):
        return pd.Series({
            "predicted_class" : "Unclassified",
            "confidence"      : 0.0,
            "rule_triggered"  : "no_match",
            "matched_keywords": "[]"
        })

    top_class  = max(scores, key=scores.get)
    top_score  = scores[top_class]
    total      = sum(scores.values())
    confidence = round(top_score / total, 2) if total > 0 else 0

    return pd.Series({
        "predicted_class" : top_class if confidence >= 0.4 else "Unclassified",
        "confidence"      : confidence,
        "rule_triggered"  : "basket_score_fallback",
        "matched_keywords": str(matched_words[top_class])
    })


# ── Apply to unmatched emails ─────────────────────────────────────────────────
df_unmatched[["predicted_class", "confidence",
              "rule_triggered", "matched_keywords"]] = df_unmatched.apply(
    lambda row: classify_email(row, weighted_basket), axis=1
)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"✅ Classification Results")
print(f"{'─'*40}")
print(df_unmatched["predicted_class"].value_counts())

print(f"\n── Confidence Distribution ──────────────────────")
print(f"High   (> 0.7) : {(df_unmatched['confidence'] >  0.7).sum()}")
print(f"Medium (0.4-0.7): {((df_unmatched['confidence'] >= 0.4) & (df_unmatched['confidence'] <= 0.7)).sum()}")
print(f"Low    (< 0.4) : {(df_unmatched['confidence'] <  0.4).sum()}")

print(f"\n── Rule Triggered Breakdown ─────────────────────")
print(df_unmatched["rule_triggered"].value_counts())

df_unmatched.to_excel("df_classified_scored.xlsx", index=False)
print(f"\n✅ Saved to df_classified_scored.xlsx")


##3 for accuracy score
# ── Run classifier on already labeled emails ──────────────────────────────────
df_labeled = df[df["comment"].notna()].copy()

print(f"Total labeled emails : {len(df_labeled)}")
print(f"Class distribution   :")
print(df_labeled["comment"].value_counts())

# ── Apply classification ───────────────────────────────────────────────────────
df_labeled[["predicted_class", "confidence",
            "rule_triggered", "matched_keywords"]] = df_labeled.apply(
    lambda row: classify_email(row, weighted_basket), axis=1
)

# ── Normalize actual comment to match predicted class names ───────────────────
def normalize_comment(comment):
    comment = str(comment).strip().lower()
    if "dsd" in comment:
        return "DSD Acknowledgement"
    elif "follow" in comment:
        return "For Follow Up"
    elif "argus" in comment:
        return "Argus ID"
    else:
        return comment

df_labeled["actual_class"] = df_labeled["comment"].apply(normalize_comment)

# ── Overall Accuracy ──────────────────────────────────────────────────────────
correct   = (df_labeled["predicted_class"] == df_labeled["actual_class"]).sum()
total     = len(df_labeled)
accuracy  = round(correct / total * 100, 2)

print(f"\n✅ Overall Accuracy : {correct}/{total}  ({accuracy}%)")

# ── Per Class Accuracy ────────────────────────────────────────────────────────
print(f"\n── Per Class Accuracy ────────────────────────────────────")
print(f"{'Class':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Wrong':>8}")
print(f"{'─'*65}")

for class_name in df_labeled["actual_class"].unique():
    class_df   = df_labeled[df_labeled["actual_class"] == class_name]
    class_correct = (class_df["predicted_class"] == class_name).sum()
    class_total   = len(class_df)
    class_acc     = round(class_correct / class_total * 100, 2)
    class_wrong   = class_total - class_correct
    flag          = "✅" if class_acc >= 80 else "⚠️" if class_acc >= 60 else "❌"
    print(f"{class_name:<25} {class_correct:>8} {class_total:>8} {class_acc:>9}%  {class_wrong:>6}  {flag}")

# ── Misclassified Breakdown ───────────────────────────────────────────────────
print(f"\n── Misclassification Breakdown ───────────────────────────")
df_wrong = df_labeled[df_labeled["predicted_class"] != df_labeled["actual_class"]]

print(f"Total wrong : {len(df_wrong)}")
print(f"\nActual → Predicted breakdown:")
print(df_wrong.groupby(["actual_class", "predicted_class"]).size().reset_index(name="count").to_string(index=False))

# ── Rule Triggered Breakdown per class ────────────────────────────────────────
print(f"\n── Rule Triggered per Class ──────────────────────────────")
print(df_labeled.groupby(["actual_class", "rule_triggered"]).size().reset_index(name="count").to_string(index=False))

# ── Unclassified Breakdown ────────────────────────────────────────────────────
df_unclassified = df_labeled[df_labeled["predicted_class"] == "Unclassified"]
print(f"\n── Unclassified by Actual Class ──────────────────────────")
print(df_unclassified["actual_class"].value_counts())

# ── Save for inspection ───────────────────────────────────────────────────────
df_labeled.to_excel("accuracy_check.xlsx", index=False)
print(f"\n✅ Saved to accuracy_check.xlsx")


#accuracy for 3 class
# ── Filter only the 3 classes we built rules for ─────────────────────────────
df_3class = df[df["comment"].notna()].copy()

df_3class = df_3class[
    df_3class["comment"].str.contains("DSD",    case=False, na=False) |
    df_3class["comment"].str.contains("Follow", case=False, na=False) |
    df_3class["comment"].str.contains("Argus",  case=False, na=False)
].copy()

print(f"Total 3 class emails : {len(df_3class)}")
print(df_3class["comment"].value_counts())

# ── Normalize actual comment ───────────────────────────────────────────────────
def normalize_comment(comment):
    comment = str(comment).strip().lower()
    if "dsd" in comment:
        return "DSD Acknowledgement"
    elif "follow" in comment:
        return "For Follow Up"
    elif "argus" in comment:
        return "Argus ID"

df_3class["actual_class"] = df_3class["comment"].apply(normalize_comment)

# ── Apply classifier ───────────────────────────────────────────────────────────
df_3class[["predicted_class", "confidence",
           "rule_triggered",  "matched_keywords"]] = df_3class.apply(
    lambda row: classify_email(row, weighted_basket), axis=1
)

# ── Overall Accuracy within 3 classes only ────────────────────────────────────
correct  = (df_3class["predicted_class"] == df_3class["actual_class"]).sum()
total    = len(df_3class)
accuracy = round(correct / total * 100, 2)

print(f"\n✅ Overall Accuracy (3 classes only) : {correct}/{total}  ({accuracy}%)")

# ── Per Class Accuracy ─────────────────────────────────────────────────────────
print(f"\n── Per Class Accuracy ────────────────────────────────────────")
print(f"{'Class':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Wrong':>8}")
print(f"{'─'*65}")

for class_name in ["DSD Acknowledgement", "For Follow Up", "Argus ID"]:
    class_df      = df_3class[df_3class["actual_class"] == class_name]
    class_correct = (class_df["predicted_class"] == class_name).sum()
    class_total   = len(class_df)
    class_acc     = round(class_correct / class_total * 100, 2) if class_total > 0 else 0
    class_wrong   = class_total - class_correct
    flag          = "✅" if class_acc >= 90 else "⚠️" if class_acc >= 70 else "❌"
    print(f"{class_name:<25} {class_correct:>8} {class_total:>8} {class_acc:>9}%  {class_wrong:>6}  {flag}")

# ── Misclassification Breakdown ───────────────────────────────────────────────
print(f"\n── Misclassification Breakdown ───────────────────────────────")
df_wrong = df_3class[df_3class["predicted_class"] != df_3class["actual_class"]]
print(f"Total wrong : {len(df_wrong)}")
print(f"\nActual → Predicted:")
print(df_wrong.groupby(["actual_class","predicted_class"]).size().reset_index(name="count").to_string(index=False))

# ── Rule Triggered Breakdown ──────────────────────────────────────────────────
print(f"\n── Rule Triggered per Class ──────────────────────────────────")
print(df_3class.groupby(["actual_class","rule_triggered"]).size().reset_index(name="count").to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
df_3class.to_excel("accuracy_3class.xlsx", index=False)
print(f"\n✅ Saved to accuracy_3class.xlsx")


## new classification rule added in classify email 5 class 
def classify_email(row):

    subject  = str(row["subject"]).lower()   if pd.notna(row["subject"])   else ""
    body     = str(row["pure_body"]).lower() if pd.notna(row["pure_body"]) else ""
    combined = f"{subject} {body}"
    words    = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined))

    # ── Pre-compute all hits ───────────────────────────────────────────────────
    argus_hits    = [w for w in ARGUS_TRIGGER         if w in words]
    ppm_hits      = [w for w in PPM_TRIGGER           if w in words]
    dsd_hits      = [w for w in DSD_TRIGGER           if w in words]
    followup_hits = [w for w in FOLLOWUP_UNIQUE_WORDS if w in words]
    overlap_hits  = [w for w in OVERLAP_WORDS         if w in words]

    # ── Rule 1: Argus ID ──────────────────────────────────────────────────────
    if argus_hits:
        return pd.Series({
            "predicted_class" : "Argus ID",
            "confidence"      : 0.97,
            "rule_triggered"  : "argus_trigger",
            "matched_keywords": str(argus_hits)
        })

    # ── Rule 2: CQA Acknowledgement ───────────────────────────────────────────
    # Must have acknowledge + receipt + compliant all present
    has_acknowledge = any(w in words for w in DSD_TRIGGER)
    has_required    = all(w in words for w in CQA_REQUIRED_WORDS)

    if has_acknowledge and has_required:
        return pd.Series({
            "predicted_class" : "CQA Acknowledgement",
            "confidence"      : 0.97,
            "rule_triggered"  : "cqa_trigger",
            "matched_keywords": str([w for w in CQA_TRIGGER if w in words])
        })

    # ── Rule 3: PPM Request ───────────────────────────────────────────────────
    if len(ppm_hits) >= PPM_MIN_MATCHES:
        confidence = min(0.50 + (len(ppm_hits) * 0.10), 0.99)
        return pd.Series({
            "predicted_class" : "PPM Request",
            "confidence"      : round(confidence, 2),
            "rule_triggered"  : f"ppm_{len(ppm_hits)}_words_matched",
            "matched_keywords": str(ppm_hits)
        })

    # ── Rule 4: DSD Acknowledgement ───────────────────────────────────────────
    if dsd_hits:
        return pd.Series({
            "predicted_class" : "DSD Acknowledgement",
            "confidence"      : 0.97,
            "rule_triggered"  : "dsd_trigger",
            "matched_keywords": str(dsd_hits)
        })

    # ── Rule 5: For Follow Up ─────────────────────────────────────────────────
    if len(followup_hits) >= FOLLOWUP_MIN_MATCHES:
        confidence = min(0.50 + (len(followup_hits) * 0.10), 0.99)
        return pd.Series({
            "predicted_class" : "For Follow Up",
            "confidence"      : round(confidence, 2),
            "rule_triggered"  : f"followup_{len(followup_hits)}_words_matched",
            "matched_keywords": str(followup_hits)
        })

    # ── Rule 6: Weak Follow Up ────────────────────────────────────────────────
    if len(followup_hits) == 1 and len(overlap_hits) >= 2:
        return pd.Series({
            "predicted_class" : "For Follow Up",
            "confidence"      : 0.45,
            "rule_triggered"  : "followup_weak_signal",
            "matched_keywords": str(followup_hits + overlap_hits)
        })

    # ── Rule 7: Unclassified ──────────────────────────────────────────────────
    return pd.Series({
        "predicted_class" : "Unclassified",
        "confidence"      : 0.0,
        "rule_triggered"  : "no_match",
        "matched_keywords": "[]"
    })

print("✅ classify_email with 5 classes defined")

## update  of 5 accuracy 
# ── Filter to 5 classes only ──────────────────────────────────────────────────
df_5class = df[df["comment"].notna()].copy()

df_5class = df_5class[
    df_5class["comment"].str.contains("DSD",     case=False, na=False) |
    df_5class["comment"].str.contains("Follow",  case=False, na=False) |
    df_5class["comment"].str.contains("Argus",   case=False, na=False) |
    df_5class["comment"].str.contains("PPM",     case=False, na=False) |
    df_5class["comment"].str.contains("CQA",     case=False, na=False)
].copy()

print(f"Total 5 class emails : {len(df_5class)}")
print(df_5class["comment"].value_counts())

# ── Normalize actual comment ───────────────────────────────────────────────────
def normalize_comment(comment):
    comment = str(comment).strip().lower()
    if "cqa"    in comment: return "CQA Acknowledgement"
    elif "dsd"    in comment: return "DSD Acknowledgement"
    elif "follow" in comment: return "For Follow Up"
    elif "argus"  in comment: return "Argus ID"
    elif "ppm"    in comment: return "PPM Request"

df_5class["actual_class"] = df_5class["comment"].apply(normalize_comment)

# ── Apply classifier ───────────────────────────────────────────────────────────
df_5class[["predicted_class", "confidence",
           "rule_triggered",  "matched_keywords"]] = df_5class.apply(
    classify_email, axis=1
)

# ── Overall Accuracy ──────────────────────────────────────────────────────────
correct  = (df_5class["predicted_class"] == df_5class["actual_class"]).sum()
total    = len(df_5class)
accuracy = round(correct / total * 100, 2)

print(f"\n✅ Overall Accuracy (5 classes) : {correct}/{total}  ({accuracy}%)")

# ── Per Class Accuracy ────────────────────────────────────────────────────────
print(f"\n── Per Class Accuracy ────────────────────────────────────────")
print(f"{'Class':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Wrong':>8}")
print(f"{'─'*65}")

for class_name in ["Argus ID", "CQA Acknowledgement", "DSD Acknowledgement",
                   "For Follow Up", "PPM Request"]:
    class_df      = df_5class[df_5class["actual_class"] == class_name]
    if len(class_df) == 0:
        print(f"{class_name:<25} {'N/A':>8} {'0':>8} {'N/A':>10} {'N/A':>8}")
        continue
    class_correct = (class_df["predicted_class"] == class_name).sum()
    class_total   = len(class_df)
    class_acc     = round(class_correct / class_total * 100, 2)
    class_wrong   = class_total - class_correct
    flag          = "✅" if class_acc >= 90 else "⚠️" if class_acc >= 70 else "❌"
    print(f"{class_name:<25} {class_correct:>8} {class_total:>8} {class_acc:>9}%  {class_wrong:>6}  {flag}")

# ── Misclassification Breakdown ───────────────────────────────────────────────
print(f"\n── Misclassification Breakdown ───────────────────────────────")
df_wrong = df_5class[df_5class["predicted_class"] != df_5class["actual_class"]]
print(f"Total wrong : {len(df_wrong)}")
print(df_wrong.groupby(["actual_class","predicted_class"]).size().reset_index(name="count").to_string(index=False))

# ── Rule Triggered Breakdown ──────────────────────────────────────────────────
print(f"\n── Rule Triggered per Class ──────────────────────────────────")
print(df_5class.groupby(["actual_class","rule_triggered"]).size().reset_index(name="count").to_string(index=False))

# ── Misclassified files per class ─────────────────────────────────────────────
df_wrong_dsd  = df_wrong[df_wrong["actual_class"] == "DSD Acknowledgement"]
df_wrong_fu   = df_wrong[df_wrong["actual_class"] == "For Follow Up"]
df_wrong_argus= df_wrong[df_wrong["actual_class"] == "Argus ID"]
df_wrong_ppm  = df_wrong[df_wrong["actual_class"] == "PPM Request"]
df_wrong_cqa  = df_wrong[df_wrong["actual_class"] == "CQA Acknowledgement"]

# ── Save ──────────────────────────────────────────────────────────────────────
with pd.ExcelWriter("accuracy_5class.xlsx", engine="openpyxl") as writer:
    df_5class.to_excel(writer,      sheet_name="All",          index=False)
    df_wrong.to_excel(writer,       sheet_name="All Wrong",    index=False)
    df_wrong_dsd.to_excel(writer,   sheet_name="DSD Wrong",    index=False)
    df_wrong_fu.to_excel(writer,    sheet_name="FollowUp Wrong",index=False)
    df_wrong_argus.to_excel(writer, sheet_name="Argus Wrong",  index=False)
    df_wrong_ppm.to_excel(writer,   sheet_name="PPM Wrong",    index=False)
    df_wrong_cqa.to_excel(writer,   sheet_name="CQA Wrong",    index=False)

print(f"\n✅ Saved to accuracy_5class.xlsx with separate sheets per class")

##find which followups following to ppm mail 
# ── Follow Up emails now predicted as PPM ─────────────────────────────────────
df_fu_as_ppm = df_4class[
    (df_4class["actual_class"]    == "For Follow Up") &
    (df_4class["predicted_class"] == "PPM Request")
].copy()

print(f"Follow Up → PPM misclassified : {len(df_fu_as_ppm)}")
print(f"\n── PPM words triggering in Follow Up emails ──────")
print(df_fu_as_ppm["matched_keywords"].value_counts().head(20))
##finding the culprit word
from collections import Counter
import re

# See which PPM words appear most in wrongly classified Follow Up emails
all_keywords = []
for kw in df_fu_as_ppm["matched_keywords"]:
    all_keywords.extend(eval(kw))

print("PPM words causing Follow Up misclassification:")
for word, count in Counter(all_keywords).most_common():
    print(f"   {word:<20} {count:>5}")


##trigger list
# ── CQA Acknowledgement — ALL 3 words must match ──────────────────────────────
CQA_TRIGGER = [
    "acknowledge",   # or acknowledged/acknowledgement
    "receipt",
    "compliant",
]
CQA_REQUIRED_WORDS = ["receipt", "compliant"]   # these must be present along with acknowledge

# ── Argus ID ──────────────────────────────────────────────────────────────────
ARGUS_TRIGGER = ["argus"]

# ── PPM Request ───────────────────────────────────────────────────────────────
PPM_TRIGGER = [
    "revert", "prepaid", "mailer", "ppm",
    "investigated", "initiate", "findings",
]
PPM_MIN_MATCHES = 2

# ── DSD Acknowledgement ───────────────────────────────────────────────────────
DSD_TRIGGER = [
    "acknowledge", "acknowledged", "acknowledgement", "acknowledgment",
]

# ── For Follow Up ─────────────────────────────────────────────────────────────
FOLLOWUP_UNIQUE_WORDS = [
    "investigation", "batch", "sample", "kindly", "team",
    "observed", "provide", "patient", "information", "discrepancy",
    "found", "were",
]
FOLLOWUP_MIN_MATCHES = 2

# ── Overlap words ─────────────────────────────────────────────────────────────
OVERLAP_WORDS = ["colleague", "below", "find", "case", "greetings", "receipt"]

print("✅ All trigger lists loaded")
print(f"   Argus    : {len(ARGUS_TRIGGER)}")
print(f"   CQA      : {len(CQA_TRIGGER)}")
print(f"   PPM      : {len(PPM_TRIGGER)}")
print(f"   DSD      : {len(DSD_TRIGGER)}")
print(f"   Follow Up: {len(FOLLOWUP_UNIQUE_WORDS)}")
