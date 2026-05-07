# ── PPM wrong breakdown ───────────────────────────────────────────────────────
df_ppm_wrong = df_5class[
    (df_5class["actual_class"]    == "PPM Request") &
    (df_5class["predicted_class"] != "PPM Request")
].copy()

print(f"PPM wrong total : {len(df_ppm_wrong)}")
print(f"\n── PPM misclassified as ─────────────────────────────────────")
print(df_ppm_wrong["predicted_class"].value_counts())

# ── What words are in wrong PPM emails ───────────────────────────────────────
from collections import Counter
import re

all_text = " ".join(
    (df_ppm_wrong["subject"].fillna("") + " " + 
     df_ppm_wrong["pure_body"].fillna("")).tolist()
).lower()

words     = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
words     = [w for w in words if w not in stop_words]
counter   = Counter(words)

print(f"\n── Top words in wrong PPM emails ────────────────────────────")
print(f"   {'Word':<25} {'Count':>6}")
print(f"   {'─'*35}")
for word, count in counter.most_common(30):
    in_trigger = word in PPM_STRONG_TRIGGER or word in PPM_WEAK_TRIGGER
    flag       = "✅ already" if in_trigger else "⬅️  missing"
    print(f"   {word:<25} {count:>6}  {flag}")

# ── Sample wrong PPM bodies ───────────────────────────────────────────────────
print(f"\n── Sample wrong PPM bodies ──────────────────────────────────")
for i, row in df_ppm_wrong.head(5).iterrows():
    print(f"\nPredicted as : {row['predicted_class']}")
    print(f"Body         : {row['pure_body'][:400]}")
    print("─" * 60)


## next code block

# ── CQA wrong breakdown ───────────────────────────────────────────────────────
df_cqa_wrong = df_5class[
    (df_5class["actual_class"]    == "CQA Acknowledgement") &
    (df_5class["predicted_class"] != "CQA Acknowledgement")
].copy()

print(f"\nCQA wrong total : {len(df_cqa_wrong)}")
print(f"\n── CQA misclassified as ─────────────────────────────────────")
print(df_cqa_wrong["predicted_class"].value_counts())

# ── What words are in wrong CQA emails ───────────────────────────────────────
all_text = " ".join(
    (df_cqa_wrong["subject"].fillna("") + " " + 
     df_cqa_wrong["pure_body"].fillna("")).tolist()
).lower()

words   = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
words   = [w for w in words if w not in stop_words]
counter = Counter(words)

print(f"\n── Top words in wrong CQA emails ────────────────────────────")
print(f"   {'Word':<25} {'Count':>6}  {'% of wrong CQA':>15}")
print(f"   {'─'*50}")
for word, count in counter.most_common(30):
    pct        = round(count / len(df_cqa_wrong) * 100, 1)
    in_trigger = word in CQA_REQUIRED_WORDS or word in CQA_INVESTIGATE_WORDS or word in CQA_DEVICE_WORDS
    flag       = "✅ already" if in_trigger else "⬅️  missing"
    print(f"   {word:<25} {count:>6}  {pct:>14}%  {flag}")

# ── Sample wrong CQA bodies ───────────────────────────────────────────────────
print(f"\n── Sample wrong CQA bodies ──────────────────────────────────")
for i, row in df_cqa_wrong.head(5).iterrows():
    print(f"\nPredicted as : {row['predicted_class']}")
    print(f"Body         : {row['pure_body'][:400]}")
    print("─" * 60)




# =============================================================================
# CELL 11 — EMAIL RECON FILE
# =============================================================================

# ── Build recon dataframe ─────────────────────────────────────────────────────
df_recon = pd.DataFrame()

# ── Map columns from classified report ───────────────────────────────────────
df_recon["From"]                 = df_live["sender_name"]
df_recon["Subject"]              = df_live["subject"]
df_recon["Received Time"]        = df_live["time"]          # Sun HH:MM AM/PM
df_recon["Received Date"]        = df_live["date"]
df_recon["Owner"]                = ""                        # blank — filled manually
df_recon["Action"]               = ""                        # blank — filled manually
df_recon["Status"]               = ""                        # blank — filled manually
df_recon["pure_body"]            = df_live["pure_body"]
df_recon["Comments"]             = df_live["predicted_class"]
df_recon["Checked by"]           = ""                        # blank — filled manually
df_recon["TAT status"]           = ""                        # blank — filled manually
df_recon["Communication status"] = ""                        # blank — filled manually
df_recon["case_number"]          = df_live["case_number"]

# ── Sort by date and time ─────────────────────────────────────────────────────
df_recon["sort_dt"] = pd.to_datetime(
                        df_live["date"].astype(str) + " " +
                        df_live["time"].str.extract(r"(\d{2}:\d{2} [AP]M)")[0],
                        format="%Y-%m-%d %I:%M %p",
                        errors="coerce"
                      )

df_recon = df_recon.sort_values("sort_dt", ascending=True).reset_index(drop=True)
df_recon = df_recon.drop(columns=["sort_dt"])

# ── Save recon file ───────────────────────────────────────────────────────────
recon_file = f"email_recon_{START_DATE}_{START_TIME.replace(':','')}_to_{END_DATE}_{END_TIME.replace(':','')}_IST.xlsx"

df_recon.to_excel(recon_file, index=False)

print(f"✅ Email recon saved : {recon_file}")
print(f"   Rows             : {len(df_recon)}")
print(f"   Columns          : {df_recon.columns.tolist()}")
print(f"\n── Comments distribution ─────────────────────────────────────")
print(df_recon["Comments"].value_counts())
