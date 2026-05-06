# ── Get CQA actual emails ─────────────────────────────────────────────────────
df_cqa_actual = df_5class[df_5class["actual_class"] == "CQA Acknowledgement"].copy()

print(f"Total CQA emails : {len(df_cqa_actual)}")
print(f"Predicted as CQA : {(df_cqa_actual['predicted_class'] == 'CQA Acknowledgement').sum()}")
print(f"Predicted as what:")
print(df_cqa_actual["predicted_class"].value_counts())

# ── Check if trigger words exist in CQA emails ────────────────────────────────
print(f"\n── Word presence in CQA emails ───────────────────────────────")

check_words = ["acknowledge", "acknowledged", "acknowledgement", 
               "receipt", "compliant", "below"]

for word in check_words:
    count = df_cqa_actual["pure_body"].str.contains(word, case=False, na=False).sum()
    pct   = round(count / len(df_cqa_actual) * 100, 1)
    print(f"   {word:<20} found in {count:>5} emails  ({pct}%)")

# ── Sample a CQA email body ───────────────────────────────────────────────────
print(f"\n── Sample CQA pure_body ──────────────────────────────────────")
print(df_cqa_actual["pure_body"].iloc[0][:500])
