# ── CQA wrong — see actual body patterns ──────────────────────────────────────
df_cqa_wrong = df_5class[
    (df_5class["actual_class"]    == "CQA Acknowledgement") &
    (df_5class["predicted_class"] != "CQA Acknowledgement")
].copy()

print(f"CQA wrong total : {len(df_cqa_wrong)}")
print(f"\n── CQA misclassified as ─────────────────────────────────────")
print(df_cqa_wrong["predicted_class"].value_counts())

print(f"\n── Sample wrong CQA bodies ───────────────────────────────────")
for i, row in df_cqa_wrong.head(5).iterrows():
    print(f"\nPredicted as : {row['predicted_class']}")
    print(f"Body         : {row['pure_body'][:400]}")
    print("─" * 60)

# ── Follow Up wrong — what is stealing it ────────────────────────────────────
df_fu_wrong = df_5class[
    (df_5class["actual_class"]    == "For Follow Up") &
    (df_5class["predicted_class"] != "For Follow Up")
].copy()

print(f"\nFollow Up wrong total : {len(df_fu_wrong)}")
print(f"\n── Follow Up misclassified as ───────────────────────────────")
print(df_fu_wrong["predicted_class"].value_counts())

print(f"\n── Sample wrong Follow Up bodies ────────────────────────────")
for i, row in df_fu_wrong.head(3).iterrows():
    print(f"\nPredicted as : {row['predicted_class']}")
    print(f"Body         : {row['pure_body'][:400]}")
    print("─" * 60)
