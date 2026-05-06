# ── 95 unclassified CQA bodies ────────────────────────────────────────────────
df_cqa_unclassified = df_cqa_wrong[
    df_cqa_wrong["predicted_class"] == "Unclassified"
].copy()

print("── Sample Unclassified CQA bodies ───────────────────────────")
for i, row in df_cqa_unclassified.head(5).iterrows():
    print(f"\nBody : {row['pure_body'][:400]}")
    print("─" * 60)

# ── 77 Follow Up going to CQA ─────────────────────────────────────────────────
df_fu_as_cqa = df_5class[
    (df_5class["actual_class"]    == "For Follow Up") &
    (df_5class["predicted_class"] == "CQA Acknowledgement")
].copy()

print("\n── Sample Follow Up → CQA bodies ────────────────────────────")
for i, row in df_fu_as_cqa.head(5).iterrows():
    print(f"\nBody : {row['pure_body'][:400]}")
    print("─" * 60)
