# ── Follow Up wrong ───────────────────────────────────────────────────────────
df_fu_wrong = df_5class[
    (df_5class["actual_class"]    == "For Follow Up") &
    (df_5class["predicted_class"] != "For Follow Up")
].copy()

print(f"Follow Up wrong : {len(df_fu_wrong)}")
print(df_fu_wrong["predicted_class"].value_counts())

# ── Follow Up going to CQA ────────────────────────────────────────────────────
df_fu_as_cqa = df_fu_wrong[df_fu_wrong["predicted_class"] == "CQA Acknowledgement"]
print(f"\nFollow Up → CQA : {len(df_fu_as_cqa)}")
print(f"\n── Rule triggered in FU→CQA ──────────────────────────────────")
print(df_fu_as_cqa["rule_triggered"].value_counts())

# ── DSD wrong ────────────────────────────────────────────────────────────────
df_dsd_wrong = df_5class[
    (df_5class["actual_class"]    == "DSD Acknowledgement") &
    (df_5class["predicted_class"] != "DSD Acknowledgement")
].copy()

print(f"\nDSD wrong : {len(df_dsd_wrong)}")
print(df_dsd_wrong["predicted_class"].value_counts())
print(f"\n── Rule triggered in DSD wrong ───────────────────────────────")
print(df_dsd_wrong["rule_triggered"].value_counts())
