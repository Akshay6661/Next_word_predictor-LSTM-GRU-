df_cqa_actual = df_5class[df_5class["actual_class"] == "CQA Acknowledgement"].copy()

# ── Check all word combinations ───────────────────────────────────────────────
has_ack       = df_cqa_actual["pure_body"].str.contains("acknowledge",  case=False, na=False)
has_receipt   = df_cqa_actual["pure_body"].str.contains("receipt",      case=False, na=False)
has_complaint = df_cqa_actual["pure_body"].str.contains("complaint",    case=False, na=False)
has_below     = df_cqa_actual["pure_body"].str.contains("below",        case=False, na=False)
has_revert    = df_cqa_actual["pure_body"].str.contains("revert",       case=False, na=False)
has_findings  = df_cqa_actual["pure_body"].str.contains("findings",     case=False, na=False)
has_comply    = df_cqa_actual["pure_body"].str.contains("comply",       case=False, na=False)
has_quality   = df_cqa_actual["pure_body"].str.contains("quality",      case=False, na=False)
has_cqa       = df_cqa_actual["pure_body"].str.contains("cqa",          case=False, na=False)

print("── Word Combinations in CQA emails ──────────────────────────")
print(f"ack + receipt + complaint  : {(has_ack & has_receipt & has_complaint).sum()}")
print(f"receipt + complaint only   : {(has_receipt & has_complaint).sum()}")
print(f"ack + receipt only         : {(has_ack & has_receipt).sum()}")
print(f"ack + complaint only       : {(has_ack & has_complaint).sum()}")
print(f"receipt only               : {has_receipt.sum()}")
print(f"complaint only             : {has_complaint.sum()}")
print(f"below                      : {has_below.sum()}")
print(f"revert                     : {has_revert.sum()}")
print(f"findings                   : {has_findings.sum()}")
print(f"comply                     : {has_comply.sum()}")
print(f"quality                    : {has_quality.sum()}")
print(f"cqa                        : {has_cqa.sum()}")

# ── Show samples of CQA emails that are NOT being classified correctly ─────────
df_cqa_wrong = df_cqa_actual[df_cqa_actual["predicted_class"] != "CQA Acknowledgement"]
print(f"\n── Sample wrong CQA bodies ───────────────────────────────────")
for i, row in df_cqa_wrong.head(5).iterrows():
    print(f"\nPredicted as : {row['predicted_class']}")
    print(f"Body         : {row['pure_body'][:300]}")
    print("─" * 50)



### debug dsd 

df_dsd_wrong = df_5class[
    (df_5class["actual_class"]    == "DSD Acknowledgement") &
    (df_5class["predicted_class"] != "DSD Acknowledgement")
]

print(f"\n── DSD misclassified as ─────────────────────────────────────")
print(df_dsd_wrong["predicted_class"].value_counts())

print(f"\n── DSD wrong sample bodies ──────────────────────────────────")
for i, row in df_dsd_wrong.head(3).iterrows():
    print(f"\nPredicted as : {row['predicted_class']}")
    print(f"Body         : {row['pure_body'][:300]}")
    print("─" * 50)
