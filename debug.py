# ── After extract_new_content ─────────────────────────────────────────────────
# Fallback chain: pure_body → body_full → bodyPreview
df_live["pure_body"] = df_live.apply(
    lambda row: (
        row["pure_body"]   if pd.notna(row["pure_body"])   and len(str(row["pure_body"]))   > 10
        else row["body_full"]   if pd.notna(row["body_full"])   and len(str(row["body_full"]))   > 10
        else row["bodyPreview"] if pd.notna(row["bodyPreview"]) and len(str(row["bodyPreview"])) > 10
        else ""
    ),
    axis=1
)

# ── Check how many needed fallback ───────────────────────────────────────────
print(f"✅ pure_body filled   : {(df_live['pure_body'].str.len() > 10).sum()}")
print(f"⚠️  Still empty        : {(df_live['pure_body'].str.len() <= 10).sum()}")
