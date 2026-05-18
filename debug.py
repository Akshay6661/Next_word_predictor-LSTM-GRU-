# ── Step 3b: Fallback on actual_body ─────────────────────────────────────────
df_live["actual_body"] = df_live.apply(
    lambda row: (
        row["actual_body"]
        if pd.notna(row["actual_body"]) and len(str(row["actual_body"]).strip()) > 10
        else row["bodyPreview"]
        if pd.notna(row["bodyPreview"]) and len(str(row["bodyPreview"]).strip()) > 10
        else ""
    ), axis=1
)
