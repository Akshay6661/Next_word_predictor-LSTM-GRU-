#Diagnose — Check Raw Body Dict
# Check what raw body looks like for empty emails
df_empty = df_live[
    df_live["body_full"].isna() | 
    (df_live["body_full"].str.len() < 5)
].copy()

print(f"Emails with empty body_full : {len(df_empty)}")

# Check their bodyPreview and subject
print(df_empty[["subject", "bodyPreview", "hasAttachments"]].head(10))


#Add this temporarily in fetch_all_emails to see raw body:
# After fetching, check a few raw body dicts
for email in emails[:5]:
    print(f"Subject      : {email.get('subject', '')}")
    print(f"Body type    : {email.get('body', {}).get('contentType', 'N/A')}")
    print(f"Body content : {email.get('body', {}).get('content', '')[:100]}")
    print("─" * 50)



#Fix — Handle Empty Body in extract_actual_body
def extract_actual_body(body_dict):
    if not body_dict:
        return ""

    content      = body_dict.get("content", "")
    content_type = body_dict.get("contentType", "")

    # ── If body is completely empty ────────────────────────────────────────────
    if not content or content.strip() == "":
        return ""

    # ── If body is just whitespace or HTML skeleton ────────────────────────────
    # Some emails have <html><body></body></html> with no actual content
    text_only = re.sub(r"<[^>]+>", "", content).strip()
    if len(text_only) < 5:
        return ""

    # ── Rest of existing cleaning ──────────────────────────────────────────────
    # ... your existing code continues here


#Fix — Fallback Chain in Cell 9
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
