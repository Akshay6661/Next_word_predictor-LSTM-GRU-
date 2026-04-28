# =============================================================================
# EMAIL CLASSIFICATION PIPELINE
# =============================================================================
# Pull live emails from Outlook via Graph API, clean and classify them
# based on rule-based engine with confidence scoring
# =============================================================================


# =============================================================================
# CELL 1 — IMPORTS & CONFIG
# =============================================================================

import re
import html
import time
import pytz
import requests
import msal
import pandas as pd
import urllib3
from datetime import timezone

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Azure AD Credentials ──────────────────────────────────────────────────────
CLIENT_ID     = "your-client-id"
CLIENT_SECRET = "your-client-secret"
TENANT_ID     = "your-tenant-id"
USER_EMAIL    = "yourname@yourcompany.com"

SCOPES        = ["https://graph.microsoft.com/.default"]

# ── Date + Time Range in IST — change these to pull different periods ────────
START_DATE    = "2026-04-21"   # yyyy-mm-dd
START_TIME    = "09:30"        # HH:MM 24hr IST

END_DATE      = "2026-04-28"   # yyyy-mm-dd
END_TIME      = "09:30"        # HH:MM 24hr IST


# =============================================================================
# CELL 2 — AUTHENTICATION
# =============================================================================

def get_access_token():
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{TENANT_ID}",
        client_credential=CLIENT_SECRET
    )
    result = app.acquire_token_for_client(scopes=SCOPES)
    if "access_token" not in result:
        raise Exception(f"Auth failed: {result.get('error_description')}")
    return result["access_token"]

TOKEN   = get_access_token()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

print("✅ Authentication successful")


# =============================================================================
# CELL 3 — BODY EXTRACTION (STRIPS HTML + THREAD HISTORY)
# =============================================================================

def extract_actual_body(body_dict):
    if not body_dict:
        return ""

    content = body_dict.get("content", "")

    # Step 1: Remove HTML comments
    content = re.sub(r"<!--.*?-->",             " ", content, flags=re.DOTALL)

    # Step 2: Remove style and script blocks
    content = re.sub(r"<style.*?>.*?</style>",  " ", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<script.*?>.*?</script>","  ", content, flags=re.DOTALL | re.IGNORECASE)

    # Step 3: Replace block tags with newlines
    content = re.sub(r"<br\s*/?>",  "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</p>",       "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</div>",     "\n", content, flags=re.IGNORECASE)

    # Step 4: Strip all remaining HTML tags
    content = re.sub(r"<[^>]+>", "", content)

    # Step 5: Decode HTML entities
    content = html.unescape(content)
    content = re.sub(r"&[a-zA-Z]+;", " ", content)
    content = re.sub(r"&#\d+;",      " ", content)

    # Step 6: Cut at thread dividers
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

    # Step 7: Clean garbage characters
    content = re.sub(r"[^\x00-\x7F]+", " ", content)
    content = re.sub(r"[\t\r]+",        " ", content)
    content = re.sub(r"\n{3,}",        "\n\n", content)
    content = re.sub(r" {2,}",          " ", content)
    content = re.sub(r"[;:]{2,}",       "",  content)
    content = content.strip()

    return content if content else ""

print("✅ extract_actual_body defined")


# =============================================================================
# CELL 4 — PURE BODY (STRIPS SIGNATURES, CAUTION BANNERS, PEOPLE TAGS)
# =============================================================================

def extract_pure_body(text):
    if not text or pd.isna(text):
        return ""

    # Step 1: Remove CAUTION banner
    text = re.sub(
        r"CAUTION.*?safe\.?",
        " ", text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Step 2: Cut at signature indicators
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

    # Step 3: Remove From/To/CC/Sent header lines
    text = re.sub(r"from\s*:.*",    "", text, flags=re.IGNORECASE)
    text = re.sub(r"to\s*:.*",      "", text, flags=re.IGNORECASE)
    text = re.sub(r"cc\s*:.*",      "", text, flags=re.IGNORECASE)
    text = re.sub(r"sent\s*:.*",    "", text, flags=re.IGNORECASE)
    text = re.sub(r"subject\s*:.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"date\s*:.*",    "", text, flags=re.IGNORECASE)

    # Step 4: Remove signature lines
    text = re.sub(r"phone\s*:.*",   "", text, flags=re.IGNORECASE)
    text = re.sub(r"tel\s*:.*",     "", text, flags=re.IGNORECASE)
    text = re.sub(r"mob\s*:.*",     "", text, flags=re.IGNORECASE)
    text = re.sub(r"fax\s*:.*",     "", text, flags=re.IGNORECASE)
    text = re.sub(r"email\s*:.*",   "", text, flags=re.IGNORECASE)
    text = re.sub(r"website\s*:.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.\S+",      "", text, flags=re.IGNORECASE)

    # Step 5: Remove phone numbers
    text = re.sub(r"\+?[\d\s\-\(\)]{7,}", " ", text)

    # Step 6: Remove pipe separated signature lines
    text = re.sub(r"[^.!?]*\|[^.!?]*", " ", text)

    # Step 7: Remove disclaimer blocks
    disclaimer_patterns = [
        r"this email.*?confidential.*",
        r"this message.*?intended.*",
        r"disclaimer.*",
        r"caution.*?external email.*",
    ]
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Step 8: Final cleanup
    text = re.sub(r"\b[a-zA-Z]{1,2}\b", " ", text)
    text = re.sub(r" {2,}",              " ", text)
    text = re.sub(r"\n{2,}",            "\n", text)
    text = text.strip()

    return text

print("✅ extract_pure_body defined")


# =============================================================================
# CELL 5 — FETCH ALL EMAILS
# =============================================================================

def fetch_all_emails(start_date=None, start_time="00:00",
                     end_date=None,   end_time="23:59"):

    from datetime import datetime

    ist = pytz.timezone("Asia/Kolkata")

    # Build date filter
    if start_date and end_date:

        # Parse IST datetime and convert to UTC
        start_ist = ist.localize(datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M"))
        end_ist   = ist.localize(datetime.strptime(f"{end_date} {end_time}",     "%Y-%m-%d %H:%M"))

        start_utc = start_ist.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc   = end_ist.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        date_filter = f"&$filter=receivedDateTime ge {start_utc} and receivedDateTime le {end_utc}"

        print(f"📅 IST Range : {start_date} {start_time} → {end_date} {end_time} IST")
        print(f"🌐 UTC Range : {start_utc} → {end_utc}")

    elif start_date:
        start_ist   = ist.localize(datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M"))
        start_utc   = start_ist.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_filter = f"&$filter=receivedDateTime ge {start_utc}"
        print(f"📅 IST From  : {start_date} {start_time} IST → {start_utc} UTC")

    elif end_date:
        end_ist     = ist.localize(datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M"))
        end_utc     = end_ist.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_filter = f"&$filter=receivedDateTime le {end_utc}"
        print(f"📅 IST To    : {end_date} {end_time} IST → {end_utc} UTC")

    else:
        date_filter = ""
        print(f"📅 No date filter — fetching all emails")

    url = (
        f"https://graph.microsoft.com/v1.0/users/{USER_EMAIL}/messages"
        f"?$select=id,subject,body,bodyPreview,from,toRecipients,"
        f"receivedDateTime,hasAttachments,conversationId"
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
        print("⚠️ No emails found for this date range")
        return df

    # ── Flatten nested columns ────────────────────────────────────────────────
    df["sender_name"]   = df["from"].apply(lambda x: x["emailAddress"]["name"])
    df["sender_email"]  = df["from"].apply(lambda x: x["emailAddress"]["address"])
    df["body_full"]     = df["body"].apply(extract_actual_body)
    df["to_recipients"] = df["toRecipients"].apply(
                            lambda x: ", ".join([r["emailAddress"]["address"] for r in x])
                          )

    # ── Drop raw nested columns ───────────────────────────────────────────────
    cols_to_drop = ["@odata.etag", "@odata.type", "from", "body", "toRecipients"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # ── Sort by conversation and time ─────────────────────────────────────────
    df = df.sort_values(["conversationId", "receivedDateTime"]).reset_index(drop=True)

    # ── Thread details ────────────────────────────────────────────────────────
    df["reply_count"]          = df.groupby("conversationId")["id"].transform("count")
    df["reply_position"]       = df.groupby("conversationId")["receivedDateTime"].rank(method="first").astype(int)
    df["is_thread"]            = df["reply_count"] > 1
    df["is_original_email"]    = df["reply_position"] == 1
    df["thread_started_at"]    = df.groupby("conversationId")["receivedDateTime"].transform("min")
    df["thread_last_reply_at"] = df.groupby("conversationId")["receivedDateTime"].transform("max")
    df["all_participants"]     = df.groupby("conversationId")["sender_email"].transform(
                                    lambda x: ", ".join(x.unique())
                                 )

    # ── Reorder columns ───────────────────────────────────────────────────────
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

print("✅ fetch_all_emails defined")


# =============================================================================
# CELL 6 — CONVERT DATETIME TO EST + SPLIT DATE & TIME + CASE NUMBER
# =============================================================================

def process_datetime_and_case(df):

    est = pytz.timezone("US/Eastern")

    # Convert to EST
    df["receivedDateTime"] = pd.to_datetime(df["receivedDateTime"], utc=True)
    df["date"] = df["receivedDateTime"].dt.tz_convert(est).dt.strftime("%Y-%m-%d")
    df["time"] = df["receivedDateTime"].dt.tz_convert(est).dt.strftime("%I:%M %p")

    # Strip timezone for Excel compatibility
    df["receivedDateTime"]    = df["receivedDateTime"].dt.tz_localize(None)
    df["thread_started_at"]   = pd.to_datetime(df["thread_started_at"],   utc=True).dt.tz_localize(None)
    df["thread_last_reply_at"]= pd.to_datetime(df["thread_last_reply_at"],utc=True).dt.tz_localize(None)

    # Extract case number from subject
    CASE_NUMBER_PATTERN = r"[A-Za-z]{5}\d{2}-\d{4,5}"
    df["case_number"] = df["subject"].apply(
        lambda x: re.search(CASE_NUMBER_PATTERN, x).group()
                  if pd.notna(x) and re.search(CASE_NUMBER_PATTERN, x)
                  else None
    )

    print(f"✅ Date/time converted to EST")
    print(f"✅ Case numbers extracted : {df['case_number'].notna().sum()}")
    return df

print("✅ process_datetime_and_case defined")


# =============================================================================
# CELL 7 — EXTRACT ACTUAL BODY PER REPLY POSITION
# =============================================================================

def extract_new_content(df_input):

    df_output             = df_input.copy()
    df_output["actual_body"] = ""

    for conv_id, group in df_output.groupby("conversationId"):
        group     = group.sort_values("reply_position")
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

    # Apply pure body cleaning
    df_output["pure_body"] = df_output["actual_body"].apply(extract_pure_body)

    print(f"✅ actual_body extracted  : {(df_output['actual_body'] != '').sum()}")
    print(f"✅ pure_body cleaned      : {(df_output['pure_body']   != '').sum()}")
    return df_output

print("✅ extract_new_content defined")


# =============================================================================
# CELL 8 — CLASSIFICATION RULES + SCORING ENGINE
# =============================================================================

# ── Word Baskets ──────────────────────────────────────────────────────────────

ARGUS_TRIGGER = [
    "argus",
    # ← add more Argus trigger words here
]

DSD_TRIGGER = [
    "acknowledge", "acknowledged", "acknowledgement", "acknowledgment",
    # ← add more DSD trigger words here
]

FOLLOWUP_UNIQUE_WORDS = [
    "investigation", "batch", "sample", "kindly", "team",
    "observed", "provide", "patient", "information", "discrepancy",
    "found", "were",
    # ← add more Follow Up words here
]

OVERLAP_WORDS = [
    "colleague", "below", "find", "case", "greetings", "receipt"
]

FOLLOWUP_MIN_MATCHES = 2   # minimum unique words to confirm Follow Up

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


def classify_email(row):

    subject  = str(row["subject"]).lower()   if pd.notna(row["subject"])   else ""
    body     = str(row["pure_body"]).lower() if pd.notna(row["pure_body"]) else ""
    combined = f"{subject} {body}"
    words    = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined))

    # ── Rule 1: Argus ID — single word trigger ────────────────────────────────
    argus_hits = [w for w in ARGUS_TRIGGER if w in words]
    if argus_hits:
        return pd.Series({
            "predicted_class" : "Argus ID",
            "confidence"      : 0.97,
            "rule_triggered"  : "argus_trigger",
            "matched_keywords": str(argus_hits)
        })

    # ── Rule 2: DSD Acknowledgement — single word trigger ────────────────────
    dsd_hits = [w for w in DSD_TRIGGER if w in words]
    if dsd_hits:
        return pd.Series({
            "predicted_class" : "DSD Acknowledgement",
            "confidence"      : 0.97,
            "rule_triggered"  : "dsd_trigger",
            "matched_keywords": str(dsd_hits)
        })

    # ── Rule 3: Follow Up — needs multiple unique word matches ────────────────
    followup_hits = [w for w in FOLLOWUP_UNIQUE_WORDS if w in words]

    if len(followup_hits) >= FOLLOWUP_MIN_MATCHES:
        confidence = min(0.50 + (len(followup_hits) * 0.10), 0.99)
        return pd.Series({
            "predicted_class" : "For Follow Up",
            "confidence"      : round(confidence, 2),
            "rule_triggered"  : f"followup_{len(followup_hits)}_words_matched",
            "matched_keywords": str(followup_hits)
        })

    # ── Rule 4: Weak Follow Up — 1 unique + 2 overlap words ──────────────────
    overlap_hits = [w for w in OVERLAP_WORDS if w in words]

    if len(followup_hits) == 1 and len(overlap_hits) >= 2:
        return pd.Series({
            "predicted_class" : "For Follow Up",
            "confidence"      : 0.45,
            "rule_triggered"  : "followup_weak_signal",
            "matched_keywords": str(followup_hits + overlap_hits)
        })

    # ── Rule 5: Unclassified ──────────────────────────────────────────────────
    return pd.Series({
        "predicted_class" : "Unclassified",
        "confidence"      : 0.0,
        "rule_triggered"  : "no_match",
        "matched_keywords": "[]"
    })

print("✅ classify_email defined")


# =============================================================================
# CELL 9 — RUN FULL PIPELINE
# =============================================================================

print(f"\n📅 Pulling emails")
print(f"   From : {START_DATE} {START_TIME} IST")
print(f"   To   : {END_DATE} {END_TIME} IST")
print("─" * 50)

# Step 1: Fetch emails
df_live = fetch_all_emails(
    start_date=START_DATE, start_time=START_TIME,
    end_date=END_DATE,     end_time=END_TIME
)

if df_live.empty:
    print("⚠️ No emails found — check date range or credentials")
else:
    # Step 2: Convert datetime + extract case number
    df_live = process_datetime_and_case(df_live)

    # Step 3: Extract actual body per reply position + pure body
    df_live = extract_new_content(df_live)

    # Step 4: Classify
    df_live[["predicted_class", "confidence",
             "rule_triggered",  "matched_keywords"]] = df_live.apply(
        classify_email, axis=1
    )

    # Step 5: Summary
    print(f"\n✅ Classification Complete")
    print(f"{'─'*40}")
    print(df_live["predicted_class"].value_counts())

    print(f"\n── Confidence Distribution ──────────────────────")
    print(f"High   (> 0.7)   : {(df_live['confidence'] >  0.7).sum()}")
    print(f"Medium (0.4-0.7) : {((df_live['confidence'] >= 0.4) & (df_live['confidence'] <= 0.7)).sum()}")
    print(f"Low    (< 0.4)   : {(df_live['confidence'] <  0.4).sum()}")

    print(f"\n── Rule Triggered Breakdown ─────────────────────")
    print(df_live["rule_triggered"].value_counts())

    print(f"\n✅ Total emails processed : {len(df_live)}")


# =============================================================================
# CELL 10 — SAVE OUTPUT
# =============================================================================

    output_file = f"classified_emails_{START_DATE}_{START_TIME.replace(':','')}_to_{END_DATE}_{END_TIME.replace(':','')}_IST.xlsx"

    df_live.to_excel(output_file, index=False)
    print(f"\n✅ Saved to {output_file}")
    print(f"   Rows    : {len(df_live)}")
    print(f"   Columns : {len(df_live.columns)}")
