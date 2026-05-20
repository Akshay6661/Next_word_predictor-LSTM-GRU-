# =============================================================================
# EXTRACT.PY — Data Ingestion + Body Extraction + Datetime Processing
# Handles: fetch emails, extract body, process datetime, extract case number
# =============================================================================

import re
import html
import time
import pytz
import requests
import pandas as pd
import urllib3
from datetime import datetime

import config
from logger import log

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# BODY EXTRACTION — Strips HTML + Thread History
# =============================================================================

def extract_actual_body(body_dict):
    """Extract clean text from raw Graph API body dict"""
    if not body_dict:
        return ""

    content = body_dict.get("content", "")

    # Step 1: Remove HTML comments
    content = re.sub(r"<!--.*?-->", " ", content, flags=re.DOTALL)

    # Step 2: Remove style and script blocks
    content = re.sub(r"<style.*?>.*?</style>",  " ", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<script.*?>.*?</script>", " ", content, flags=re.DOTALL | re.IGNORECASE)

    # Step 3: Replace block tags with newlines
    content = re.sub(r"<br\s*/?>", "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</p>",      "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</div>",    "\n", content, flags=re.IGNORECASE)

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

    return content.strip() if content.strip() else ""


# =============================================================================
# PURE BODY — Strips Signatures, CAUTION Banners, People Tags
# =============================================================================

def extract_pure_body(text):
    """Remove signatures, CAUTION banners, headers from body text"""
    if not text or pd.isna(text):
        return ""

    # Step 1: Remove CAUTION banner
    text = re.sub(r"CAUTION.*?safe\.?", " ", text, flags=re.IGNORECASE | re.DOTALL)

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
    for pattern in [r"this email.*?confidential.*", r"this message.*?intended.*",
                    r"disclaimer.*", r"caution.*?external email.*"]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Step 8: Final cleanup
    text = re.sub(r"\b[a-zA-Z]{1,2}\b", " ", text)
    text = re.sub(r" {2,}",              " ", text)
    text = re.sub(r"\n{2,}",            "\n", text)

    return text.strip()


# =============================================================================
# FETCH ALL EMAILS — Graph API with IST to UTC conversion
# =============================================================================

def fetch_all_emails(headers, start_date=None, start_time="00:00:00",
                                end_date=None,   end_time="23:59:59"):
    """Fetch all emails from Outlook via Graph API within IST date range"""

    ist_tz = pytz.timezone("Asia/Kolkata")

    if start_date and end_date:
        start_ist   = ist_tz.localize(datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S"))
        end_ist     = ist_tz.localize(datetime.strptime(f"{end_date}   {end_time}",   "%Y-%m-%d %H:%M:%S"))
        start_utc   = start_ist.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc     = end_ist.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_filter = f"&$filter=receivedDateTime ge {start_utc} and receivedDateTime le {end_utc}"
        log(f"IST Range : {start_date} {start_time} to {end_date} {end_time} IST")
        log(f"UTC Range : {start_utc} to {end_utc}")

    elif start_date:
        start_ist   = ist_tz.localize(datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S"))
        start_utc   = start_ist.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_filter = f"&$filter=receivedDateTime ge {start_utc}"
        log(f"IST From : {start_date} {start_time} IST")

    elif end_date:
        end_ist     = ist_tz.localize(datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M:%S"))
        end_utc     = end_ist.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_filter = f"&$filter=receivedDateTime le {end_utc}"
        log(f"IST To : {end_date} {end_time} IST")

    else:
        date_filter = ""
        log("No date filter — fetching all emails")

    url = (
        f"https://graph.microsoft.com/v1.0/users/{config.USER_EMAIL}/messages"
        f"?$select=id,subject,body,bodyPreview,from,toRecipients,"
        f"receivedDateTime,hasAttachments,conversationId"
        f"&$top=1000"
        f"&$orderby=receivedDateTime desc"
        f"{date_filter}"
    )

    emails  = []
    t_start = time.time()

    while url:
        resp = requests.get(url, headers=headers, verify=False).json()
        if "error" in resp:
            log(f"API Error: {resp['error']['message']}", "ERROR")
            break
        emails.extend(resp.get("value", []))
        url = resp.get("@odata.nextLink")

    elapsed = round(time.time() - t_start, 1)
    log(f"Fetched {len(emails)} emails in {elapsed}s")

    df = pd.DataFrame(emails)
    if df.empty:
        log("No emails found for this date range", "WARNING")
        return df

    # Flatten nested columns
    df["sender_name"]   = df["from"].apply(lambda x: x["emailAddress"]["name"])
    df["sender_email"]  = df["from"].apply(lambda x: x["emailAddress"]["address"])
    df["body_full"]     = df["body"].apply(extract_actual_body)
    df["to_recipients"] = df["toRecipients"].apply(
                            lambda x: ", ".join([r["emailAddress"]["address"] for r in x]))

    # Drop raw nested columns
    cols_to_drop = ["@odata.etag", "@odata.type", "from", "body", "toRecipients"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Sort + thread details
    df = df.sort_values(["conversationId", "receivedDateTime"]).reset_index(drop=True)
    df["reply_count"]          = df.groupby("conversationId")["id"].transform("count")
    df["reply_position"]       = df.groupby("conversationId")["receivedDateTime"].rank(method="first").astype(int)
    df["is_thread"]            = df["reply_count"] > 1
    df["is_original_email"]    = df["reply_position"] == 1
    df["thread_started_at"]    = df.groupby("conversationId")["receivedDateTime"].transform("min")
    df["thread_last_reply_at"] = df.groupby("conversationId")["receivedDateTime"].transform("max")
    df["all_participants"]     = df.groupby("conversationId")["sender_email"].transform(
                                     lambda x: ", ".join(x.unique()))

    desired_cols = [
        "conversationId", "is_thread", "reply_count", "reply_position",
        "is_original_email", "thread_started_at", "thread_last_reply_at",
        "all_participants", "id", "receivedDateTime", "sender_name",
        "sender_email", "to_recipients", "subject", "bodyPreview",
        "body_full", "hasAttachments"
    ]
    df = df[[col for col in desired_cols if col in df.columns]]
    log(f"Shape after fetch: {df.shape}")
    return df


# =============================================================================
# DATETIME + CASE NUMBER PROCESSING
# =============================================================================

def process_datetime_and_case(df):
    """Convert receivedDateTime to EST, split date/time, extract case number"""
    est = pytz.timezone("US/Eastern")

    df["receivedDateTime"]     = pd.to_datetime(df["receivedDateTime"], utc=True)
    df["date"]                 = df["receivedDateTime"].dt.tz_convert(est).dt.strftime("%Y-%m-%d")
    df["time"]                 = df["receivedDateTime"].dt.tz_convert(est).dt.strftime("%a %I:%M %p")

    # Strip timezone for Excel compatibility
    df["receivedDateTime"]     = df["receivedDateTime"].dt.tz_localize(None)
    df["thread_started_at"]    = pd.to_datetime(df["thread_started_at"],    utc=True).dt.tz_localize(None)
    df["thread_last_reply_at"] = pd.to_datetime(df["thread_last_reply_at"], utc=True).dt.tz_localize(None)

    # Extract case number from subject
    CASE_NUMBER_PATTERN = r"[A-Za-z]{5}\d{2}-\d{4,5}"
    df["case_number"] = df["subject"].apply(
        lambda x: re.search(CASE_NUMBER_PATTERN, x).group()
                  if pd.notna(x) and re.search(CASE_NUMBER_PATTERN, x)
                  else None
    )

    log(f"Datetime converted to EST | Case numbers: {df['case_number'].notna().sum()}")
    return df


# =============================================================================
# EXTRACT ACTUAL BODY PER REPLY POSITION
# =============================================================================

def extract_new_content(df_input):
    """Extract only new content per email in thread (strips previous replies)"""
    df_output                = df_input.copy()
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
                    df_output.at[idx, "actual_body"] = (
                        current_body[:overlap_pos].strip() if overlap_pos > 0 else current_body
                    )
                else:
                    df_output.at[idx, "actual_body"] = current_body

            prev_body = current_body

    # Apply pure body cleaning
    df_output["pure_body"] = df_output["actual_body"].apply(extract_pure_body)

    log(f"actual_body: {(df_output['actual_body'] != '').sum()} | pure_body: {(df_output['pure_body'] != '').sum()}")
    return df_output


# =============================================================================
# FALLBACK CHAIN
# =============================================================================

def apply_fallback(df):
    """Apply fallback chain: actual_body → bodyPreview → empty"""

    # actual_body fallback
    df["actual_body"] = df.apply(
        lambda row: (
            row["actual_body"]
            if pd.notna(row["actual_body"]) and len(str(row["actual_body"]).strip()) > 10
            else row["bodyPreview"]
            if pd.notna(row["bodyPreview"]) and len(str(row["bodyPreview"]).strip()) > 10
            else ""
        ), axis=1
    )

    # pure_body fallback — runs through extract_pure_body to clean CAUTION
    df["pure_body"] = df.apply(
        lambda row: (
            row["pure_body"]
            if pd.notna(row["pure_body"]) and len(str(row["pure_body"]).strip()) > 10
            else extract_pure_body(str(row["bodyPreview"]))
            if pd.notna(row["bodyPreview"]) and len(str(row["bodyPreview"]).strip()) > 10
            else ""
        ), axis=1
    )

    log(f"Fallback — actual_body: {(df['actual_body'] != '').sum()} | pure_body: {(df['pure_body'] != '').sum()}")
    return df
