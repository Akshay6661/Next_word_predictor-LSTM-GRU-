# =============================================================================
# MAIN.PY — Pipeline Orchestrator
# Entry point — imports and runs all modules in order
# Run: python main.py
# =============================================================================

import msal
import requests
import traceback
import urllib3

import config
from logger import log, save_log_to_sharepoint, log_lines
from extract import (
    fetch_all_emails,
    process_datetime_and_case,
    extract_new_content,
    apply_fallback
)
from classification import run_classification
from loader import save_raw_file, build_recon_df, save_recon_file

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================================
# AUTHENTICATION
# =============================================================================

def get_access_token():
    app = msal.ConfidentialClientApplication(
        config.CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{config.TENANT_ID}",
        client_credential=config.CLIENT_SECRET
    )
    result = app.acquire_token_for_client(scopes=config.SCOPES)
    if "access_token" not in result:
        raise Exception(f"Auth failed: {result.get('error_description')}")
    return result["access_token"]


def get_site_id(token):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp    = requests.get(
        f"https://graph.microsoft.com/v1.0/sites/{config.HOSTNAME}:/sites/{config.SITE_NAME}",
        headers=headers, verify=False
    )
    sp_site_id = resp.json()["id"]
    log(f"Site Name : {resp.json()['displayName']}")
    log(f"Site ID   : {sp_site_id}")
    return sp_site_id, headers


# =============================================================================
# LOG FILE NAME
# =============================================================================

LOG_FILE_NAME = (
    f"log_{config.START_DATE}_{config.START_TIME.replace(':','')}"
    f"_to_{config.END_DATE}_{config.END_TIME.replace(':','')}_IST.txt"
)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    TOKEN      = None
    SP_SITE_ID = None

    try:
        log("=" * 60)
        log("PIPELINE START")
        log(f"From : {config.START_DATE} {config.START_TIME} IST")
        log(f"To   : {config.END_DATE}   {config.END_TIME}   IST")
        log("=" * 60)

        # ── Auth ──────────────────────────────────────────────────────────────
        log("Step 0 : Authenticating...")
        TOKEN      = get_access_token()
        SP_SITE_ID, HEADERS = get_site_id(TOKEN)
        log("Authentication successful")

        # ── Step 1: Fetch Emails ──────────────────────────────────────────────
        log("Step 1 : Fetching emails...")
        df_live = fetch_all_emails(
            headers    = HEADERS,
            start_date = config.START_DATE,
            start_time = config.START_TIME,
            end_date   = config.END_DATE,
            end_time   = config.END_TIME
        )

        # ── Domain Filter ─────────────────────────────────────────────────────
        if config.ENABLE_DOMAIN_FILTER and not df_live.empty:
            before  = len(df_live)
            df_live = df_live[
                df_live["sender_email"].apply(
                    lambda x: any(domain in str(x).lower() for domain in config.FILTER_DOMAINS)
                )
            ].reset_index(drop=True)
            log(f"Domain filter — Before: {before} | After: {len(df_live)}")

        if df_live.empty:
            log("No emails found — check date range or credentials", "WARNING")
            return

        # ── Step 2: Process Datetime + Case Number ────────────────────────────
        log("Step 2 : Processing datetime + case number...")
        df_live = process_datetime_and_case(df_live)

        # ── Step 3: Extract Body ──────────────────────────────────────────────
        log("Step 3 : Extracting body content...")
        df_live = extract_new_content(df_live)

        # ── Step 3b/3c: Fallback Chain ────────────────────────────────────────
        log("Step 3b: Applying fallback chain...")
        df_live = apply_fallback(df_live)

        # ── Step 4: Sort by EST Time ──────────────────────────────────────────
        log("Step 4 : Sorting by EST time...")
        import pandas as pd
        df_live["sort_dt"] = pd.to_datetime(
            df_live["date"].astype(str) + " " + df_live["time"].astype(str),
            errors="coerce"
        )
        df_live = df_live.sort_values("sort_dt", ascending=True).reset_index(drop=True)
        df_live = df_live.drop(columns={"sort_dt"})

        # ── Step 5: Classify ──────────────────────────────────────────────────
        log("Step 5 : Classifying emails...")
        df_live = run_classification(df_live)

        # ── Step 6: Save Raw File ─────────────────────────────────────────────
        log("Step 6 : Saving raw file to SharePoint...")
        save_raw_file(
            df         = df_live,
            token      = TOKEN,
            sp_site_id = SP_SITE_ID,
            start_date = config.START_DATE,
            start_time = config.START_TIME,
            end_date   = config.END_DATE,
            end_time   = config.END_TIME
        )

        # ── Step 7: Build Recon ───────────────────────────────────────────────
        log("Step 7 : Building recon file...")
        df_recon = build_recon_df(df_live)

        # ── Step 8: Save Recon File ───────────────────────────────────────────
        log("Step 8 : Saving recon file to SharePoint...")
        save_recon_file(
            df_recon   = df_recon,
            token      = TOKEN,
            sp_site_id = SP_SITE_ID,
            start_date = config.START_DATE,
            start_time = config.START_TIME,
            end_date   = config.END_DATE,
            end_time   = config.END_TIME
        )

        log("=" * 60)
        log("PIPELINE COMPLETE")
        log("=" * 60)

    except Exception as e:
        log(f"PIPELINE FAILED : {str(e)}", "ERROR")
        log(traceback.format_exc(), "ERROR")

    finally:
        # ── Always save log to SharePoint ─────────────────────────────────────
        if TOKEN and SP_SITE_ID:
            log("Saving log to SharePoint...")
            save_log_to_sharepoint(TOKEN, SP_SITE_ID, LOG_FILE_NAME)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
