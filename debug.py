# =============================================================================
# CONFIG.PY — All credentials, paths and settings
# Update this file only — no changes needed in other modules
# =============================================================================

import pytz
from datetime import datetime, timedelta

# ── Azure AD Credentials ──────────────────────────────────────────────────────
CLIENT_ID     = ""
CLIENT_SECRET = ""
TENANT_ID     = ""
USER_EMAIL    = ""
SCOPES        = ["https://graph.microsoft.com/.default"]

# ── SharePoint Site ───────────────────────────────────────────────────────────
HOSTNAME      = ""                    # e.g. yourcompany.sharepoint.com
SITE_NAME     = "TransformationTeam"  # from /sites/SiteName in browser URL
SP_DRIVE_ID   = ""                    # paste your drive id here

# ── SharePoint Folder Paths ───────────────────────────────────────────────────
SP_RAW_FOLDER   = "/Analytics and Dashboards/daily_cipla_email/raw_file"
SP_RECON_FOLDER = "/Analytics and Dashboards/daily_cipla_email/daily_landing_zone"
SP_LOGS_FOLDER  = "/Analytics and Dashboards/daily_cipla_email/logs"

# ── Domain Filter ─────────────────────────────────────────────────────────────
FILTER_DOMAINS       = [""]   # e.g. ["vendor.com", "supplier.com"]
ENABLE_DOMAIN_FILTER = True   # set False to pull all emails

# ── Auto Calculate IST Date Range ─────────────────────────────────────────────
# Runs daily: yesterday 09:30:00 IST → today 09:29:59 IST
IST      = pytz.timezone("Asia/Kolkata")
now      = datetime.now(IST)
end_dt   = now.replace(hour=9, minute=29, second=59, microsecond=0)
start_dt = end_dt.replace(hour=9, minute=30, second=0) - timedelta(days=1)

START_DATE = start_dt.strftime("%Y-%m-%d")
START_TIME = start_dt.strftime("%H:%M:%S")   # 09:30:00
END_DATE   = end_dt.strftime("%Y-%m-%d")
END_TIME   = end_dt.strftime("%H:%M:%S")     # 09:29:59
