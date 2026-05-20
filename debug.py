# =============================================================================
# LOGGER.PY — Logging setup
# Logs to console + in-memory buffer → saved to SharePoint at end
# =============================================================================

import requests
import urllib3
import pytz
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── In-memory log buffer ──────────────────────────────────────────────────────
log_lines = []
_IST      = pytz.timezone("Asia/Kolkata")

def log(msg, level="INFO"):
    """Log to console and in-memory buffer"""
    timestamp = datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S IST")
    line      = f"[{timestamp}] [{level}] {msg}"
    print(line)
    log_lines.append(line)

def save_log_to_sharepoint(token, sp_site_id, sp_drive_id, sp_logs_folder, log_file_name):
    """Upload accumulated log buffer to SharePoint logs folder"""
    try:
        log_content    = "\n".join(log_lines).encode("utf-8")
        url            = (
            f"https://graph.microsoft.com/v1.0/sites/{sp_site_id}"
            f"/drives/{sp_drive_id}/root:{sp_logs_folder}/{log_file_name}:/content"
        )
        headers_upload = {
            "Authorization": f"Bearer {token}",
            "Content-Type" : "text/plain"
        }
        resp = requests.put(url, headers=headers_upload, data=log_content, verify=False)
        if resp.status_code in [200, 201]:
            print(f"[LOG] Saved : {sp_logs_folder}/{log_file_name}")
        else:
            print(f"[LOG] Save failed : {resp.status_code}")
    except Exception as e:
        print(f"[LOG] Upload error : {str(e)}")
