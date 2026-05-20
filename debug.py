# ── From your browser URL ─────────────────────────────────────────────────────
HOSTNAME  = "yourcompany.sharepoint.com"   # ← from browser URL
SITE_NAME = "FinanceTeam"                  # ← from browser URL /sites/SiteName

resp = requests.get(
    f"https://graph.microsoft.com/v1.0/sites/{HOSTNAME}:/sites/{SITE_NAME}",
    headers=HEADERS, verify=False
)

SP_SITE_ID = resp.json()["id"]
print(f"✅ Site Name : {resp.json()['displayName']}")
print(f"✅ Site ID   : {SP_SITE_ID}")



resp = requests.get(
    f"https://graph.microsoft.com/v1.0/sites/{SP_SITE_ID}/drives",
    headers=HEADERS, verify=False
)

print("── Available Drives ─────────────────────────────────────")
for drive in resp.json().get("value", []):
    print(f"   Drive ID : {drive['id']}")
    print(f"   Name     : {drive['name']}")
    print("─" * 50)



SP_DRIVE_ID = "your-drive-id"   # ← paste your drive id here

# ── List root folders ─────────────────────────────────────────────────────────
resp = requests.get(
    f"https://graph.microsoft.com/v1.0/sites/{SP_SITE_ID}/drives/{SP_DRIVE_ID}/root/children",
    headers=HEADERS, verify=False
)

print("── Folders at Root ──────────────────────────────────────")
for item in resp.json().get("value", []):
    item_type = "📁" if "folder" in item else "📄"
    print(f"   {item_type} {item['name']}")






# ── Verify daily_landing_zone path ────────────────────────────────────────────
recon_path = "/Analytics and Dashboards/daily_cipla_email/daily_landing_zone"

resp = requests.get(
    f"https://graph.microsoft.com/v1.0/sites/{SP_SITE_ID}/drives/{SP_DRIVE_ID}/root:{recon_path}",
    headers=HEADERS, verify=False
)

if resp.status_code == 200:
    print(f"✅ Recon folder found : {resp.json()['name']}")
else:
    print(f"❌ Recon folder not found : {resp.status_code}")
    print(resp.json())

# ── Verify raw_file path ──────────────────────────────────────────────────────
raw_path = "/Analytics and Dashboards/daily_cipla_email/raw_file"

resp = requests.get(
    f"https://graph.microsoft.com/v1.0/sites/{SP_SITE_ID}/drives/{SP_DRIVE_ID}/root:{raw_path}",
    headers=HEADERS, verify=False
)

if resp.status_code == 200:
    print(f"✅ Raw folder found   : {resp.json()['name']}")
else:
    print(f"❌ Raw folder not found : {resp.status_code}")
    print(resp.json())



#puling
from datetime import datetime, timedelta
import pytz

# ── Auto calculate IST date range ─────────────────────────────────────────────
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)

# ── End   = Today      09:29:59 IST ──────────────────────────────────────────
# ── Start = Yesterday  09:30:00 IST ──────────────────────────────────────────
end_dt   = now.replace(hour=9, minute=29, second=59, microsecond=0)
start_dt = end_dt.replace(hour=9, minute=30, second=0) - timedelta(days=1)

START_DATE = start_dt.strftime("%Y-%m-%d")
START_TIME = start_dt.strftime("%H:%M:%S")   # 09:30:00
END_DATE   = end_dt.strftime("%Y-%m-%d")
END_TIME   = end_dt.strftime("%H:%M:%S")     # 09:29:59

print(f"✅ Start : {START_DATE} {START_TIME} IST")
print(f"✅ End   : {END_DATE}   {END_TIME}   IST")
