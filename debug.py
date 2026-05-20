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
