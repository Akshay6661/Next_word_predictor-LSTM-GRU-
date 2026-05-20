SP_SITE_ID  = "your-site-id"    # you already have this
SP_DRIVE_ID = "your-drive-id"   # you already have this

# ── Confirm site is accessible ────────────────────────────────────────────────
resp = requests.get(
    f"https://graph.microsoft.com/v1.0/sites/{SP_SITE_ID}",
    headers=HEADERS, verify=False
)
print(f"✅ Site Name : {resp.json().get('displayName')}")
print(f"✅ Site ID   : {resp.json().get('id')}")



#Step 3 — List Root Folders in Drive
# ── See all folders at root ───────────────────────────────────────────────────
resp = requests.get(
    f"https://graph.microsoft.com/v1.0/sites/{SP_SITE_ID}/drives/{SP_DRIVE_ID}/root/children",
    headers=HEADERS, verify=False
)

print("── Folders at Root ──────────────────────────────────────")
for item in resp.json().get("value", []):
    item_type = "📁 Folder" if "folder" in item else "📄 File"
    print(f"   {item_type} : {item['name']}")


#Step 4 — Drill Into Your Folder
# ── Change this to your folder name from Step 3 ───────────────────────────────
PARENT_FOLDER = "Shared Documents"   # ← update this

resp = requests.get(
    f"https://graph.microsoft.com/v1.0/sites/{SP_SITE_ID}/drives/{SP_DRIVE_ID}/root:/{PARENT_FOLDER}:/children",
    headers=HEADERS, verify=False
)

print(f"── Contents of {PARENT_FOLDER} ──────────────────────────")
for item in resp.json().get("value", []):
    item_type = "📁 Folder" if "folder" in item else "📄 File"
    print(f"   {item_type} : {item['name']}")
