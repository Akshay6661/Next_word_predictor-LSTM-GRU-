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



#loading path
# ── SharePoint Folder Paths ───────────────────────────────────────────────────
SP_RECON_FOLDER = "/Analytics and Dashboards/daily_cipla_email/daily_landing_zone"
SP_RAW_FOLDER   = "/Analytics and Dashboards/daily_cipla_email/raw_file"


# =============================================================================
# CELL 10 — SAVE RAW FILE TO SHAREPOINT
# =============================================================================

from io import BytesIO

def write_to_sharepoint(df, folder_path, file_name):

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Classified Emails", index=False)
    buffer.seek(0)

    url = (
        f"https://graph.microsoft.com/v1.0/sites/{SP_SITE_ID}"
        f"/drives/{SP_DRIVE_ID}/root:{folder_path}/{file_name}:/content"
    )
    headers_upload = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }

    resp = requests.put(url, headers=headers_upload, data=buffer.read(), verify=False)

    if resp.status_code in [200, 201]:
        print(f"✅ Saved successfully : {folder_path}/{file_name}")
    else:
        print(f"❌ Failed. Status     : {resp.status_code}")
        print(resp.json())

# ── Auto generated file name ───────────────────────────────────────────────────
raw_file_name = f"classified_emails_{START_DATE}_{START_TIME.replace(':','')}_to_{END_DATE}_{END_TIME.replace(':','')}_IST.xlsx"

# ── Upload to raw_file folder ─────────────────────────────────────────────────
write_to_sharepoint(df_live, SP_RAW_FOLDER, raw_file_name)

## for recon 
# ── Auto generated recon file name ───────────────────────────────────────────
recon_file_name = f"email_recon_{START_DATE}_{START_TIME.replace(':','')}_to_{END_DATE}_{END_TIME.replace(':','')}_IST.xlsx"

# ── Save formatted recon to buffer ────────────────────────────────────────────
buffer = BytesIO()
df_recon.to_excel(buffer, index=False)
buffer.seek(0)

# ── Apply formatting ──────────────────────────────────────────────────────────
wb = load_workbook(buffer)
ws = wb.active

header_fill  = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")
header_font  = Font(bold=True, color="000000")
header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
thin_border  = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"),  bottom=Side(style="thin")
)

for cell in ws[1]:
    cell.fill      = header_fill
    cell.font      = header_font
    cell.alignment = header_align
    cell.border    = thin_border

for col in ws.columns:
    max_length = 0
    col_letter = col[0].column_letter
    for cell in col:
        try:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        except:
            pass
    ws.column_dimensions[col_letter].width = min(max_length + 4, 40)

ws.freeze_panes = "A2"

# ── Save formatted to new buffer ──────────────────────────────────────────────
formatted_buffer = BytesIO()
wb.save(formatted_buffer)
formatted_buffer.seek(0)

# ── Upload to daily_landing_zone folder ───────────────────────────────────────
url = (
    f"https://graph.microsoft.com/v1.0/sites/{SP_SITE_ID}"
    f"/drives/{SP_DRIVE_ID}/root:{SP_RECON_FOLDER}/{recon_file_name}:/content"
)
headers_upload = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
}

resp = requests.put(url, headers=headers_upload, data=formatted_buffer.read(), verify=False)

if resp.status_code in [200, 201]:
    print(f"✅ Recon saved : {SP_RECON_FOLDER}/{recon_file_name}")
else:
    print(f"❌ Failed      : {resp.status_code}")
    print(resp.json())
