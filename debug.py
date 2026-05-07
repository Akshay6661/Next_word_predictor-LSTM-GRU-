# ── PPM wrong breakdown ───────────────────────────────────────────────────────
df_ppm_wrong = df_5class[
    (df_5class["actual_class"]    == "PPM Request") &
    (df_5class["predicted_class"] != "PPM Request")
].copy()

print(f"PPM wrong total : {len(df_ppm_wrong)}")
print(f"\n── PPM misclassified as ─────────────────────────────────────")
print(df_ppm_wrong["predicted_class"].value_counts())

# ── What words are in wrong PPM emails ───────────────────────────────────────
from collections import Counter
import re

all_text = " ".join(
    (df_ppm_wrong["subject"].fillna("") + " " + 
     df_ppm_wrong["pure_body"].fillna("")).tolist()
).lower()

words     = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
words     = [w for w in words if w not in stop_words]
counter   = Counter(words)

print(f"\n── Top words in wrong PPM emails ────────────────────────────")
print(f"   {'Word':<25} {'Count':>6}")
print(f"   {'─'*35}")
for word, count in counter.most_common(30):
    in_trigger = word in PPM_STRONG_TRIGGER or word in PPM_WEAK_TRIGGER
    flag       = "✅ already" if in_trigger else "⬅️  missing"
    print(f"   {word:<25} {count:>6}  {flag}")

# ── Sample wrong PPM bodies ───────────────────────────────────────────────────
print(f"\n── Sample wrong PPM bodies ──────────────────────────────────")
for i, row in df_ppm_wrong.head(5).iterrows():
    print(f"\nPredicted as : {row['predicted_class']}")
    print(f"Body         : {row['pure_body'][:400]}")
    print("─" * 60)


## next code block

# ── CQA wrong breakdown ───────────────────────────────────────────────────────
df_cqa_wrong = df_5class[
    (df_5class["actual_class"]    == "CQA Acknowledgement") &
    (df_5class["predicted_class"] != "CQA Acknowledgement")
].copy()

print(f"\nCQA wrong total : {len(df_cqa_wrong)}")
print(f"\n── CQA misclassified as ─────────────────────────────────────")
print(df_cqa_wrong["predicted_class"].value_counts())

# ── What words are in wrong CQA emails ───────────────────────────────────────
all_text = " ".join(
    (df_cqa_wrong["subject"].fillna("") + " " + 
     df_cqa_wrong["pure_body"].fillna("")).tolist()
).lower()

words   = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
words   = [w for w in words if w not in stop_words]
counter = Counter(words)

print(f"\n── Top words in wrong CQA emails ────────────────────────────")
print(f"   {'Word':<25} {'Count':>6}  {'% of wrong CQA':>15}")
print(f"   {'─'*50}")
for word, count in counter.most_common(30):
    pct        = round(count / len(df_cqa_wrong) * 100, 1)
    in_trigger = word in CQA_REQUIRED_WORDS or word in CQA_INVESTIGATE_WORDS or word in CQA_DEVICE_WORDS
    flag       = "✅ already" if in_trigger else "⬅️  missing"
    print(f"   {word:<25} {count:>6}  {pct:>14}%  {flag}")

# ── Sample wrong CQA bodies ───────────────────────────────────────────────────
print(f"\n── Sample wrong CQA bodies ──────────────────────────────────")
for i, row in df_cqa_wrong.head(5).iterrows():
    print(f"\nPredicted as : {row['predicted_class']}")
    print(f"Body         : {row['pure_body'][:400]}")
    print("─" * 60)




# =============================================================================
# CELL 11 — EMAIL RECON FILE WITH FORMATTING
# =============================================================================

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from io import BytesIO

# ── Build recon dataframe ─────────────────────────────────────────────────────
df_recon = pd.DataFrame()

df_recon["From"]                 = df_live["sender_name"]
df_recon["Subject"]              = df_live["subject"]
df_recon["Received Time"]        = df_live["time"]
df_recon["Received Date"]        = pd.to_datetime(df_live["date"]).dt.strftime("%d-%b-%y")
df_recon["Owner"]                = ""
df_recon["Action"]               = ""
df_recon["Status"]               = ""
df_recon["pure_body"]            = df_live["pure_body"]
df_recon["Comments"]             = df_live["predicted_class"]
df_recon["Checked by"]           = ""
df_recon["TAT status"]           = ""
df_recon["Communication status"] = ""
df_recon["case_number"]          = df_live["case_number"]

# ── Sort by date and time ─────────────────────────────────────────────────────
df_recon["sort_dt"] = pd.to_datetime(
                        df_live["date"].astype(str) + " " +
                        df_live["time"].str.extract(r"(\d{2}:\d{2} [AP]M)")[0],
                        format="%Y-%m-%d %I:%M %p",
                        errors="coerce"
                      )
df_recon = df_recon.sort_values("sort_dt", ascending=True).reset_index(drop=True)
df_recon = df_recon.drop(columns=["sort_dt"])

# ── Save to Excel with formatting ─────────────────────────────────────────────
recon_file = f"email_recon_{START_DATE}_{START_TIME.replace(':','')}_to_{END_DATE}_{END_TIME.replace(':','')}_IST.xlsx"

# Write dataframe first
df_recon.to_excel(recon_file, index=False)

# ── Apply formatting using openpyxl ──────────────────────────────────────────
wb = load_workbook(recon_file)
ws = wb.active

# Header fill color #f8cbad
header_fill   = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")
header_font   = Font(bold=True, color="000000")
header_align  = Alignment(horizontal="center", vertical="center", wrap_text=True)
thin_border   = Border(
    left   = Side(style="thin"),
    right  = Side(style="thin"),
    top    = Side(style="thin"),
    bottom = Side(style="thin")
)

# ── Format header row ─────────────────────────────────────────────────────────
for cell in ws[1]:
    cell.fill      = header_fill
    cell.font      = header_font
    cell.alignment = header_align
    cell.border    = thin_border

# ── Auto fit column widths ────────────────────────────────────────────────────
for col in ws.columns:
    max_length = 0
    col_letter = col[0].column_letter
    for cell in col:
        try:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        except:
            pass
    adjusted_width = min(max_length + 4, 40)   # cap at 40
    ws.column_dimensions[col_letter].width = adjusted_width

# ── Freeze top row ────────────────────────────────────────────────────────────
ws.freeze_panes = "A2"

# ── Save formatted workbook ───────────────────────────────────────────────────
wb.save(recon_file)

print(f"✅ Email recon saved    : {recon_file}")
print(f"   Rows                : {len(df_recon)}")
print(f"   Header color        : #F8CBAD ✅")
print(f"   Columns auto-fitted : ✅")
print(f"   Top row frozen      : ✅")
print(f"\n── Comments distribution ────────────────────────────────────")
print(df_recon["Comments"].value_counts())
