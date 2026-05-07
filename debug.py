# =============================================================================
# TRIGGER LISTS — ALL CLASSES
# =============================================================================

# ── CQA Acknowledgement ───────────────────────────────────────────────────────
CQA_REQUIRED_WORDS    = ["receipt", "complaint"]
CQA_INVESTIGATE_WORDS = ["revert", "findings", "investigate", "investigation"]
CQA_DEVICE_WORDS      = ["inhaler", "troubleshooting", "logging"]
CQA_DEVICE_MIN_MATCHES = 2

CQA_PHRASES = [
    "acknowledge the receipt the below complaint",
    "acknowledge receipt below complaint",
    "acknowledge the receipt below complaint",
    "receipt of the below complaint",
    "receipt the below complaint",
]

# ── PPM Request ───────────────────────────────────────────────────────────────
PPM_STRONG_TRIGGER = ["prepaid", "mailer", "ppm"]
PPM_WEAK_TRIGGER   = ["initiated", "investigated"]
PPM_MIN_MATCHES    = 2

# ── Argus ID ──────────────────────────────────────────────────────────────────
ARGUS_TRIGGER = ["argus"]

# ── DSD Acknowledgement ───────────────────────────────────────────────────────
DSD_TRIGGER = [
    "acknowledge", "acknowledged",
    "acknowledgement", "acknowledgment",
]

# ── For Follow Up ─────────────────────────────────────────────────────────────
FOLLOWUP_UNIQUE_WORDS = [
    "investigation", "batch",       "sample",      "kindly",
    "team",          "observed",    "provide",     "patient",
    "information",   "discrepancy", "found",       "were",
    "preliminary",   "records",     "defect",      "analytical",
    "review",        "complaint",   "reported",    "adverse",
    "event",         "outcome",     "follow",      "update",
    "status",        "pending",     "resolution",  "closure",
]
FOLLOWUP_MIN_MATCHES = 3

# ── Overlap words ─────────────────────────────────────────────────────────────
OVERLAP_WORDS = [
    "colleague", "below", "find",
    "case",      "greetings", "receipt",
]

print("✅ Trigger lists updated")
print(f"   Argus triggers        : {len(ARGUS_TRIGGER)}")
print(f"   CQA required          : {len(CQA_REQUIRED_WORDS)}")
print(f"   CQA investigate       : {len(CQA_INVESTIGATE_WORDS)}")
print(f"   CQA device            : {len(CQA_DEVICE_WORDS)} (min {CQA_DEVICE_MIN_MATCHES})")
print(f"   CQA phrases           : {len(CQA_PHRASES)}")
print(f"   PPM strong            : {len(PPM_STRONG_TRIGGER)}")
print(f"   PPM weak              : {len(PPM_WEAK_TRIGGER)} (min {PPM_MIN_MATCHES})")
print(f"   DSD triggers          : {len(DSD_TRIGGER)}")
print(f"   Follow Up triggers    : {len(FOLLOWUP_UNIQUE_WORDS)} (min {FOLLOWUP_MIN_MATCHES})")
print(f"   Overlap words         : {len(OVERLAP_WORDS)}")



def classify_email(row):

    subject  = str(row["subject"]).lower()   if pd.notna(row["subject"])   else ""
    body     = str(row["pure_body"]).lower() if pd.notna(row["pure_body"]) else ""
    combined = f"{subject} {body}"
    words    = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined))

    # ── Pre-compute all hits ───────────────────────────────────────────────────
    argus_hits        = [w for w in ARGUS_TRIGGER          if w in words]
    ppm_strong_hits   = [w for w in PPM_STRONG_TRIGGER     if w in words]
    ppm_weak_hits     = [w for w in PPM_WEAK_TRIGGER       if w in words]
    dsd_hits          = [w for w in DSD_TRIGGER            if w in words]
    followup_hits     = [w for w in FOLLOWUP_UNIQUE_WORDS  if w in words]
    overlap_hits      = [w for w in OVERLAP_WORDS          if w in words]
    cqa_device_hits   = [w for w in CQA_DEVICE_WORDS       if w in words]
    all_ppm_hits      = ppm_strong_hits + ppm_weak_hits

    has_cqa_device    = len(cqa_device_hits) >= CQA_DEVICE_MIN_MATCHES
    has_acknowledge   = any(w in words for w in DSD_TRIGGER)
    has_cqa_required  = all(w in words for w in CQA_REQUIRED_WORDS)
    has_cqa_invest    = any(w in words for w in CQA_INVESTIGATE_WORDS)
    has_cqa_phrase    = any(phrase in combined for phrase in CQA_PHRASES)
    has_batch_invest  = ("batch" in words or "preliminary" in words or
                         "discrepancy" in words or "analytical" in words)

    # ── Computed signals ──────────────────────────────────────────────────────
    has_strong_followup = len(followup_hits) >= 3
    has_ppm_signal      = len(ppm_weak_hits) >= 1

    # ── Rule 1: Argus ID ──────────────────────────────────────────────────────
    if argus_hits:
        return pd.Series({
            "predicted_class" : "Argus ID",
            "confidence"      : 0.97,
            "rule_triggered"  : "argus_trigger",
            "matched_keywords": str(argus_hits)
        })

    # ── Rule 2: PPM Strong ────────────────────────────────────────────────────
    if ppm_strong_hits:
        confidence = min(0.60 + (len(all_ppm_hits) * 0.10), 0.99)
        return pd.Series({
            "predicted_class" : "PPM Request",
            "confidence"      : round(confidence, 2),
            "rule_triggered"  : f"ppm_strong_{len(ppm_strong_hits)}_matched",
            "matched_keywords": str(all_ppm_hits)
        })

    # ── Rule 3: CQA Device ────────────────────────────────────────────────────
    if has_cqa_device:
        return pd.Series({
            "predicted_class" : "CQA Acknowledgement",
            "confidence"      : 0.95,
            "rule_triggered"  : "cqa_device_trigger",
            "matched_keywords": str(cqa_device_hits)
        })

    # ── Rule 4: CQA Phrase — guarded ─────────────────────────────────────────
    if has_cqa_phrase and not has_strong_followup and not has_ppm_signal:
        return pd.Series({
            "predicted_class" : "CQA Acknowledgement",
            "confidence"      : 0.97,
            "rule_triggered"  : "cqa_phrase_trigger",
            "matched_keywords": str([p for p in CQA_PHRASES if p in combined])
        })

    # ── Rule 5: CQA Investigation ─────────────────────────────────────────────
    if has_cqa_required and has_cqa_invest and not has_batch_invest:
        return pd.Series({
            "predicted_class" : "CQA Acknowledgement",
            "confidence"      : 0.97,
            "rule_triggered"  : "cqa_invest_trigger",
            "matched_keywords": str(
                [w for w in CQA_REQUIRED_WORDS    if w in words] +
                [w for w in CQA_INVESTIGATE_WORDS if w in words]
            )
        })

    # ── Rule 6: PPM Weak ──────────────────────────────────────────────────────
    if len(ppm_weak_hits) >= PPM_MIN_MATCHES and not has_cqa_required:
        confidence = min(0.50 + (len(ppm_weak_hits) * 0.10), 0.99)
        return pd.Series({
            "predicted_class" : "PPM Request",
            "confidence"      : round(confidence, 2),
            "rule_triggered"  : f"ppm_weak_{len(ppm_weak_hits)}_matched",
            "matched_keywords": str(ppm_weak_hits)
        })

    # ── Rule 7: DSD — guarded against Follow Up ───────────────────────────────
    if dsd_hits and len(followup_hits) < 3:
        return pd.Series({
            "predicted_class" : "DSD Acknowledgement",
            "confidence"      : 0.97,
            "rule_triggered"  : "dsd_trigger",
            "matched_keywords": str(dsd_hits)
        })

    # ── Rule 8: For Follow Up — min 3 matches ────────────────────────────────
    if len(followup_hits) >= FOLLOWUP_MIN_MATCHES:
        confidence = min(0.50 + (len(followup_hits) * 0.10), 0.99)
        return pd.Series({
            "predicted_class" : "For Follow Up",
            "confidence"      : round(confidence, 2),
            "rule_triggered"  : f"followup_{len(followup_hits)}_words_matched",
            "matched_keywords": str(followup_hits)
        })

    # ── Rule 9: Weak Follow Up ────────────────────────────────────────────────
    if len(followup_hits) >= 1 and len(overlap_hits) >= 2:
        return pd.Series({
            "predicted_class" : "For Follow Up",
            "confidence"      : 0.45,
            "rule_triggered"  : "followup_weak_signal",
            "matched_keywords": str(followup_hits + overlap_hits)
        })

    # ── Rule 10: Unclassified ─────────────────────────────────────────────────
    return pd.Series({
        "predicted_class" : "Unclassified",
        "confidence"      : 0.0,
        "rule_triggered"  : "no_match",
        "matched_keywords": "[]"
    })

print("✅ classify_email updated — 10 rules with guards")



### debug fu with unique words 

# ── Follow Up emails going to Unclassified ────────────────────────────────────
df_fu_unclassified = df_5class[
    (df_5class["actual_class"]    == "For Follow Up") &
    (df_5class["predicted_class"] == "Unclassified")
].copy()

print(f"Follow Up → Unclassified : {len(df_fu_unclassified)}")

# ── Top words in these emails ─────────────────────────────────────────────────
all_text = " ".join(
    (df_fu_unclassified["subject"].fillna("") + " " +
     df_fu_unclassified["pure_body"].fillna("")).tolist()
).lower()

words   = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
words   = [w for w in words if w not in stop_words]
counter = Counter(words)

# ── Check against DSD emails to find safe words ───────────────────────────────
dsd_text = " ".join(
    (df_5class[df_5class["actual_class"] == "DSD Acknowledgement"]["subject"].fillna("") + " " +
     df_5class[df_5class["actual_class"] == "DSD Acknowledgement"]["pure_body"].fillna("")).tolist()
).lower()

dsd_words   = re.findall(r"\b[a-zA-Z]{3,}\b", dsd_text)
dsd_counter = Counter(dsd_words)
total_dsd   = len(df_5class[df_5class["actual_class"] == "DSD Acknowledgement"])
total_fu_un = len(df_fu_unclassified)

print(f"\n── Top words in Unclassified Follow Up emails ───────────────")
print(f"   {'Word':<20} {'FU Count':>10} {'FU %':>8} {'DSD Count':>10} {'DSD %':>8} {'Safe?':>8}")
print(f"   {'─'*65}")

for word, count in counter.most_common(30):
    fu_pct  = round(count / total_fu_un * 100, 1)
    dsd_cnt = dsd_counter.get(word, 0)
    dsd_pct = round(dsd_cnt / total_dsd * 100, 1)
    in_trigger = word in FOLLOWUP_UNIQUE_WORDS
    
    # Safe if appears a lot in FU but NOT much in DSD
    safe = "✅" if dsd_pct < 20 and fu_pct > 30 else "⚠️"
    flag = "already" if in_trigger else safe
    
    print(f"   {word:<20} {count:>10} {fu_pct:>7}% {dsd_cnt:>10} {dsd_pct:>7}% {flag:>8}")

# ── Also show sample bodies ───────────────────────────────────────────────────
print(f"\n── Sample Unclassified Follow Up bodies ─────────────────────")
for i, row in df_fu_unclassified.head(5).iterrows():
    print(f"\nBody : {row['pure_body'][:300]}")
    print("─" * 60)


### updated followup basket 
FOLLOWUP_UNIQUE_WORDS = [
    # ── Original strong words ────────────────────────────────────────────────
    "investigation", "batch",      "sample",
    "observed",      "patient",    "discrepancy",
    "found",         "were",       "preliminary",
    "defect",        "analytical", "records",
    "adverse",       "reported",

    # ── Safe words from analysis ─────────────────────────────────────────────
    "team",          "kindly",     "below",
    "cipla",         "follow",     "response",
    "share",         "provide",    "medical",
    "final",         "recon",      "information",
    "reporter",      "qinecsa",    "confirm",
    "note",

    # ── From sample bodies — very FU specific ────────────────────────────────
    "cipsc",         "reconciliation",
]
FOLLOWUP_MIN_MATCHES = 3

print(f"✅ Follow Up trigger words : {len(FOLLOWUP_UNIQUE_WORDS)}")



## debug ppm req 
# ── PPM → Follow Up emails ────────────────────────────────────────────────────
df_ppm_as_fu = df_5class[
    (df_5class["actual_class"]    == "PPM Request") &
    (df_5class["predicted_class"] == "For Follow Up")
].copy()

print(f"PPM → Follow Up : {len(df_ppm_as_fu)}")

# ── Top words in PPM→FU emails ────────────────────────────────────────────────
all_text = " ".join(
    (df_ppm_as_fu["subject"].fillna("") + " " +
     df_ppm_as_fu["pure_body"].fillna("")).tolist()
).lower()

words   = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)
words   = [w for w in words if w not in stop_words]
counter = Counter(words)

# ── Check against FU emails ───────────────────────────────────────────────────
fu_text = " ".join(
    (df_5class[df_5class["actual_class"] == "For Follow Up"]["subject"].fillna("") + " " +
     df_5class[df_5class["actual_class"] == "For Follow Up"]["pure_body"].fillna("")).tolist()
).lower()

fu_words   = re.findall(r"\b[a-zA-Z]{3,}\b", fu_text)
fu_counter = Counter(fu_words)
total_fu   = len(df_5class[df_5class["actual_class"] == "For Follow Up"])
total_ppm  = len(df_ppm_as_fu)

print(f"\n── Top words in PPM→FU emails ────────────────────────────────")
print(f"   {'Word':<20} {'PPM Count':>10} {'PPM %':>8} {'FU Count':>10} {'FU %':>8} {'Add to PPM?'}")
print(f"   {'─'*70}")

for word, count in counter.most_common(25):
    ppm_pct = round(count / total_ppm * 100, 1)
    fu_cnt  = fu_counter.get(word, 0)
    fu_pct  = round(fu_cnt / total_fu * 100, 1)
    in_ppm  = word in PPM_STRONG_TRIGGER or word in PPM_WEAK_TRIGGER
    
    # Good PPM word if high in PPM→FU but relatively lower in FU
    safe = "✅ add" if ppm_pct > 30 and fu_pct < 50 and not in_ppm else "⚠️"
    flag = "already" if in_ppm else safe
    print(f"   {word:<20} {count:>10} {ppm_pct:>7}% {fu_cnt:>10} {fu_pct:>7}%  {flag}")

# ── PPM Unclassified emails ───────────────────────────────────────────────────
df_ppm_unclass = df_5class[
    (df_5class["actual_class"]    == "PPM Request") &
    (df_5class["predicted_class"] == "Unclassified")
].copy()

print(f"\n── Sample PPM→FU bodies ──────────────────────────────────────")
for i, row in df_ppm_as_fu.head(3).iterrows():
    print(f"\nBody : {row['pure_body'][:300]}")
    print("─" * 60)

print(f"\n── Sample PPM Unclassified bodies ───────────────────────────")
for i, row in df_ppm_unclass.head(3).iterrows():
    print(f"\nBody : {row['pure_body'][:300]}")
    print("─" * 60)
