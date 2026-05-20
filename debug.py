# =============================================================================
# CLASSIFICATION.PY — Rule Engine + Scoring
# Handles: word baskets, classify_email function (5 classes, 10 rules)
# To add new class: add trigger list + new rule in classify_email()
# =============================================================================

import re
import pandas as pd
from logger import log


# =============================================================================
# WORD BASKETS — Update here to tune classification
# =============================================================================

# ── CQA Acknowledgement ───────────────────────────────────────────────────────
CQA_REQUIRED_WORDS     = ["receipt", "complaint"]
CQA_INVESTIGATE_WORDS  = ["revert", "findings", "investigate", "investigation"]
CQA_DEVICE_WORDS       = ["inhaler", "troubleshooting", "logging"]
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
PPM_WEAK_TRIGGER   = [
    "initiated",   "investigated",
    "revert",      "investigate",
    "formoterol",  "zone",
    "code",        "ltd",
]
PPM_MIN_MATCHES    = 2

# ── Argus ID ──────────────────────────────────────────────────────────────────
ARGUS_TRIGGER = ["argus"]

# ── DSD Acknowledgement ───────────────────────────────────────────────────────
DSD_TRIGGER = [
    "acknowledge",     "acknowledged",
    "acknowledgement", "acknowledgment",
]

# ── For Follow Up ─────────────────────────────────────────────────────────────
FOLLOWUP_UNIQUE_WORDS = [
    "investigation", "batch",       "sample",
    "observed",      "patient",     "discrepancy",
    "found",         "were",        "preliminary",
    "defect",        "analytical",  "records",
    "adverse",       "reported",    "team",
    "kindly",        "below",       "cipla",
    "follow",        "response",    "share",
    "provide",       "medical",     "final",
    "recon",         "information", "reporter",
    "qinecsa",       "confirm",     "note",
    "cipsc",         "reconciliation",
]
FOLLOWUP_MIN_MATCHES = 3

# ── Overlap words — weak supporting signal ────────────────────────────────────
OVERLAP_WORDS = [
    "colleague", "below",     "find",
    "case",      "greetings", "receipt",
]


# =============================================================================
# CLASSIFY EMAIL — 5 Classes, 10 Rules
# Priority: Argus → PPM Strong → CQA Device → CQA Phrase →
#           CQA Invest → PPM Weak → DSD → Follow Up → Weak FU → Unclassified
# =============================================================================

def classify_email(row):
    """Classify a single email row based on rule engine"""

    subject  = str(row["subject"]).lower()   if pd.notna(row["subject"])   else ""
    body     = str(row["pure_body"]).lower() if pd.notna(row["pure_body"]) else ""
    combined = f"{subject} {body}"
    words    = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined))

    # ── Pre-compute all hits ───────────────────────────────────────────────────
    argus_hits       = [w for w in ARGUS_TRIGGER         if w in words]
    ppm_strong_hits  = [w for w in PPM_STRONG_TRIGGER    if w in words]
    ppm_weak_hits    = [w for w in PPM_WEAK_TRIGGER       if w in words]
    dsd_hits         = [w for w in DSD_TRIGGER            if w in words]
    followup_hits    = [w for w in FOLLOWUP_UNIQUE_WORDS  if w in words]
    overlap_hits     = [w for w in OVERLAP_WORDS          if w in words]
    cqa_device_hits  = [w for w in CQA_DEVICE_WORDS       if w in words]
    all_ppm_hits     = ppm_strong_hits + ppm_weak_hits

    has_cqa_device   = len(cqa_device_hits) >= CQA_DEVICE_MIN_MATCHES
    has_cqa_required = all(w in words for w in CQA_REQUIRED_WORDS)
    has_cqa_invest   = any(w in words for w in CQA_INVESTIGATE_WORDS)
    has_cqa_phrase   = any(phrase in combined for phrase in CQA_PHRASES)
    has_batch_invest = any(w in words for w in ["batch","preliminary","discrepancy","analytical"])

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

    # ── Rule 4: CQA Phrase ────────────────────────────────────────────────────
    if has_cqa_phrase:
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

    # ── Rule 7: DSD Acknowledgement ───────────────────────────────────────────
    if dsd_hits and len(followup_hits) < 4:
        return pd.Series({
            "predicted_class" : "DSD Acknowledgement",
            "confidence"      : 0.97,
            "rule_triggered"  : "dsd_trigger",
            "matched_keywords": str(dsd_hits)
        })

    # ── Rule 8: For Follow Up ─────────────────────────────────────────────────
    if len(followup_hits) >= FOLLOWUP_MIN_MATCHES and len(all_ppm_hits) < 3:
        confidence = min(0.50 + (len(followup_hits) * 0.10), 0.99)
        return pd.Series({
            "predicted_class" : "For Follow Up",
            "confidence"      : round(confidence, 2),
            "rule_triggered"  : f"followup_{len(followup_hits)}_words_matched",
            "matched_keywords": str(followup_hits)
        })

    # ── Rule 9: Weak Follow Up ────────────────────────────────────────────────
    if len(followup_hits) == 1 and len(overlap_hits) >= 2:
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


def run_classification(df):
    """Apply classify_email to all rows and return df with results"""
    log("Running classification...")
    df[["predicted_class", "confidence",
        "rule_triggered",  "matched_keywords"]] = df.apply(classify_email, axis=1)

    log("Classification Complete")
    for cls, cnt in df["predicted_class"].value_counts().items():
        log(f"  {cls:<25} : {cnt}")
    log(f"Total : {len(df)} | High conf (>0.7): {(df['confidence'] > 0.7).sum()} | Low conf (<0.4): {(df['confidence'] < 0.4).sum()}")
    return df
