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
    all_ppm_hits      = ppm_strong_hits + ppm_weak_hits

    # ✅ Device needs min 2 specific words
    cqa_device_hits   = [w for w in CQA_DEVICE_WORDS       if w in words]
    has_cqa_device    = len(cqa_device_hits) >= CQA_DEVICE_MIN_MATCHES

    has_acknowledge   = any(w in words for w in DSD_TRIGGER)
    has_cqa_required  = all(w in words for w in CQA_REQUIRED_WORDS)
    has_cqa_invest    = any(w in words for w in CQA_INVESTIGATE_WORDS)
    has_cqa_phrase    = any(phrase in combined for phrase in CQA_PHRASES)
    has_batch_invest  = ("batch" in words or "preliminary" in words or
                         "discrepancy" in words or "analytical" in words)

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

    # ── Rule 3: CQA Device — needs 2+ specific words ──────────────────────────
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
    if dsd_hits:
        return pd.Series({
            "predicted_class" : "DSD Acknowledgement",
            "confidence"      : 0.97,
            "rule_triggered"  : "dsd_trigger",
            "matched_keywords": str(dsd_hits)
        })

    # ── Rule 8: For Follow Up ─────────────────────────────────────────────────
    if len(followup_hits) >= FOLLOWUP_MIN_MATCHES:
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

print("✅ classify_email updated — device trigger now needs 2+ words")
