def score_pair(blank_norm, blank_tok, blank_comp,
               mapped_norm, mapped_tok, mapped_comp) -> float:

    shared = blank_tok & mapped_tok

    # ── Path 1: Word token overlap ────────────────────────────────────────────
    if shared:
        coverage  = len(shared) / len(blank_tok)  if blank_tok  else 0.0
        precision = len(shared) / len(mapped_tok) if mapped_tok else 0.0
        f1 = (2 * coverage * precision / (coverage + precision)
              if (coverage + precision) > 0 else 0.0)
        substring_bonus = 0.25 if blank_norm in mapped_norm else 0.0
        fuzzy = SequenceMatcher(None, blank_norm, mapped_norm).ratio()
        return min(1.0, 0.50 * f1
                       + 0.25 * substring_bonus
                       + 0.15 * coverage
                       + 0.10 * fuzzy)

    # ── Path 2: Compound matching ─────────────────────────────────────────────
    b, m    = blank_comp, mapped_comp
    comp_sim = SequenceMatcher(None, b, m).ratio()

    # Prefix match: "armslingpouch".startswith("armsling")
    # Min length 5 avoids noise from short tokens like "arm", "bed"
    prefix_match = (
        (b.startswith(m) or m.startswith(b))
        and min(len(b), len(m)) >= 5
    )

    if prefix_match:
        # Strong signal: mapped item is the "base" of the blank item
        # e.g. armsling → armslingpouch / armslingpouchmedium
        return min(1.0, 0.70 + 0.30 * comp_sim)

    if comp_sim >= 0.80:
        # High similarity but not prefix (e.g. armsling ↔ armslings)
        return comp_sim * 0.85

    # ── Path 3: Pure fuzzy fallback ───────────────────────────────────────────
    return SequenceMatcher(None, blank_norm, mapped_norm).ratio() * 0.40
