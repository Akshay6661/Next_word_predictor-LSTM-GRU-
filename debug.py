import pandas as pd
import re
from difflib import SequenceMatcher
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE        = "input.csv"
OUTPUT_FILLED     = "output_filled.csv"
OUTPUT_REVIEW     = "output_review_log.csv"
CONFIDENCE_ACCEPT = 0.50   # auto-fill if score >= this
CONFIDENCE_FLAG   = 0.30   # suggest but flag for review if between this and ACCEPT
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> set:
    return set(normalize(text).split())

def score_pair(blank_norm: str, blank_tok: set,
               mapped_norm: str, mapped_tok: set) -> float:
    """
    Coverage-first scoring:
      - Coverage  : % of blank's tokens found in mapped item  (most important)
      - Precision : % of matched tokens relative to mapped item (avoids over-broad matches)
      - Substring : bonus if blank string appears inside mapped string
      - Fuzzy     : character-level similarity fallback
    """
    shared = blank_tok & mapped_tok
    if not shared:
        return 0.0

    coverage  = len(shared) / len(blank_tok)   if blank_tok   else 0.0
    precision = len(shared) / len(mapped_tok)  if mapped_tok  else 0.0

    # F1-style blend of coverage + precision
    f1 = (2 * coverage * precision / (coverage + precision)
          if (coverage + precision) > 0 else 0.0)

    # Substring bonus — e.g. "bandage crepe" found inside "bandage crepe expiry"
    substring_bonus = 0.25 if blank_norm in mapped_norm else 0.0

    # Fuzzy character similarity (helps with typos like "nametaag" vs "nametag")
    fuzzy = SequenceMatcher(None, blank_norm, mapped_norm).ratio()

    return min(1.0, 0.50 * f1
                   + 0.25 * substring_bonus
                   + 0.15 * coverage        # extra weight on coverage
                   + 0.10 * fuzzy)


def main():
    # ── 1. Load ──────────────────────────────────────────────────────────────
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"✔  Loaded  : {len(df_raw):,} rows  |  {len(df_raw.columns)} columns")

    # Auto-detect the 3 columns
    col_map = {}
    for col in df_raw.columns:
        cl = col.lower().strip()
        if cl == "item" and "item" not in col_map:
            col_map["item"] = col
        elif "i3" in cl:
            col_map["i3"] = col
        elif "payable" in cl or "p/np" in cl or "p)/not" in cl:
            col_map["payable"] = col

    missing = [k for k in ["item", "i3", "payable"] if k not in col_map]
    if missing:
        print(f"\n⚠  Could not auto-detect columns: {missing}")
        print("   Columns found:", list(df_raw.columns))
        print("   Please rename them to match Item / i3_Standard_item / Payable")
        return

    df = df_raw[[col_map["item"], col_map["i3"], col_map["payable"]]].copy()
    df.columns = ["Item", "i3_Standard_item", "Payable"]
    print(f"✔  Kept 3 columns: {list(df.columns)}")

    # ── 2. Split mapped vs blank ──────────────────────────────────────────────
    blank_mask = df["i3_Standard_item"].isna() | \
                 (df["i3_Standard_item"].astype(str).str.strip() == "")

    df_mapped = df[~blank_mask].reset_index(drop=True)
    df_blank  = df[blank_mask].copy()

    print(f"\n   Mapped rows : {len(df_mapped):,}")
    print(f"   Blank  rows : {len(df_blank):,}")

    if df_blank.empty:
        print("\n✅ No blanks found.")
        df.to_csv(OUTPUT_FILLED, index=False)
        return

    # ── 3. Pre-compute everything for mapped rows (once) ─────────────────────
    print("\n⚙  Building inverted token index...")

    mapped_items  = df_mapped["Item"].tolist()
    mapped_i3     = df_mapped["i3_Standard_item"].tolist()
    mapped_norms  = [normalize(x) for x in mapped_items]
    mapped_tokens = [tokenize(x) for x in mapped_items]

    # Inverted index: token → [row indices in mapped]
    token_index: dict = defaultdict(list)
    for i, toks in enumerate(mapped_tokens):
        for tok in toks:
            token_index[tok].append(i)

    print(f"✔  Index built  :  {len(token_index):,} unique tokens")

    # ── 4. Match blanks ───────────────────────────────────────────────────────
    print("⚙  Matching blank rows...\n")

    review_log = []

    for idx, row in df_blank.iterrows():
        item_val   = str(row["Item"]).strip()
        norm_blank = normalize(item_val)
        toks_blank = tokenize(item_val)

        # --- Candidate lookup via token index ---
        candidate_idx = set()
        for tok in toks_blank:
            candidate_idx.update(token_index.get(tok, []))

        # --- Score each candidate ---
        vote: dict = defaultdict(float)           # i3_val → cumulative score
        best_single: dict = defaultdict(float)    # i3_val → best single-pair score

        for ci in candidate_idx:
            s = score_pair(norm_blank, toks_blank,
                           mapped_norms[ci], mapped_tokens[ci])
            i3_val = mapped_i3[ci]
            vote[i3_val]        += s
            best_single[i3_val]  = max(best_single[i3_val], s)

        # --- Pick winner by vote ---
        if vote:
            best_i3    = max(vote, key=vote.get)
            raw_score  = best_single[best_i3]
            candidates = len(candidate_idx)
        else:
            best_i3    = "NO_MATCH"
            raw_score  = 0.0
            candidates = 0

        # --- Decide status ---
        if raw_score >= CONFIDENCE_ACCEPT:
            df.at[idx, "i3_Standard_item"] = best_i3
            status = "AUTO_FILLED"
        elif raw_score >= CONFIDENCE_FLAG:
            df.at[idx, "i3_Standard_item"] = best_i3   # fill but flag
            status = "FILLED_REVIEW"                    # human should verify
        else:
            status = "NO_MATCH"

        review_log.append({
            "Row"              : idx,
            "Item"             : item_val,
            "Suggested_i3"     : best_i3,
            "Confidence"       : round(raw_score, 3),
            "Candidates_Found" : candidates,
            "Status"           : status,
        })

    # ── 5. Summary ────────────────────────────────────────────────────────────
    review_df = pd.DataFrame(review_log)
    counts    = review_df["Status"].value_counts()

    print(f"{'─'*48}")
    print(f"  AUTO_FILLED    (high confidence) : {counts.get('AUTO_FILLED', 0):>5}")
    print(f"  FILLED_REVIEW  (verify these)    : {counts.get('FILLED_REVIEW', 0):>5}")
    print(f"  NO_MATCH       (manual needed)   : {counts.get('NO_MATCH', 0):>5}")
    print(f"{'─'*48}")

    # Show items needing attention
    needs_attention = review_df[review_df["Status"].isin(["FILLED_REVIEW", "NO_MATCH"])]
    if not needs_attention.empty:
        print("\n⚠  Rows needing manual review:")
        print(needs_attention[["Item", "Suggested_i3", "Confidence", "Status"]]
              .to_string(index=False))

    # ── 6. Save ───────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILLED, index=False)
    review_df.to_csv(OUTPUT_REVIEW, index=False)

    print(f"\n✅  {OUTPUT_FILLED}")
    print(f"✅  {OUTPUT_REVIEW}  ← full audit trail with confidence scores")


if __name__ == "__main__":
    main()
