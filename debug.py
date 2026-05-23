import pandas as pd
import re
from difflib import SequenceMatcher
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE        = "input.csv"
OUTPUT_FILLED     = "output_filled.csv"
OUTPUT_REVIEW     = "output_review_log.csv"
CONFIDENCE_ACCEPT = 0.50
CONFIDENCE_FLAG   = 0.30
MIN_TOKEN_LEN     = 2
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> set:
    return {t for t in normalize(text).split() if len(t) >= MIN_TOKEN_LEN}

def compound(norm: str) -> str:
    return norm.replace(" ", "")

def score_pair(blank_norm, blank_tok, blank_comp,
               mapped_norm, mapped_tok, mapped_comp) -> float:
    shared = blank_tok & mapped_tok

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

    comp_score = SequenceMatcher(None, blank_comp, mapped_comp).ratio()
    if comp_score >= 0.80:
        return comp_score * 0.85

    return SequenceMatcher(None, blank_norm, mapped_norm).ratio() * 0.40


def main():
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"✔  Loaded : {len(df_raw):,} rows | {len(df_raw.columns)} columns")

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
        print(f"⚠  Could not detect: {missing}")
        print("   Found:", list(df_raw.columns))
        return

    df = df_raw[[col_map["item"], col_map["i3"], col_map["payable"]]].copy()
    df.columns = ["Item", "i3_Standard_item", "Payable"]

    blank_mask = (df["i3_Standard_item"].isna() |
                  df["i3_Standard_item"].astype(str).str.strip().eq(""))
    df_mapped = df[~blank_mask].reset_index(drop=True)
    df_blank  = df[blank_mask].copy()

    print(f"   Mapped : {len(df_mapped):,} | Blank : {len(df_blank):,}")

    if df_blank.empty:
        print("✅ No blanks.")
        df.to_csv(OUTPUT_FILLED, index=False)
        return

    print("⚙  Building indexes...")

    mapped_norms  = [normalize(x) for x in df_mapped["Item"]]
    mapped_tokens = [tokenize(x)  for x in df_mapped["Item"]]
    mapped_comps  = [compound(n)  for n in mapped_norms]
    mapped_i3     = df_mapped["i3_Standard_item"].tolist()

    token_index = defaultdict(list)
    for i, toks in enumerate(mapped_tokens):
        for tok in toks:
            token_index[tok].append(i)

    compound_index = defaultdict(list)
    for i, comp in enumerate(mapped_comps):
        compound_index[comp].append(i)

    print(f"✔  Token index: {len(token_index):,} | Compound index: {len(compound_index):,}")
    print("⚙  Matching blank rows...\n")

    review_log = []

    for idx, row in df_blank.iterrows():
        item_val   = str(row["Item"]).strip()
        norm_blank = normalize(item_val)
        toks_blank = tokenize(item_val)
        comp_blank = compound(norm_blank)

        # ── Candidate lookup ─────────────────────────────────────────────────
        candidates = set()
        for tok in toks_blank:
            candidates.update(token_index.get(tok, []))
        candidates.update(compound_index.get(comp_blank, []))
        for mc, idxs in compound_index.items():
            if comp_blank and mc and (comp_blank in mc or mc in comp_blank):
                candidates.update(idxs)

        # ── Score ─────────────────────────────────────────────────────────────
        vote        = defaultdict(float)   # i3 → cumulative (tiebreaker only)
        best_single = defaultdict(float)   # i3 → best single-pair score ← KEY

        for ci in candidates:
            s      = score_pair(norm_blank, toks_blank, comp_blank,
                                mapped_norms[ci], mapped_tokens[ci], mapped_comps[ci])
            i3_val = mapped_i3[ci]
            vote[i3_val]        += s
            best_single[i3_val]  = max(best_single[i3_val], s)

        # ── Winner: best precision, vote as tiebreaker ────────────────────────
        if best_single:
            max_score     = max(best_single.values())
            top_candidates = [i3 for i3, s in best_single.items()
                              if s >= max_score - 0.001]
            best_i3   = max(top_candidates, key=lambda i3: vote[i3])
            raw_score = max_score
            n_cands   = len(candidates)
        else:
            best_i3, raw_score, n_cands = "NO_MATCH", 0.0, 0

        # ── Status ────────────────────────────────────────────────────────────
        if raw_score >= CONFIDENCE_ACCEPT:
            df.at[idx, "i3_Standard_item"] = best_i3
            status = "AUTO_FILLED"
        elif raw_score >= CONFIDENCE_FLAG:
            df.at[idx, "i3_Standard_item"] = best_i3
            status = "FILLED_REVIEW"
        else:
            status = "NO_MATCH"

        review_log.append({
            "Row"          : idx,
            "Item"         : item_val,
            "Suggested_i3" : best_i3,
            "Confidence"   : round(raw_score, 3),
            "Candidates"   : n_cands,
            "Status"       : status,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    review_df = pd.DataFrame(review_log)
    counts    = review_df["Status"].value_counts()

    print(f"{'─'*52}")
    print(f"  AUTO_FILLED   (confident)   : {counts.get('AUTO_FILLED',   0):>5}")
    print(f"  FILLED_REVIEW (verify)      : {counts.get('FILLED_REVIEW', 0):>5}")
    print(f"  NO_MATCH      (manual)      : {counts.get('NO_MATCH',      0):>5}")
    print(f"{'─'*52}")

    needs_attention = review_df[review_df["Status"].isin(["FILLED_REVIEW", "NO_MATCH"])]
    if not needs_attention.empty:
        print("\n⚠  Rows needing review:")
        print(needs_attention[["Item", "Suggested_i3", "Confidence", "Status"]]
              .to_string(index=False))

    df.to_csv(OUTPUT_FILLED, index=False)
    review_df.to_csv(OUTPUT_REVIEW, index=False)
    print(f"\n✅  {OUTPUT_FILLED}")
    print(f"✅  {OUTPUT_REVIEW}")


if __name__ == "__main__":
    main()
