import pandas as pd
import re
from difflib import SequenceMatcher
from collections import Counter

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FILE        = "input.csv"
OUTPUT_FILLED     = "output_filled.csv"
OUTPUT_REVIEW     = "output_review_log.csv"

# Column name aliases — adjust if your CSV headers differ
COL_ITEM     = "Item"
COL_I3       = "i3_Standard_item"
COL_PAYABLE  = "Payable (P)/Not-Payable(NP)"   # or "Payable" etc.

FUZZY_THRESHOLD   = 0.55   # min similarity score to accept a match (0–1)
# ─────────────────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set:
    return set(normalize(text).split())


def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def token_overlap_score(item_blank: str, item_mapped: str) -> float:
    """
    Score based on how many tokens of the blank item appear in the mapped item
    and vice versa (Jaccard-like, but weighted toward the shorter string).
    """
    t_blank  = tokenize(item_blank)
    t_mapped = tokenize(item_mapped)
    if not t_blank or not t_mapped:
        return 0.0
    intersection = t_blank & t_mapped
    # weight: intersection / min-length tokens (favour short blank items matching longer mapped ones)
    return len(intersection) / min(len(t_blank), len(t_mapped))


def combined_score(item_blank: str, item_mapped: str) -> float:
    tok = token_overlap_score(item_blank, item_mapped)
    fuz = fuzzy_score(item_blank, item_mapped)
    # token overlap matters more for partial-word cases like BUDS vs EAR BUDS
    return 0.65 * tok + 0.35 * fuz


def find_best_match(item_blank: str, mapped_df: pd.DataFrame):
    """
    Against all rows that already have an i3 value, find the best matching
    i3_Standard_item using a vote+score approach.
    """
    scores = []
    for _, row in mapped_df.iterrows():
        score = combined_score(item_blank, row[COL_ITEM])
        if score > 0:
            scores.append((score, row[COL_I3]))

    if not scores:
        return None, 0.0

    # Group by i3 value → sum scores (voting: multiple items pointing to same i3 wins)
    vote: dict = {}
    for score, i3_val in scores:
        vote[i3_val] = vote.get(i3_val, 0) + score

    best_i3    = max(vote, key=vote.get)
    best_score = vote[best_i3]

    # Normalise score to 0–1 range for interpretability
    max_possible = len(mapped_df)
    norm_score = min(best_score / max_possible, 1.0) if max_possible else 0.0

    # Raw best single-pair score for threshold check
    raw_best = max(s for s, i3 in scores if i3 == best_i3)
    return best_i3, raw_best


def main():
    # ── 1. Load & trim columns ───────────────────────────────────────────────
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"✔ Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # Auto-detect column names (case-insensitive partial match)
    col_map = {}
    for col in df_raw.columns:
        col_lower = col.lower().strip()
        if "item" == col_lower and "item" not in col_map:
            col_map["item"] = col
        elif "i3" in col_lower:
            col_map["i3"] = col
        elif "payable" in col_lower or "p/np" in col_lower or "p)/not" in col_lower:
            col_map["payable"] = col

    missing = [k for k in ["item", "i3", "payable"] if k not in col_map]
    if missing:
        print(f"\n⚠ Could not auto-detect columns: {missing}")
        print("   Available columns:", list(df_raw.columns))
        print("   Update COL_ITEM / COL_I3 / COL_PAYABLE at the top of the script.")
        return

    # Rename to standard names & keep only 3 columns
    df = df_raw[[col_map["item"], col_map["i3"], col_map["payable"]]].copy()
    df.columns = [COL_ITEM, COL_I3, COL_PAYABLE]
    print(f"✔ Kept 3 columns: {list(df.columns)}")

    # ── 2. Split mapped vs blank ─────────────────────────────────────────────
    blank_mask = df[COL_I3].isna() | (df[COL_I3].astype(str).str.strip() == "")
    df_mapped  = df[~blank_mask].copy()
    df_blank   = df[blank_mask].copy()

    print(f"\n   Rows with i3 mapped : {len(df_mapped)}")
    print(f"   Rows with i3 BLANK  : {len(df_blank)}")

    if df_blank.empty:
        print("\n✅ No blank rows found. Nothing to fill.")
        df.to_csv(OUTPUT_FILLED, index=False)
        return

    # ── 3. Fill blanks ───────────────────────────────────────────────────────
    review_log = []

    for idx, row in df_blank.iterrows():
        item_val  = str(row[COL_ITEM]).strip()
        best_i3, score = find_best_match(item_val, df_mapped)

        if best_i3 and score >= FUZZY_THRESHOLD:
            df.at[idx, COL_I3] = best_i3
            status = "AUTO_FILLED"
        else:
            best_i3 = best_i3 or "NO_MATCH"
            status  = "NEEDS_REVIEW"

        review_log.append({
            "Row"              : idx,
            COL_ITEM           : item_val,
            "Suggested_i3"     : best_i3,
            "Confidence_Score" : round(score, 3),
            "Status"           : status,
        })

    # ── 4. Summary ───────────────────────────────────────────────────────────
    review_df   = pd.DataFrame(review_log)
    filled_cnt  = (review_df["Status"] == "AUTO_FILLED").sum()
    review_cnt  = (review_df["Status"] == "NEEDS_REVIEW").sum()

    print(f"\n{'─'*45}")
    print(f"  AUTO_FILLED   : {filled_cnt}")
    print(f"  NEEDS_REVIEW  : {review_cnt}")
    print(f"{'─'*45}")

    if review_cnt:
        print("\n  ⚠ Rows needing manual review:")
        print(review_df[review_df["Status"] == "NEEDS_REVIEW"][[COL_ITEM, "Suggested_i3", "Confidence_Score"]].to_string(index=False))

    # ── 5. Save outputs ──────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILLED, index=False)
    review_df.to_csv(OUTPUT_REVIEW, index=False)

    print(f"\n✅ Saved: {OUTPUT_FILLED}")
    print(f"✅ Saved: {OUTPUT_REVIEW}  ← check this for manual review rows")


if __name__ == "__main__":
    main()
