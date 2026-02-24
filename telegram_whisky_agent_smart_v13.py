import logging
import time
import re
import json
import uuid
from datetime import datetime, date
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from google.cloud import bigquery

# --- הגדרות ולוגים ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- נתיבים ---
TOKEN_PATH = r"C:\Users\iroyp\OneDrive\שולחן העבודה\TELEGRAM\Telegram-Autoforwarder-master\telegram_bot_token.txt"
GEMINI_KEY_PATH = r"C:\Users\iroyp\OneDrive\שולחן העבודה\TELEGRAM\Telegram-Autoforwarder-master\gemini_key_api.txt"
SERVICE_ACCOUNT_FILE = r"C:\Users\iroyp\OneDrive\שולחן העבודה\TELEGRAM\Telegram-Autoforwarder-master\hopeful-flash-478009-b7-1acfbd3ccca6.json"

PROJECT_ID = "hopeful-flash-478009-b7"
DATASET_ID = "Whisky_Collection"
TABLE_ID = "my_whisky_collection"
HISTORY_TABLE_ID = "alcohol_update"
FORECAST_TABLE_ID = "consumption_forecast"
VIEW_ID = "bottles_flavor_aroma_mapping"

TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
HISTORY_TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{HISTORY_TABLE_ID}"
FORECAST_TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{FORECAST_TABLE_ID}"
VIEW_REF = f"{PROJECT_ID}.{DATASET_ID}.{VIEW_ID}"

with open(TOKEN_PATH, "r", encoding="utf-8") as f:
    TELEGRAM_TOKEN = f.read().strip()
with open(GEMINI_KEY_PATH, "r", encoding="utf-8") as f:
    GEMINI_API_KEY = f.read().strip()

# NOTE: Gemini נשאר כ-Fallback לשיחות כלליות, אבל
# ל-Inventory + Update אנחנו עובדים דטרמיניסטית (בלי hallucinations).
ai_client = genai.Client(api_key=GEMINI_API_KEY)
bq_client = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT_FILE, project=PROJECT_ID)

# ==========================================
# מנגנון Cache (DF) + הבאת נתונים
# ==========================================
CACHE_DATA = {"df": None, "last_update": 0}

import re

STOPWORDS = set([
    "איזה","אילו","יש","לי","תן","תביא","כל","שלי","בבקשה",
    "וויסקי","בקבוקים","בקבוקי","בטעם","בטעמי","טעם","טעמים",
    "ארומה","ארומות","ריח","נוז","של","עם","בלי","ב","על","את",
])

def extract_keywords(user_text: str) -> list[str]:
    t = user_text.strip().lower()

    # נרמול מפרידים
    t = re.sub(r"[(){}\[\].!?\":;]", " ", t)
    t = t.replace(" ו", ",").replace(" עם", ",")
    t = re.sub(r"\s+", " ", t)

    # חיתוך ראשוני לפי פסיקים
    raw_parts = [p.strip() for p in t.split(",") if p.strip()]

    # טוקניזציה נוספת (כדי לתפוס “שוקולד וקפה” בלי פסיק)
    tokens = []
    for p in raw_parts:
        for w in p.split():
            if w and w not in STOPWORDS and len(w) >= 2:
                tokens.append(w)

    # dedupe שומר סדר
    Picked = []
    for x in tokens:
        if x not in Picked:
            Picked.append(x)

    return Picked

COL_ALIASES = {
    "palette": "palate",
    "orignal_volume": "original_volume",
}

def should_skip_planner(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return True
    # עדכונים/פעולות שעדיף hard-rule
    if any(x in t for x in ["שתיתי", "שתיה", "עדכן", "עדכון", "הוסף", "להוסיף", "מחיקה", "/start", "/help"]):
        return True
    return False

import re

_FLAVOR_MARKERS = ("בטעמי", "בטעם", "טעמים של", "עם טעמים של")

def enforce_palette_free_text(plan: dict, user_text: str) -> dict:
    if not isinstance(plan, dict):
        return plan

    t = (user_text or "").strip()
    tl = t.lower()

    # אם אין "בטעם/בטעמי" – לא נוגעים
    if not any(m in tl for m in _FLAVOR_MARKERS):
        return plan

    # קח את כל מה שאחרי הסמן הראשון שמופיע
    idx = min([tl.find(m) for m in _FLAVOR_MARKERS if tl.find(m) != -1])
    tail = t[idx:]
    # חתוך את המילים "בטעם/בטעמי/טעמים של" עצמן
    for m in _FLAVOR_MARKERS:
        if tail.lower().startswith(m):
            tail = tail[len(m):].strip()
            break

    if not tail:
        return plan

    # נרמול מפרידים: פסיק, ו, או, /, &
    tail = tail.replace("&", ",").replace("/", ",")
    tail = re.sub(r"\s+ו\s+", ",", tail)      # "... שוקולד ו קפה ..." -> ","
    tail = tail.replace(" או ", ",")          # "... שוקולד או קפה ..." -> ","
    tail = re.sub(r"[(){}\[\].!?\":;]", " ", tail)

    parts = [p.strip() for p in tail.split(",") if p.strip()]

    stop = {"וויסקי","בקבוק","בקבוקים","בקבוקי","טעם","טעמים","של","עם","יש","לי","איזה","אילו"}
    keys = []
    for p in parts:
        # הסר מילים כלליות מתוך הביטוי
        words = [w for w in p.split() if w not in stop]
        k = " ".join(words).strip()
        if k:
            keys.append(k)

    # dedupe
    Picked = []
    for k in keys:
        if k not in Picked:
            Picked.append(k)

    if not Picked:
        return plan

    # OR/AND
    logic = "OR" if " או " in tl else ("AND" if ("וגם" in tl or "שניהם" in tl) else (plan.get("filter_logic") or "OR"))

    # מכריח פילטרים על palette
    plan["filter_logic"] = logic
    plan["filters"] = [{"col": "palette", "op": "contains", "value": k} for k in Picked]

    # בוחר עמודות מינימליות לרשימה
    plan["select"] = ["distillery", "bottle_name", "bottle_id"]
    plan["limit"] = min(max(int(plan.get("limit") or 20), 1), 50)
    return plan


def normalize_plan_columns(plan: dict, df) -> dict:
    p = plan
    for f in (p.get("filters") or []):
        c = f.get("col")
        if c not in df.columns and c in COL_ALIASES and COL_ALIASES[c] in df.columns:
            f["col"] = COL_ALIASES[c]
    p["select"] = [COL_ALIASES.get(c, c) for c in (p.get("select") or [])]
    return p

SWEETNESS_RANGES = {
    'Very Sweet': (0, 1.5),
    'Sweet-Citrucy': (1.51, 2.0),
    'Citrucy-Spicy': (2.01, 2.749),
    'Coffee Like- Sea Salt': (2.75, 4.0),
    'Minerals - Sulfur': (4.01, 5.5),
    'Ash - BBQ Smoke': (5.51, 7.5),
    'Heavy Peat - Medicinal Smoke': (7.51, 10.0)
}

RICHNESS_RANGES = {
    'Very Watery': (0, 3.0),
    'Very Delicate': (3.01, 5.3),
    'Delicate': (5.31, 8.0),
    'Full Body': (8.01, 10.2),
    'Rich': (10.21, 12.5),
    'Very Rich': (12.51, 17.5),
    'Syrup Like': (17.51, 100.0)
}

import re

def _is_extremes_question(text: str) -> bool:
    """
    Only 'מה הכי ...' style questions.
    """
    if not text:
        return False
    t = text.strip().lower()
    return bool(re.search(r"\bהכי\b", t)) and bool(re.search(r"(מתוק|מעושן|עשן|עדין|עשיר|סמיך|כבד|מלא)", t))

def _is_focus_flavor_question(text: str) -> bool:
    """
    Bottle-specific follow-ups like:
    - כמה מתוק הוא?
    - האם הוא עדין?
    - הוא עשיר?
    - הוא סמיך?
    -הוא מעושן?
    -הוא מר?
    """
    if not text:
        return False
    t = text.strip().lower()
    return bool(re.search(r"(כמה\s*(הוא)?\s*(מתוק|מעושן|עדין|עשיר|סמיך)|האם\s*הוא\s*(מתוק|מעושן|עדין|עשיר|סמיך)|\bהוא\b\s*(מתוק|מעושן|עדין|עשיר|סמיך)|מתוק\?|מעושן\?|עדין\?|עשיר\?|סמיך\?)", t))


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _label_from_ranges(score: float, ranges_dict: dict) -> str:
    """
    Returns label from dict like {label: (low, high)} inclusive bounds.
    """
    if score is None:
        return "Unknown"
    try:
        v = float(score)
    except Exception:
        return "Unknown"

    for label, (low, high) in ranges_dict.items():
        if float(low) <= v <= float(high):
            return label
    return "Out of Range"

def _safe_float(x):
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return None
    return float(v)

def resolve_bottle_row_from_text(user_text: str, df: pd.DataFrame, last_bottle_id: int | None) -> pd.Series | None:
    """
    Resolve a bottle row using:
    1) Pronoun reference (הוא/זה/בו/אותו/עליו...) -> last_bottle_id
    2) Exact/substring match on bottle_name
    3) Distillery match (fallback: pick first bottle from that distillery)
    """
    if df is None or df.empty:
        return None

    t = _normalize_text(user_text)

    # (1) Pronoun / implicit reference -> last bottle
    if re.search(r"\b(הוא|זה|בו|בה|אותו|אותה|עליו|עליה)\b", t) and last_bottle_id:
        if "bottle_id" in df.columns:
            hit = df[df["bottle_id"].astype(str) == str(last_bottle_id)]
            if not hit.empty:
                return hit.iloc[0]

    # (2) Bottle name mention
    if "bottle_name" in df.columns:
        names = df["bottle_name"].dropna().astype(str).unique().tolist()
        # longest-first helps when one name contains another
        names = sorted(names, key=lambda x: len(_normalize_text(x)), reverse=True)
        for name in names:
            n = _normalize_text(name)
            if n and n in t:
                hit = df[df["bottle_name"].astype(str) == str(name)]
                if not hit.empty:
                    return hit.iloc[0]

    # (3) Distillery mention fallback
    if "distillery" in df.columns:
        dists = df["distillery"].dropna().astype(str).unique().tolist()
        dists = sorted(dists, key=lambda x: len(_normalize_text(x)), reverse=True)
        for dist in dists:
            d = _normalize_text(dist)
            if d and d in t:
                hit = df[df["distillery"].astype(str) == str(dist)]
                if not hit.empty:
                    return hit.iloc[0]

    return None

def try_handle_focus_bottle_flavor_questions(
    user_text: str,
    active_df: pd.DataFrame,
    context: ContextTypes.DEFAULT_TYPE
) -> str | None:
    """
    Answers bottle-specific questions about sweetness/smokiness and delicacy/richness.
    Uses your focus bottle mechanism:
      - First tries _get_focus_bottle_row(active_df, context)
      - If no focus, tries to detect bottle from text and then _set_focus_bottle(context, row)

    Recognizes:
      - מתוק/מעושן
      - עדין/עשיר/סמיך
      - "כמה הוא ..." / "הוא ..." / "...?" phrasing
    """
    if not user_text:
        return None

    t = _normalize_text(user_text)

    ask_sweet_smoky = bool(re.search(r"(כמה\s*הוא\s*מתוק|כמה\s*הוא\s*מעושן|הוא\s*מתוק|הוא\s*מעושן|מתוק\?|מעושן\?)", t))
    ask_richness    = bool(re.search(r"(כמה\s*הוא\s*עדין|כמה\s*הוא\s*עשיר|הוא\s*עדין|הוא\s*עשיר|עדין\?|עשיר\?|סמיך\?)", t))

    if not (ask_sweet_smoky or ask_richness):
        return None

    # 1) Try focus bottle
    row = _get_focus_bottle_row(active_df, context)

    # 2) If no focus, try to resolve from text and set focus
    if row is None:
        row = find_best_bottle_match(user_text, active_df )
        if row is None:
            return "על איזה בקבוק מדובר? תכתוב שם בקבוק/מזקקה, או תשאל קודם שאלה שמחזירה בקבוק ואז תגיד 'הוא מתוק?'"
        _set_focus_bottle(context, row)

    dist = str(row.get("distillery") or "-")
    bottle = str(row.get("bottle_name") or row.get("full_name") or "-")

    sep = "━━━━━━━━━━━━━━━━━━"
    out = [
        f"{sep}\n🧾 פרופיל בקבוק\n{sep}\n\n"
        f"{dist} – {bottle}\n"
    ]

    if ask_sweet_smoky:
        if "final_smoky_sweet_score" not in row.index:
            out.append("\n⚠️ אין לי final_smoky_sweet_score לבקבוק הזה.")
        else:
            s_val = _safe_float(row.get("final_smoky_sweet_score"))
            if s_val is None:
                out.append("\n⚠️ אין ערך תקין למדד מתוק↔עשן לבקבוק הזה.")
            else:
                label = _label_from_ranges(s_val, SWEETNESS_RANGES)
                out.append(
                    f"\n🍯 מתוק↔עשן: {s_val:.2f}\n"
                    f"🏷️ פרופיל: {label}\n"
                    "ℹ️ נמוך=יותר מתוק · גבוה=יותר מעושן"
                )

    if ask_richness:
        if "final_richness_score" not in row.index:
            out.append("\n⚠️ אין לי final_richness_score לבקבוק הזה.")
        else:
            r_val = _safe_float(row.get("final_richness_score"))
            if r_val is None:
                out.append("\n⚠️ אין ערך תקין למדד עדין↔עשיר לבקבוק הזה.")
            else:
                label = _label_from_ranges(r_val, RICHNESS_RANGES)
                out.append(
                    f"\n\n🌿 עדין↔עשיר: {r_val:.2f}\n"
                    f"🏷️ פרופיל: {label}\n"
                    "ℹ️ נמוך=יותר עדין · גבוה=יותר עשיר"
                )

    return "".join(out).strip()

def get_all_data_as_df(force_refresh: bool = False) -> pd.DataFrame:
    global CACHE_DATA
    if (
        (not force_refresh)
        and CACHE_DATA["df"] is not None
        and (time.time() - CACHE_DATA["last_update"] < 1800)
    ):
        return CACHE_DATA["df"]

    logging.info("Fetching fresh data from BigQuery to DataFrame...")
    query = f"""
    SELECT t1.*,
           t2.final_smoky_sweet_score, t2.final_richness_score,
           t3.avg_consumption_vol_per_day, t3.est_consumption_date, t3.predicted_finish_date, t3.Best_Before
    FROM `{TABLE_REF}` t1
    LEFT JOIN `{VIEW_REF}` t2 ON t1.bottle_name = t2.bottle_name AND t1.distillery = t2.distillery
    LEFT JOIN `{FORECAST_TABLE_REF}` t3 ON t1.bottle_id = t3.bottle_id
    """
    df = bq_client.query(query).to_dataframe()

    df["full_name"] = (df["distillery"].fillna("").astype(str) + " " + df["bottle_name"].fillna("").astype(str)).str.strip()

    CACHE_DATA["df"] = df
    CACHE_DATA["last_update"] = time.time()
    return df


# ==========================================
# Forecast helpers (optional extra columns)
# ==========================================
_FORECAST_CACHE = {"stock_map": None, "last_update": 0}

_HE_STOPWORDS = {
    "האם","יש","לי","של","את","זה","זו","בבקשה","בבקבוק","בקבוק","בקבוקים",
    "do","i","have","got","a","an","the","of","and","in","to"
}

def _tokenize_simple(s: str) -> set[str]:
    t = _normalize_text(s)
    # keep words and numbers
    parts = re.findall(r"[a-z0-9א-ת]+", t, flags=re.IGNORECASE)
    return {p for p in parts if p and p not in _HE_STOPWORDS and len(p) >= 2}

def _token_overlap_ok(query: str, candidate: str, min_overlap: float = 0.60) -> bool:
    q = _tokenize_simple(query)
    c = _tokenize_simple(candidate)
    if not q:
        return False
    inter = len(q & c)
    ratio = inter / max(1, len(q))
    return ratio >= min_overlap


def get_forecast_current_status_map(force_refresh: bool = False) -> dict:
    """Try to fetch bottle_id -> current_status from forecast table.

    If the column doesn't exist / query fails, returns empty dict.
    Cached for 30 minutes.
    """
    global _FORECAST_CACHE
    if (
        (not force_refresh)
        and _FORECAST_CACHE["stock_map"] is not None
        and (time.time() - _FORECAST_CACHE["last_update"] < 1800)
    ):
        return _FORECAST_CACHE["stock_map"]

    q = f"""
    SELECT bottle_id, current_status
    FROM `{FORECAST_TABLE_REF}`
    """
    try:
        df_cs = bq_client.query(q).to_dataframe()
        m = {}
        if not df_cs.empty and "bottle_id" in df_cs.columns and "current_status" in df_cs.columns:
            for _, r in df_cs.iterrows():
                try:
                    m[int(r["bottle_id"])] = float(r["current_status"]) if pd.notnull(r["current_status"]) else None
                except Exception:
                    continue
        _FORECAST_CACHE["stock_map"] = m
        _FORECAST_CACHE["last_update"] = time.time()
        return m
    except Exception as e:
        logging.warning(f"Could not fetch current_status from forecast table: {e}")
        _FORECAST_CACHE["stock_map"] = {}
        _FORECAST_CACHE["last_update"] = time.time()
        return {}



import ast
import re

def normalize_to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            out.extend(normalize_to_list(v))
        return out

    s = str(x).strip()
    if not s:
        return []

    # try literal_eval for "['A','B']"
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            return normalize_to_list(parsed)
        except Exception:
            # fallback: extract quoted tokens
            toks = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
            out = []
            for a,b in toks:
                t = (a or b or "").strip()
                if t:
                    out.append(t)
            if out:
                return out
            # last resort split
            inner = s.strip("[]()")
            return [p.strip() for p in re.split(r"[,;\n]+", inner) if p.strip()]

    # plain string
    return [p.strip() for p in re.split(r"[,;\n]+", s) if p.strip()]


def compute_cask_ranking(df: pd.DataFrame, col="casks_aged_in") -> pd.DataFrame:
    counts = {}
    for x in df[col].dropna().tolist():
        for it in normalize_to_list(x):
            it = str(it).strip()
            if not it:
                continue
            counts[it] = counts.get(it, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["casks_aged_in", "count"])

    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return pd.DataFrame(items, columns=["casks_aged_in", "count"])

def format_result(res):
    # אם זה מספר בודד (COUNT / SUM וכו')
    if not hasattr(res, "shape"):
        return f"התוצאה היא: {res}"

    rows, cols = res.shape

    # אם אין תוצאות
    if rows == 0:
        return "לא נמצאו תוצאות."

    # אם תא בודד
    if rows == 1 and cols == 1:
        val = res.iloc[0, 0]
        return f"התוצאה היא: {val}"

    # אם שורה אחת (תוצאה מצומצמת)
    if rows == 1:
        parts = [f"{col}: {res.iloc[0][col]}" for col in res.columns]
        return " | ".join(parts)

    # אם עד 5 שורות – טבלה קטנה
    if rows <= 5:
        return res.to_string(index=False)

    # טבלה גדולה
    preview = res.head(10).to_string(index=False)
    return f"נמצאו {rows} תוצאות.\n\nטופ 10:\n{preview}"

def format_top_casks(df_rank):
    top_name = df_rank.iloc[0]["casks_aged_in"]
    top_count = int(df_rank.iloc[0]["count"])

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    lines = []
    for i, (_, row) in enumerate(df_rank.head(5).iterrows()):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        lines.append(f"{medal} {row['casks_aged_in']} — {int(row['count'])}")

    return (
        "🥃 החבית הכי פופולרית אצלך\n"
        f"{top_name}\n"
        f"סה״כ: {top_count} בקבוקים\n\n"
        "🏆 טופ 5:\n"
        + "\n".join(lines)
    )

def _split_concatenated_by_vocab(s: str, options: list[str]) -> list[str] | None:
    """
    If s looks like multiple vocab tokens stuck together (e.g. 'Red Wine OakFruits'),
    try to split it into a list of known options using DP on a normalized representation.
    Returns list[str] if split found (len>=2), else None.
    """
    if not s or not options:
        return None

    # normalize: keep only letters+digits (same spirit as your _letters_only)
    def norm(x: str) -> str:
        x = (x or "").lower()
        return re.sub(r"[^a-z0-9]", "", x)

    target = norm(s)
    if not target:
        return None

    # map normalized option -> original option (keep longest first)
    norm_opts = []
    for opt in options:
        no = norm(opt)
        if no:
            norm_opts.append((no, opt))
    norm_opts.sort(key=lambda t: len(t[0]), reverse=True)

    # DP: dp[i] = best split up to i
    dp = {0: []}
    n = len(target)
    for i in range(n + 1):
        if i not in dp:
            continue
        for no, opt in norm_opts:
            if target.startswith(no, i):
                j = i + len(no)
                cand = dp[i] + [opt]
                # keep first found / shortest list; either is fine
                if j not in dp or len(cand) < len(dp[j]):
                    dp[j] = cand

    res = dp.get(n)
    if res and len(res) >= 2:
        return res
    return None
# ==========================================
# Conversation focus (per-user)
# ==========================================
# We keep a lightweight "focus bottle" so the user can ask follow-ups like:
# "וכמה אחוז נשאר בו?" / "מתי הוא נגמר?" / "תעדכן לי 60ml" (refers to last bottle)
def _set_focus_bottle(context: ContextTypes.DEFAULT_TYPE, row: pd.Series | dict):
    try:
        bid = int(row["bottle_id"]) if isinstance(row, dict) else int(row.get("bottle_id"))
    except Exception:
        return
    context.user_data["focus_bottle_id"] = bid
    # optional debug/context fields
    try:
        context.user_data["focus_full_name"] = str(row.get("full_name") or row.get("bottle_name") or "")
        context.user_data["focus_distillery"] = str(row.get("distillery") or "")
    except Exception:
        pass



def _get_focus_bottle_row(active_df: pd.DataFrame, context: ContextTypes.DEFAULT_TYPE) -> pd.Series | None:
    bid = context.user_data.get("focus_bottle_id")
    if not bid:
        return None
    try:
        sub = active_df[active_df["bottle_id"] == int(bid)]
        if sub.empty:
            return None
        return sub.iloc[0]
    except Exception:
        return None

def _clear_focus(context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("focus_bottle_id", None)
    context.user_data.pop("focus_full_name", None)
    context.user_data.pop("focus_distillery", None)

# ==========================================
# אלגוריתם Levenshtein + ניקוי טקסט
# ==========================================
def _normalize_text(s: str) -> str:
    """Normalize for fuzzy matching.

    - Lowercase, strip
    - Unify punctuation to spaces
    - Keep '&' semantics by expanding to 'and'
    - Add practical alias normalizations (e.g., m&h -> milk honey)
    """
    if s is None:
        return ""
    s = str(s).lower().strip()

    # --- alias shortcuts BEFORE punctuation stripping ---
    s = re.sub(r"\bm\s*&\s*h\b", "milk honey", s)
    s = re.sub(r"\bm\s+and\s+h\b", "milk honey", s)

    # make '&' meaningful for names like 'Milk & Honey'
    s = s.replace("&", " and ")

    # unify quotes & punctuation -> space
    s = re.sub(r"[’'`\"“”]", " ", s)
    s = re.sub(r"[_\-/\\|(){}\[\],.:;!?+*=<>@#$%^~]", " ", s)

    s = re.sub(r"\s+", " ", s).strip()

    # common noise words (english)
    s = re.sub(r"\bthe\b", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _letters_only(s: str) -> str:
    s = _normalize_text(s)
    return re.sub(r"[^a-z0-9]", "", s)

def _initialism(name: str) -> str:
    """Initials of words: 'Milk and Honey' -> 'mh'."""
    n = _normalize_text(name)
    parts = [p for p in n.split() if p and p not in ("and",)]
    return "".join([p[0] for p in parts if p]).lower()

def get_levenshtein_distance(s1: str, s2: str) -> int:
    s1, s2 = _normalize_text(s1), _normalize_text(s2)
    if len(s1) < len(s2):
        return get_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def _similarity_ratio(a: str, b: str) -> float:
    a_n, b_n = _normalize_text(a), _normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    d = get_levenshtein_distance(a_n, b_n)
    return 1.0 - (d / max(len(a_n), len(b_n)))

def format_df_answer(df: pd.DataFrame, plan: dict) -> str:
    if df is None or df.empty:
        return "לא נמצאו תוצאות."

    # אם יש aggregations (למשל כמה בקבוקים)
    if plan.get("aggregations"):
        # לוקחים את הערך הראשון מהתוצאה
        row = df.iloc[0]
        lines = []
        for col in df.columns:
            val = row.get(col)
            lines.append(f"{col}: {val}")
        return "\n".join(lines)

    # אם זו רשימת בקבוקים
    cols = df.columns.tolist()

    # אם יש bottle_name ו-distillery
    if "bottle_name" in cols:
        lines = []
        for _, r in df.iterrows():
            dist = r.get("distillery", "")
            name = r.get("bottle_name", "")
            if dist:
                lines.append(f"• {dist} — {name}")
            else:
                lines.append(f"• {name}")
        return "\n".join(lines)

    # אם זו שאלה על שדה בודד (למשל ABV)
    if len(cols) == 1:
        val = df.iloc[0][cols[0]]
        return f"{cols[0]}: {val}"

    # fallback כללי
    return df.to_string(index=False)


# ==========================================
# SQL helpers
# ==========================================
def sql_array(arr):
    if not arr:
        return "[]"
    safe = [f"'{str(x).replace(chr(39), chr(92)+chr(39))}'" for x in arr]
    return f"[{', '.join(safe)}]"

def _escape_sql_str(s: str) -> str:
    return str(s).replace("'", "\\'")

# ==========================================
# Intent detection (count vs update) + parsing
# ==========================================
_COUNT_HINTS = ("כמה", "וכמה", "how many", "count")
_BOTTLE_HINTS = ("בקבוק", "bottle", "bottles")
_HAVE_HINTS = ("יש לי", "do i have", "have i got")
_UPDATE_HINTS = ("שתיתי", "מזגתי", "מזיגה", "שתייה", "עדכן", "הורד", "פחת", "drank", "poured", "drink", "update", "reduce")
_CONFIRM_YES = ("כן", "כן.", "כן!", "יאפ", "y", "yes", "sure", "ok", "אוקיי", "אוקי")
_CONFIRM_NO = ("לא", "לא.", "לא!", "n", "no", "nope")
_CANCEL_WORDS = ("ביטול", "בטל", "/בטל", "cancel", "/cancel", "צא", "exit", "stop")

def _looks_like_count_query(text: str) -> bool:
    t = _normalize_text(text)
    return any(h in t for h in _COUNT_HINTS) and (any(h in t for h in _BOTTLE_HINTS) or any(h in t for h in _HAVE_HINTS))

import re
import pandas as pd

def try_handle_extremes_sweet_smoky_rich_delicate(user_text: str, df: pd.DataFrame) -> str | None:
    """
    Handles deterministic Q&A for:
    - sweetest / smokiest based on final_smoky_sweet_score (0-20)
    - most delicate / richest based on final_richness_score (0-30)

    Rules:
      final_smoky_sweet_score (0..20): low=sweeter, high=smokier
      final_richness_score   (0..30): low=more delicate, high=richer
    """
    if not user_text:
        return None

    t = user_text.strip().lower()

    # --- intent detection (Hebrew-focused) ---
    want_sweetest = bool(re.search(r"הכי\s*מתוק", t))
    want_smokiest = bool(re.search(r"הכי\s*מעושן|הכי\s*עשן|הכי\s*מעשן", t))
    want_delicate = bool(re.search(r"הכי\s*עדין|הכי\s*קליל|הכי\s*רך", t))
    want_richest  = bool(re.search(r"הכי\s*עשיר|הכי\s*סמיך|הכי\s*כבד|הכי\s*מלא", t))

    if not any([want_sweetest, want_smokiest, want_delicate, want_richest]):
        return None

    if df is None or df.empty:
        return "אין לי נתונים כרגע כדי לחשב את זה."

    # Optional: filter active bottles if stock_status_per exists
    sub = df[df["alcohol_type"].isin([
        "Single Malt Whisky",
        "Blended Scotch Whisky"
    ])].copy()
    all_series = sub["final_smoky_sweet_score"]
    all_series_richness = sub["final_richness_score"]
    if "stock_status_per" in sub.columns:
        sub = sub[pd.to_numeric(sub["stock_status_per"], errors="coerce").fillna(0) > 0]

    def _top_rows(score_col: str, ascending: bool, k: int = 3):
        if score_col not in sub.columns:
            return None, f"לא מצאתי את העמודה {score_col} בטבלה."

        s = sub.copy()
        s[score_col] = pd.to_numeric(s[score_col], errors="coerce")
        s = s.dropna(subset=[score_col])

        if s.empty:
            return None, f"אין ערכים תקינים בעמודה {score_col} כדי לבחור קצה סקאלה."

        s = s.sort_values(by=score_col, ascending=ascending)
        return s.head(k), None

    def _fmt_line(i: int, dist: str, bottle: str, score: float, max_score: int) -> str:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
        dist = (dist or "-").strip()
        bottle = (bottle or "-").strip()
        return f"{medal} {dist} – {bottle}  ·  {score:.2f}/{max_score}"

    def _pretty_top3(title: str, metric_label: str, max_score: int, max_2: int, hint: str, rows: pd.DataFrame, score_col: str, all_series: pd.Series, richness: pd.Series) -> str:
        sep = "━━━━━━━━━━━━━━━━━━"
        lines = []
        n = min(3, len(rows))
        if 'מתוקים' in title or 'מעושנים' in title:
            max_score = max_score
        elif 'עשירים/סמיכים' in title or 'עדינים' in title:
            max_score = max_2
            
        for idx in range(n):
            r = rows.iloc[idx]
            score = float(r.get(score_col))
            lines.append(_fmt_line(idx + 1, str(r.get("distillery", "-")), str(r.get("bottle_name", "-")), score, max_score))

        avg = float(pd.to_numeric(all_series, errors="coerce").dropna().mean()) if all_series is not None else None
        avg_richness = float(pd.to_numeric(richness, errors="coerce").dropna().mean()) if richness is not None else None
        avg_line = (
    f"📌 ממוצע מתיקות: {avg:.2f} / {max_score}\n"
    f"📌 ממוצע סמיכות: {avg_richness:.2f} / {max_2}\n"
) 

        return (
            f"{sep}\n"
            f"{title}\n"
            f"{sep}\n\n"
            + "\n".join(lines)
            + "\n\n"
            + f"📊 {metric_label} (טופ {n})\n"
            + avg_line
            + hint
        )

    # --- Decide which extreme to compute (Top 3) ---

    if want_sweetest:
        rows, err = _top_rows("final_smoky_sweet_score", ascending=True, k=3)  # low = sweetest
        if err:
            return err
        
       
        return _pretty_top3(
            title="🍯 טופ 3 הבקבוקים הכי מתוקים אצלך",
            metric_label="מדד מתוק↔עשן",
            max_score=12,
            max_2 = 30,
            hint="🟢 נמוך יותר = מתוק יותר",
            rows=rows,
            score_col="final_smoky_sweet_score", all_series=all_series, richness=all_series_richness
        )

    if want_smokiest:
        rows, err = _top_rows("final_smoky_sweet_score", ascending=False, k=3)  # high = smokiest
        if err:
            return err
       
        return _pretty_top3(
            title="🔥 טופ 3 הבקבוקים הכי מעושנים אצלך",
            metric_label="מדד מתוק↔עשן",
            max_score=12,
            max_2 = 30,
            hint="🟠 גבוה יותר = מעושן יותר",
            rows=rows,
            score_col="final_smoky_sweet_score", all_series=all_series, richness=all_series_richness
        )

    if want_delicate:
        rows, err = _top_rows("final_richness_score", ascending=True, k=3)  # low = most delicate
        if err:
            return err
        
        return _pretty_top3(
            title="🌿 טופ 3 הבקבוקים הכי עדינים אצלך",
            metric_label="מדד עדין↔עשיר",
            max_score = 12,
            max_2=30,
            
            hint="🟢 נמוך יותר = עדין יותר",
            rows=rows,
            score_col="final_richness_score", all_series=all_series, richness=all_series_richness
        )

    if want_richest:
        rows, err = _top_rows("final_richness_score", ascending=False, k=3)  # high = richest
        if err:
            return err
       
        return _pretty_top3(
            title="🥃 טופ 3 הבקבוקים הכי עשירים/סמיכים אצלך",
            metric_label="מדד עדין↔עשיר",
            max_score = 12,
            max_2=30,
            hint="🟠 גבוה יותר = עשיר יותר",
            rows=rows,
            score_col="final_richness_score", all_series=all_series, richness=all_series_richness
        )

    return None

def _looks_like_have_query(text: str) -> bool:
    """
    True for: "האם יש לי X?", "יש לי X?", "do i have X?"
    Excludes "כמה..." which is handled by count intent.
    """
    t = _normalize_text(text)
    if any(h in t for h in _COUNT_HINTS):
        return False
    return any(h in t for h in _HAVE_HINTS)

def _looks_like_flavors_of_bottle_query(text: str) -> bool:
    t = _normalize_text(text)
    # "טעמים/ארומות/נוז/פלטה" – אבל לא "הכי פופולרי"
    if any(h in t for h in _POPULAR_HINTS):
        return False
    return any(k in t for k in (
        "מה הטעמים", "טעמים", "טעם", "ארומות", "ארומה",
        "nose", "palate", "palette", "פרופיל טעם", "פרופיל הטעם"
    ))

def _looks_like_casks_of_bottle_query(text: str) -> bool:
    # אם זה "פופולרי" → זה לא “איזה חבית הוא”, זה דירוג כללי
    if _is_popular_cask_question(text) or _looks_like_popular_query(text):
        return False
    t = _normalize_text(text)
    has_cask = ("חבית" in t) or ("חביות" in t) or ("cask" in t)
    # ניסוחים טיפוסיים של "איזה חבית הוא"
    has_which = ("איזה" in t) or ("מה" in t) or ("שלו" in t) or ("של" in t) or ("aged" in t)
    return has_cask and has_which

def _looks_like_age_of_bottle_query(text: str) -> bool:
    t = _normalize_text(text)
    # גיל / שנים / בן כמה / age / aged
    return any(k in t for k in ("גיל", "בן כמה", "כמה שנים", "age", "aged"))


def build_age_reply(row) -> str:
    full_name = str(row.get("full_name") or "").strip()
    age = row.get("age")

    # נורמליזציה
    try:
        if age is None or (isinstance(age, float) and pd.isna(age)) or str(age).strip() == "":
            age_txt = "לא מוגדר לי גיל לבקבוק הזה."
        else:
            age_int = int(float(age))
            age_txt = f"{age_int} שנים"
    except Exception:
        age_txt = f"{str(age).strip()}"

    return (
        f"🎂 *הגיל של הוויסקי:*\n"
        f"🥃 *{full_name}*\n"
        f"{age_txt}"
    )
    
    
def _repair_broken_phrase_items(items: list[str]) -> list[str]:
    """
    Merge adjacent list items when the next item looks like a continuation
    (e.g., starts with lowercase): ['Pedro Xime','ez Sherry'] -> ['Pedro Ximenez Sherry']
    """
    if not items:
        return []

    raw = [str(x).strip() for x in items if str(x).strip()]
    merged = []
    i = 0

    while i < len(raw):
        cur = raw[i]

        # while next looks like continuation (starts with lowercase)
        while i + 1 < len(raw):
            nxt = raw[i + 1].strip()
            if not nxt:
                i += 1
                continue

            if nxt and nxt[0].islower():
                # join without extra space (continuation of a broken word)
                cur = cur + nxt
                i += 1
                continue

            break

        merged.append(cur)
        i += 1

    # de-dup preserving order
    seen = set()
    out = []
    for x in merged:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)

    return out


def _camel_to_words(text: str) -> list[str]:
    """
    'BerriesSherryCaramelHoneyVanilla' -> ['Berries','Sherry','Caramel','Honey','Vanilla']
    Also supports: '...Vanilla Smoke' -> adds 'Smoke' properly.
    """
    if not text:
        return []

    t = str(text).strip()
    t = t.replace("\u200f", " ").replace("\u200e", " ")
    t = re.sub(r"\s+", " ", t)

    words = []
    for chunk in t.split(" "):
        if not chunk:
            continue
        # Extract CamelCase words; fallback to the chunk itself if no matches
        found = re.findall(r"[A-Z][a-z]+", chunk)
        if found:
            words.extend(found)
        else:
            words.append(chunk)

    # Merge Light Smoke if you ever get that (optional)
    merged = []
    i = 0
    while i < len(words):
        if i + 1 < len(words) and words[i] == "Light" and words[i+1] == "Smoke":
            merged.append("Light Smoke")
            i += 2
        else:
            merged.append(words[i])
            i += 1

    return merged


def _as_bullets(items: list[str]) -> str:
    if not items:
        return "• -"
    return "\n".join([f"• {x}" for x in items])


def build_flavors_reply(row) -> str:
    full_name = str(row.get("full_name") or "").strip()

    nose_raw = normalize_to_list(row.get("nose"))
    pal_raw  = normalize_to_list(row.get("palette"))

    # Sometimes it's a single concatenated string inside a list
    nose_text = " ".join([str(x).strip() for x in (nose_raw or []) if str(x).strip()])
    pal_text  = " ".join([str(x).strip() for x in (pal_raw or []) if str(x).strip()])

    nose_words = _camel_to_words(nose_text)
    pal_words  = _camel_to_words(pal_text)

    nose_out = ", ".join(nose_words) if nose_words else "-"
    pal_out  = ", ".join(pal_words) if pal_words else "-"

    return (
        f"👃 *Nose* של הבקבוק:\n"
        f"🥃 *{full_name}*\n"
        f"{nose_out}\n\n"
        f"👅 *Palate* של הבקבוק:\n"
        f"🥃 *{full_name}*\n"
        f"{pal_out}"
    )

def build_casks_reply(row) -> str:
    full_name = str(row.get("full_name") or "").strip()

    # raw list
    casks_raw = normalize_to_list(row.get("casks_aged_in"))

    # ✅ repair broken items (like Xime + ez Sherry)
    casks_fixed = _repair_broken_phrase_items(casks_raw or [])

    # אם לפעמים זה מגיע כמחרוזת אחת עם פסיקים, תפצל רק על פסיקים/נקודה-פסיק/|
    if len(casks_fixed) == 1 and any(sep in casks_fixed[0] for sep in [",", ";", "|"]):
        parts = re.split(r"[;,|]\s*", casks_fixed[0])
        casks_fixed = [p.strip() for p in parts if p.strip()]

    bullets = _as_bullets(casks_fixed)

    return (
        f"🪵 *חביות (casks_aged_in):*\n"
        f"🥃 *{full_name}*\n"
        f"{bullets}"
    )

def _extract_entity_for_have(text: str) -> str:
    """
    Extract entity from have-queries:
      - 'האם יש לי Glenfiddich Project XX?'
      - 'יש לי m&h?'
      - 'do i have lagavulin 16?'
    """
    t = text.strip()

    patterns = [
        r"(?:האם\s+)?יש\s+לי\s+(.+?)(?:\?|$)",
        r"do\s+i\s+have\s+(.+?)(?:\?|$)",
        r"have\s+i\s+got\s+(.+?)(?:\?|$)",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # fallback: remove just the "have" hint words
    tt = _normalize_text(t)
    for h in _HAVE_HINTS:
        tt = tt.replace(_normalize_text(h), " ")
    tt = re.sub(r"\s+", " ", tt).strip()
    return tt


def _looks_like_update(text: str) -> bool:
    t = _normalize_text(text)
    return any(h in t for h in _UPDATE_HINTS)

def _extract_amount_ml(text: str) -> int | None:
    # try: "60ml", "60 ml", "60מ״ל", "60 מ\"ל", "60 מ״ל"
    t = text.replace("מ״ל", "ml").replace('מ"ל', "ml").replace("מ''ל", "ml")
    m = re.search(r"(\d{1,4})\s*(ml)\b", t, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # maybe just a number after verbs
    m2 = re.search(r"\b(\d{1,4})\b", t)
    if m2:
        # heuristic: ignore years (>=1900)
        n = int(m2.group(1))
        if n < 1900:
            return n
    return None

def _extract_entity_for_count(text: str) -> str:
    """
    Heuristics:
      - 'כמה בקבוקי X יש לי'
      - 'כמה בקבוקים של X יש לי'
      - 'how many X bottles do I have'
    """
    t = text.strip()

    # Hebrew patterns
    patterns = [
        r"כמה\s+בקבוק(?:ים|י)?\s+(?:של\s+)?(.+?)(?:\s+יש\s+לי|\?|$)",
        r"יש\s+לי\s+כמה\s+בקבוק(?:ים|י)?\s+(?:של\s+)?(.+?)(?:\?|$)",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # English-ish
    m = re.search(r"how\s+many\s+(.+?)\s+bottles", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # fallback: remove common words
    tt = _normalize_text(t)
    tt = re.sub(r"\bכמה\b", "", tt).strip()
    tt = re.sub(r"\bבקבוק(?:ים|י)?\b", "", tt).strip()
    tt = re.sub(r"\bיש לי\b", "", tt).strip()
    return tt.strip()

def _extract_entity_for_update(text: str) -> str:
    t = text.strip()
    # Remove amount + ml tokens
    t2 = re.sub(r"\b\d{1,4}\s*(ml|מ״ל|מ\"ל)\b", "", t, flags=re.IGNORECASE)
    t2 = re.sub(r"\b\d{1,4}\b", "", t2)
    # Remove verb-ish tokens (hebrew + english)
    for w in ["שתיתי", "מזגתי", "עדכן", "תעדכן", "הורד", "להוריד", "פחת", "שתייה", "מזיגה", "drank", "poured", "drink", "update", "reduce"]:
        t2 = re.sub(rf"\b{re.escape(w)}\b", "", t2, flags=re.IGNORECASE)
    t2 = re.sub(r"\s+", " ", t2).strip()
    return t2

# ==========================================
# Intent detection: analytics / research questions
# ==========================================
_POPULAR_HINTS = ("הכי פופולרי", "most popular", "popular", "פופולרי", "הכי נצרך", "הכי נשתה")
_OXIDIZED_HINTS = (
    "מחומצן", "חמצון", "oxid", "oxidized",
    "best before", "בסט ביפור", "bestbefore",
    "תוקף", "תאריך תוקף", "פג תוקף",
    "סטטוס החמצון", "סטטוס חמצון"
)

# We split "recommend" into two different user intents:
# 1) "What should I drink soon"  -> Best Before within ~3 months
# 2) "What should I drink now / next dram / based on taste profile" -> Estimated consumption date (forecast urgency)
_RECOMMEND_SOON_HINTS = (
    "מה כדאי לשתות בקרוב", "מה לשתות בקרוב", "בקרוב", "בזמן הקרוב", "במהלך החודשים הקרובים",
    "שלושה חודשים", "3 חודשים", "3חודשים", "best before", "בסט ביפור"
)
_RECOMMEND_NOW_HINTS = (
    "מה לשתות עכשיו", "מה כדאי לשתות עכשיו", "עכשיו", "איזה דראם", "dram", "דראם",
    "תמליץ על הדראם הבא", "הדראם הבא", "recommend", "המלץ", "תמליץ", "פרופיל הטעם", "פרופיל טעם",
    "בהתבסס על היסטוריית", "בהתבסס על היסטורית", "היסטוריית השתייה", "היסטורית השתייה"
)

_STOCK_HINTS = (
    "כמה אחוז", "כמה %", "אחוז נשאר",
    "כמה נשאר", "כמה נשאר לי", "כמה נשאר מהבקבוק", "וכמה נשאר", "נשאר",
    "remaining", "left", "stock", "מלאי", "מלאי נשאר"
)


# Portfolio / share analytics (NOT "remaining stock")
# Examples:
# - "מה אחוז בקבוקי הוויסקי שלי מכלל האלכוהול שלי?"
# - "what percentage of my bottles are whisky?"
_PORTFOLIO_SHARE_HINTS = (
    "מכלל", "מתוך", "בסך", "סה\"כ", "סהכ",
    "overall", "total", "out of", "percentage of", "percent of",
    "share of", "ratio of"
)

def _looks_like_portfolio_share_query(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False

    # Must be a percent-ish question
    has_percent = ("אחוז" in t) or ("%" in text) or ("percent" in t) or ("percentage" in t)

    if not has_percent:
        return False

    # Explicitly avoid "remaining in bottle" stock questions
    if any(w in t for w in ("נשאר", "remaining", "left", "מלאי")):
        return False

    # Needs an "overall / total" framing
    if not any(h in t for h in _PORTFOLIO_SHARE_HINTS):
        return False

    # Likely about the collection (bottles/alcohol types)
    if ("בקבוק" in t) or ("bottle" in t) or ("alcohol" in t) or ("אלכוהול" in t):
        return True

    # If user wrote just category + percent + total, still treat as portfolio share
    return True

# ==========================================
# Fast deterministic portfolio analytics (no Gemini)
# Handles: counts, percentages of total, popular casks
# ==========================================

_WHISKY_PATTERNS = (
    r"whisky", r"whiskey", r"single\\s*malt", r"blended\\s*whisky", r"scotch", r"bourbon", r"rye"
)
_WINE_PATTERNS = (r"\\bwine\\b", r"red\\s*wine", r"white\\s*wine", r"dessert\\s*wine")

def _col_as_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series([], dtype="string")
    return df[col].fillna("").astype(str)

def _count_bottles(df: pd.DataFrame, mask: pd.Series | None = None) -> int:
    if df is None or df.empty:
        return 0
    sub = df if mask is None else df[mask]
    if sub.empty:
        return 0
    if "bottle_id" in sub.columns:
        try:
            return int(sub["bottle_id"].nunique())
        except Exception:
            return int(len(sub))
    return int(len(sub))

def _detect_category(user_text: str) -> dict | None:
    t = _normalize_text(user_text)

    # wine subtypes
    if "יין אדום" in t or ("red wine" in t):
        return {"kind": "wine_red"}
    if "יין לבן" in t or ("white wine" in t):
        return {"kind": "wine_white"}
    if "יין" in t or ("wine" in t):
        return {"kind": "wine"}

    # whisky
    if "וויסקי" in t or "ויסקי" in t or "whisky" in t or "whiskey" in t:
        return {"kind": "whisky"}

    return None

def _make_category_mask(df: pd.DataFrame, kind: str) -> pd.Series:
    s = _col_as_str_series(df, "alcohol_type").str.lower()

    if kind == "whisky":
        pat = "(" + "|".join(_WHISKY_PATTERNS) + ")"
        return s.str.contains(pat, regex=True, na=False)

    if kind == "wine":
        return s.str.contains("(" + "|".join(_WINE_PATTERNS) + ")", regex=True, na=False)

    if kind == "wine_red":
        return s.str.contains(r"red\\s*wine", regex=True, na=False)

    if kind == "wine_white":
        return s.str.contains(r"white\\s*wine", regex=True, na=False)

    return pd.Series([False] * len(df))

_TEXT_SEARCH_INTENT_RE = re.compile(
    r"(בטעם|בטעמי|טעם|טעמים|ארומה|ארומות|ריח|נוז|nose|aroma|palate|palette|taste|flavor|חבית|חביות|יישון|cask|casks|aged|שרי|sherry)",
    re.IGNORECASE
)    

def looks_like_text_intent(user_text: str) -> bool:
    return bool(_TEXT_SEARCH_INTENT_RE.search(user_text or ""))

def _is_count_question(user_text: str) -> bool:
    t = _normalize_text(user_text)
    return ("כמה בקבוק" in t) or ("how many bottle" in t) or ("number of bottle" in t)

def _is_popular_cask_question(user_text: str) -> bool:
    t = _normalize_text(user_text)
    return ("חבית" in t or "cask" in t) and (("פופולר" in t) or ("הכי" in t) or ("most" in t) or ("popular" in t))

def _try_fast_portfolio_answer(user_text: str, df: pd.DataFrame):
    if df is None or df.empty:
        return None

    t = str(user_text).lower()
    if ("cask" in t or "חבית" in t or "חביות" in t) and ("popular" in t or "הכי" in t or "פופולר" in t):
        if "casks_aged_in" not in df.columns:
            return "אין עמודת casks_aged_in בנתונים."

        df_rank = compute_cask_ranking(df)  # הפונקציה שלך שמחזירה df עם count
        if df_rank is None or df_rank.empty:
            return "לא מצאתי נתוני חביות."

        return format_top_casks(df_rank)   # ✅ מחזיר טקסט, לא DF

    return None

# Pronouns / placeholders that usually mean: "the one we just talked about"
_FOCUS_PRONOUNS = (
    "בו", "בזה", "באותו", "אותו", "זה", "הוא", "הבקבוק", "that", "it", "this", "him"
)

def _looks_like_popular_query(text: str) -> bool:
    t = _normalize_text(text)
    return any(h in t for h in _POPULAR_HINTS)

def _looks_like_oxidized_query(text: str) -> bool:
    t = _normalize_text(text)
    return any(h in t for h in _OXIDIZED_HINTS)

def _looks_like_recommend_soon_query(text: str) -> bool:
    t = _normalize_text(text)
    # If user explicitly asks for "soon" or mentions Best Before => route to soon
    return any(h in t for h in _RECOMMEND_SOON_HINTS)

def _looks_like_recommend_now_query(text: str) -> bool:
    t = _normalize_text(text)
    return any(h in t for h in _RECOMMEND_NOW_HINTS) or ("מה לשתות" in t and "בקרוב" not in t)

def _looks_like_stock_query(text: str) -> bool:
    t = _normalize_text(text)
    return any(h in t for h in _STOCK_HINTS)

def _is_focus_placeholder(term: str) -> bool:
    """Return True if extracted term is effectively a pronoun / placeholder."""
    t = _normalize_text(term)
    if not t:
        return True
    if t in _FOCUS_PRONOUNS:
        return True
    if len(t) <= 2:
        return True
    return False

def _safe_to_datetime(x):
    try:
        if pd.isna(x):
            return pd.NaT
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

# ==========================================
# Add New Bottle via Telegram Photo (AI Scan)
# ==========================================

_ADD_BOTTLE_HINTS = (
    "הוסף בקבוק חדש", "אני רוצה להוסיף בקבוק", "בקבוק חדש",
    "add new bottle", "add bottle", "new bottle"
)

_FOCUS_PLACEHOLDERS = ("בו", "בה", "זה", "אותו", "אותה", "שלו", "שלה", "it", "this", "that")

def _looks_like_add_bottle(text: str) -> bool:
    t = _normalize_text(text)
    triggers = (
        "הוסף בקבוק",
        "הוספת בקבוק",
        "בקבוק חדש",
        "חדש בקבוק",
        "אני רוצה להוסיף",
        "add bottle",
        "new bottle"
    )
    return any(tr in t for tr in triggers)

def _set_add_stage(context, stage: str):
    context.chat_data["add_stage"] = stage

def _get_add_stage(context) -> str | None:
    return context.chat_data.get("add_stage")

def _set_add_payload(context, payload: dict):
    context.chat_data["add_payload"] = payload

def _get_add_payload(context) -> dict:
    return context.chat_data.get("add_payload", {})

def _clear_add_flow(context):
    for k in ("add_stage", "add_payload"):
        context.chat_data.pop(k, None)

def _sql_str_or_null(v):
    if v is None:
        return "NULL"
    s = str(v).strip()
    if s == "":
        return "NULL"
    return f"'{_escape_sql_str(s)}'"

def _unique_from_scalar_col(df: pd.DataFrame, col: str) -> list[str]:
    if df is None or df.empty or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).tolist()
    vals = [v.strip() for v in vals if v.strip()]
    return sorted(list(set(vals)))

def _unique_from_array_col(df: pd.DataFrame, col: str) -> list[str]:
    if df is None or df.empty or col not in df.columns:
        return []
    out = set()
    for x in df[col].dropna().tolist():
        if isinstance(x, (list, tuple)):
            for it in x:
                if it is None:
                    continue
            for tok in normalize_to_list(it):
                if tok:
                    out.add(tok)
        else:
            s = str(x).strip()
            if s:
                out.add(s)
    return sorted(out)

def _map_to_closest(val: str | None, options: list[str], threshold: float = 0.68) -> str | None:
    if val is None:
        return None
    v = str(val).strip()
    if not v:
        return None
    if not options:
        return v
    best = None
    best_score = -1.0
    for opt in options:
        sc = _similarity_ratio(v, opt)
        if sc > best_score:
            best_score = sc
            best = opt
    if best is None:
        return v
    return best if best_score >= threshold else v # return best anyway to keep controlled vocabulary

def _map_list_to_options(vals, options: list[str], threshold: float = 0.65, top_k: int = 5) -> list[str]:
    if not vals:
        return []

    # אם זה מחרוזת - נסה לפרק לרשימה
    if isinstance(vals, str):
        parts = normalize_to_list(vals)  # הפונקציה שלך
        vals = parts

    out = []
    for v in vals:
        if v is None:
            continue

        v_str = str(v).strip()
        if not v_str:
            continue

        # ✅ 1) נסה לזהות "ערכים דבוקים" ולפצל לפי ה-vocab
        split_parts = _split_concatenated_by_vocab(v_str, options)
        if split_parts:
            for sp in split_parts:
                mapped = _map_to_closest(sp, options, threshold=threshold)
                if mapped and mapped not in out:
                    out.append(mapped)
                if len(out) >= top_k:
                    return out
            continue

        # ✅ 2) אחרת — התנהגות רגילה
        mapped = _map_to_closest(v_str, options, threshold=threshold)
        if mapped and mapped not in out:
            out.append(mapped)
        if len(out) >= top_k:
            break

    return out

def _parse_discount(text: str):
    """Return (kind, value). kind in {'percent','amount'} or (None,None)."""
    t = text.strip().replace(" ", "")
    if not t:
        return None, None
    m = re.search(r"(\d+(?:\.\d+)?)%$", t)
    if m:
        return "percent", float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)(?:₪|nis|ils)?$", t, flags=re.IGNORECASE)
    if m:
        return "amount", float(m.group(1))
    return None, None

def _format_add_summary(p: dict) -> str:
    def fmt_list(x):
        if not x:
            return "-"
        if isinstance(x, (list, tuple)):
            return ", ".join([str(i) for i in x if i is not None and str(i).strip()])
        return str(x)

    lines = []
    lines.append("🆕 פרטי בקבוק (טיוטה):")
    lines.append(f"Distillery: {p.get('distillery','-')}")
    lines.append(f"Bottle: {p.get('bottle_name','-')}")
    lines.append(f"Age: {p.get('age','-')}")
    lines.append(f"ABV: {p.get('alcohol_percentage','-')}")
    lines.append(f"Alcohol Type: {p.get('alcohol_type','-')}")
    lines.append(f"Country: {p.get('origin_country','-')}")
    lines.append(f"Region: {p.get('region','-')}")
    lines.append(f"Casks: {fmt_list(p.get('casks'))}")
    lines.append(f"Nose: {fmt_list(p.get('nose'))}")
    lines.append(f"Palette: {fmt_list(p.get('palate'))}")
    lines.append(f"Volume (ml): {p.get('orignal_volume','-')}")
    lines.append(f"Special bottling: {p.get('special')}")
    lines.append(f"Limited edition: {p.get('limited')}")
    if "price_paid" in p:
        lines.append(f"Price paid: {p.get('price_paid')}₪")
    if "price_full" in p:
        lines.append(f"Full price: {p.get('price_full')}₪")
    if "was_discounted" in p:
        lines.append(f"Was discounted: {p.get('was_discounted')}")
    if "discount_amount" in p:
        lines.append(f"Discount amount: {p.get('discount_amount')}")
    if "was_a_gift" in p:
        lines.append(f"Was a gift: {p.get('was_a_gift')}")
    return "\n".join(lines)

import re

def _extract_quoted_tokens(s: str) -> list[str]:
    """
    Extract tokens inside quotes from strings like:
    "['Red Wine Oak' 'Fruits']"  -> ["Red Wine Oak", "Fruits"]
    """
    if not s:
        return []
    # pulls both '...' and "..."
    toks = re.findall(r"'([^']+)'\s*|\"([^\"]+)\"\s*", s)
    out = []
    for a, b in toks:
        t = (a or b or "").strip()
        if t:
            out.append(t)
    return out

def _normalize_vision_list(x) -> list[str]:
    """
    Robust normalization for Gemini Vision outputs:
    - list of weird strings: ["['A' 'B']", "['C']"] -> ["A","B","C"]
    - string "A, B, C" -> ["A","B","C"]
    - string "['A', 'B']" -> ["A","B"]
    - plain string without separators -> ["that string"]
    """
    if x is None:
        return []

    # If already list/tuple -> flatten
    if isinstance(x, (list, tuple)):
        out = []
        for it in x:
            out.extend(_normalize_vision_list(it))
        return out

    s = str(x).strip()
    if not s:
        return []

    # Case: looks like bracketed list but missing commas: "['A' 'B']"
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        quoted = _extract_quoted_tokens(s)
        if quoted:
            return quoted
        # fallback: split inside brackets by commas/newlines/semicolons
        inner = s.strip("[]()").strip()
        parts = re.split(r"[,;\n]+", inner)
        return [p.strip().strip("'").strip('"') for p in parts if p.strip().strip("'").strip('"')]

    # Normal comma/semicolon/newline separated
    if any(sep in s for sep in [",", ";", "\n"]):
        parts = re.split(r"[,;\n]+", s)
        return [p.strip() for p in parts if p.strip()]

    # Plain single token
    return [s]

def sanitize_scan_raw(scan: dict) -> dict:
    """
    Fix Gemini JSON inconsistencies BEFORE controlled vocab mapping.
    Ensures: casks/nose/palate are list[str] clean.
    Also tolerates key mismatch: 'palette' vs 'palate'.
    """
    scan = dict(scan or {})

    # handle palate key mismatch
    if scan.get("palate") is None and scan.get("palette") is not None:
        scan["palate"] = scan.get("palette")

    scan["casks"] = _normalize_vision_list(scan.get("casks"))
    scan["nose"] = _normalize_vision_list(scan.get("nose"))
    scan["palate"] = _normalize_vision_list(scan.get("palate"))

    return scan

def _gemini_label_scan(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """Gemini Vision -> strict JSON. Uses same prompt style as the Dash AI Scan."""
    prompt = """
You are an expert Whisky Sommelier and Data Extraction Specialist.
Your task is to analyze a whisky bottle label image and extract technical data into a strict JSON format.

### PHASE 1: DATA EXTRACTION & CALCULATION (INTERNAL LOGIC)
1. Scan for dates: Distilled and Bottled. If no age statement exists but dates do, calculate age.
2. Identify distillery and origin. Recognize logos (e.g. 'M&H' -> 'Milk & Honey Distillery', Israel).
3. Analyze cask type.
4. Infer sensory profile: if missing, generate accurate keywords based on cask + distillery + climate.

### PHASE 2: STRICT OPERATIONAL PROTOCOLS
- Exact match only. Do not conflate versions.
- If unknown and cannot be inferred, return null.
- Volumes in ml.
- limited: true if 'Single Cask' or 'Small Batch'.
- special: true if 'Distillery Exclusive' or 'Single Cask'.

### PHASE 3: OUTPUT SCHEMA
Return a single JSON object only.
{
  "bottle_name": string,
  "distillery": string,
  "age": number,
  "alcohol_percentage": number,
  "alcohol_type": string,
  "origin_country": string,
  "region": string,
  "casks": string[],
  "nose": string[],
  "palate": string[],
  "orignal_volume": number,
  "limited": boolean,
  "special": boolean,
  "confidence": number
}
"""
    resp = ai_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt
        ],
        config={
            "response_mime_type": "application/json",
            "temperature": 0.1
        }
    )
    clean_text = (resp.text or "").strip()
    if clean_text.startswith('```json'):
        clean_text = clean_text[7:-3]
    elif clean_text.startswith('```'):
        clean_text = clean_text[3:-3]
    return json.loads(clean_text)

def _apply_controlled_vocab(scan: dict, active_df: pd.DataFrame) -> dict:
    """Map extracted strings to your existing vocab using Levenshtein."""
    out = dict(scan or {})

    # options
    alcohol_type_opts = _unique_from_scalar_col(active_df, "alcohol_type")
    country_opts = _unique_from_scalar_col(active_df, "origin_country")
    region_opts = _unique_from_scalar_col(active_df, "region")
    casks_opts = _unique_from_array_col(active_df, "casks_aged_in")
    nose_opts = _unique_from_array_col(active_df, "nose")
    pal_opts = _unique_from_array_col(active_df, "palette")

    out["alcohol_type"] = _map_to_closest(out.get("alcohol_type"), alcohol_type_opts)
    out["origin_country"] = _map_to_closest(out.get("origin_country"), country_opts)
    out["region"] = _map_to_closest(out.get("region"), region_opts)

    out["casks_aged_in"] = _map_list_to_options(normalize_to_list(out.get("casks")), casks_opts)
    out["nose"] = _map_list_to_options(normalize_to_list(out.get("nose")), nose_opts)
    out["palette"] = _map_list_to_options(normalize_to_list(out.get("palate")), pal_opts)

    # normalize booleans
    out["special_bottling"] = bool(out.get("special"))
    out["limited_edition"] = bool(out.get("limited"))

    # keep naming consistent with main table
    out["bottle_name"] = out.get("bottle_name")
    out["distillery"] = out.get("distillery")

    return out

def _get_next_bottle_id() -> int:
    q = f"SELECT COALESCE(MAX(bottle_id), 0) AS mx FROM `{TABLE_REF}`"
    r = list(bq_client.query(q).result())[0]
    return int(r.get("mx") or 0) + 1

def insert_new_bottle_from_payload(p: dict) -> int:
    """Insert into my_whisky_collection + initial history row (like Dash). Returns new bottle_id."""
    new_id = _get_next_bottle_id()

    val_name = _sql_str_or_null(p.get("bottle_name"))
    val_dist = _sql_str_or_null(p.get("distillery"))
    val_type = _sql_str_or_null(p.get("alcohol_type"))
    val_country = _sql_str_or_null(p.get("origin_country"))
    val_region = _sql_str_or_null(p.get("region"))

    val_casks = sql_array(normalize_to_list(p.get("casks_aged_in")))
    val_nose  = sql_array(normalize_to_list(p.get("nose")))
    val_pal   = sql_array(normalize_to_list(p.get("palette")))

    age = p.get("age")
    abv = p.get("alcohol_percentage")
# --- force INT for BigQuery INT64 column (orignal_volume) ---
    try:
        vol = int(round(float(p.get("orignal_volume") or 700)))
    except Exception:
        vol = 700

    price_full = p.get("price_full")
    price_paid = p.get("price_paid")
    was_discounted = bool(p.get("was_discounted"))
    discount_amount = p.get("discount_amount")

    special = bool(p.get("special_bottling"))
    limited = bool(p.get("limited_edition"))
    gift = bool(p.get("was_a_gift"))

    purch = datetime.now().date().isoformat()

    sql = f"""
    BEGIN
    DECLARE ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP();

    INSERT INTO `{TABLE_REF}`
    (bottle_id, bottle_name, distillery, alcohol_type, origin_country, region,
    casks_aged_in, nose, palette, age, alcohol_percentage, price,
    limited_edition, special_bottling, was_a_gift, stock_status_per, full_or_empy,
    orignal_volume, date_of_purchase, opening_date, bottle_counter,
    was_discounted, discount_amount, discounted_price, time_of_registration,
    updating_time)
    VALUES (
        {new_id}, {val_name}, {val_dist}, {val_type}, {val_country}, {val_region},
        {val_casks}, {val_nose}, {val_pal},
        {age if age is not None else 'NULL'},
        {abv if abv is not None else 'NULL'},
        {price_full if price_full is not None else 'NULL'},
        {str(limited).lower()}, {str(special).lower()}, {str(gift).lower()},
        100, false,
        {vol}, '{purch}',
        NULL, 1,
        {str(was_discounted).lower()},
        {discount_amount if discount_amount is not None else 'NULL'},
        {price_paid if price_paid is not None else 'NULL'},
        ts,
        ts
    );

    INSERT INTO `{HISTORY_TABLE_REF}`
    (update_id, bottle_id, bottle_name, stock_status_per, update_time, nose, palette, alc_pre)
    VALUES (
        (SELECT COALESCE(MAX(update_id), 0) + 1 FROM `{HISTORY_TABLE_REF}`),
        {new_id}, {val_name}, 100, ts,
        {val_nose}, {val_pal},
        {abv if abv is not None else 'NULL'}
    );
    END;
    """

    bq_client.query(sql).result()
    global CACHE_DATA
    CACHE_DATA["df"] = None
    return new_id

# ==========================================
# Fuzzy matchers (distillery / bottle)
# ==========================================
def find_best_distillery_match(search_term: str, active_df: pd.DataFrame):
    """
    Returns dict:
      {
        "best": <distillery str or None>,
        "score": float (0..1),
        "candidates": [ {"distillery": str, "score": float}, ... up to 5]
      }
    """
    term = _normalize_text(search_term)
    dists = (
        active_df["distillery"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    scored = []
    for d in dists:
        s = _similarity_ratio(term, d)

        # bonus if substring match
        d_n = _normalize_text(d)
        if term and (term in d_n or d_n in term):
            s = max(s, 0.92)

        # abbreviation / initialism match (e.g., m&h -> Milk & Honey)
        term_letters = _letters_only(term)
        if term_letters and len(term_letters) <= 4:
            if term_letters == _initialism(d):
                s = max(s, 0.95)

        scored.append((d, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    candidates = [{"distillery": x[0], "score": round(x[1], 3)} for x in scored[:5]]
    best = candidates[0]["distillery"] if candidates else None
    best_score = candidates[0]["score"] if candidates else 0.0
    return {"best": best, "score": best_score, "candidates": candidates}

def find_best_bottle_match(search_term: str, active_df: pd.DataFrame):
    """
    Returns dict:
      {
        "best_name": full_name,
        "bottle_id": int,
        "score": float,
        "candidates": [ {"full_name":..., "bottle_id":..., "score":...}, ... up to 5]
      }
    """
    term = _normalize_text(search_term)
    scored = []
    for _, r in active_df.iterrows():
        full = str(r.get("full_name", "")).strip()
        if not full:
            continue
        s = _similarity_ratio(term, full)
        full_n = _normalize_text(full)
        if term and (term in full_n or full_n in term):
            s = max(s, 0.93)
        scored.append((full, int(r["bottle_id"]), s))

    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:5]
    candidates = [{"full_name": a, "bottle_id": b, "score": round(c, 3)} for a, b, c in top]
    if not candidates:
        return {"best_name": None, "bottle_id": None, "score": 0.0, "candidates": []}
    return {"best_name": candidates[0]["full_name"], "bottle_id": candidates[0]["bottle_id"], "score": candidates[0]["score"], "candidates": candidates}

# ==========================================
# Update execution
# ==========================================
def execute_drink_update(bottle_id: int, amount_ml: int, inventory_dict: dict):
    if bottle_id not in inventory_dict:
        return False, "❌ לא מצאתי את ה-ID הזה במלאי."

    b_data = inventory_dict[bottle_id]
    vol = b_data["vol"] or 700
    drank_per = (amount_ml / vol) * 100
    new_stock_per = round(max(b_data["stock"] - drank_per, 0), 2)
    is_empty = "true" if new_stock_per == 0 else "false"

    # currently we only update stock here; flavor updates can be added later
    f_nose = b_data["old_nose"]
    f_palette = b_data["old_palette"]
    f_abv = b_data["old_abv"]

    sql = f"""
    BEGIN
        UPDATE `{TABLE_REF}`
        SET stock_status_per = {new_stock_per},
            full_or_empy = {is_empty},
            updating_time = CURRENT_TIMESTAMP(),
            nose = {sql_array(normalize_to_list(f_nose))},
            palette = {sql_array(normalize_to_list(f_palette))},
            alcohol_percentage = {f_abv}
        WHERE bottle_id = {bottle_id};

        INSERT INTO `{HISTORY_TABLE_REF}`
          (update_id, bottle_id, bottle_name, stock_status_per, update_time, drams_counter, nose, palette, alc_pre)
        VALUES (
          (SELECT COALESCE(MAX(update_id), 0) + 1 FROM `{HISTORY_TABLE_REF}`),
          {bottle_id},
          '{_escape_sql_str(b_data["name"])}',
          {new_stock_per},
          CURRENT_TIMESTAMP(),
          {round(amount_ml / 30)},
          {sql_array(f_nose)},
          {sql_array(f_palette)},
          {f_abv}
        );
    END;
    """
    bq_client.query(sql).result()

    # invalidate cache
    global CACHE_DATA
    CACHE_DATA["df"] = None

    return True, f"✅ עדכון בוצע!\n🥃 {b_data['name']}\n📉 ירד ל-{new_stock_per}% (הפחתה של ~{round(drank_per, 2)}%)"


# ==========================================
# Gemini "Planner" (flexible NL -> JSON plan)
# ==========================================
def _strip_json(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    # remove common code fences
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()

def gemini_make_plan(user_text: str, active_df: pd.DataFrame) -> dict | None:
    """
    Use Gemini ONLY to translate free text -> a strict JSON "plan" we can execute locally.
    This keeps execution deterministic + safe (we do NOT execute arbitrary SQL from Gemini).

    Returned plan example:
    {
      "intent": "popular",
      "scope": {"type": "all" | "distillery" | "bottle", "name": "<free text or null>"},
      "metric": "avg_consumption_vol_per_day",
      "top_n": 1
    }
    """
    try:
        # Keep prompt compact to reduce latency + hallucinations
        dists = (
            active_df["distillery"].dropna().astype(str).unique().tolist()
            if active_df is not None and not active_df.empty and "distillery" in active_df.columns
            else []
        )
        # give a few examples to anchor the model
        dist_examples = ", ".join(sorted(dists)[:25])

        system = (
            "You are a parser that converts user messages into a STRICT JSON plan.\n"
            "Return JSON ONLY (no markdown, no explanation).\n"
            "Allowed intents: popular, stock, count, recommend_now, recommend_soon, oxidized, update, unknown.\n"
            "Scope types: all, distillery, bottle.\n"
            "When user asks 'most popular' (פופולרי) WITH 'של <X>' / 'of <X>' / 'מבין <X>', "
            "set scope.type='distillery' and scope.name to the mentioned distillery.\n"
            "If user asks global 'most popular' without a scope, scope.type='all'.\n"
            "If user asks 'how much is left/כמה נשאר' for a bottle, intent='stock' and scope.type='bottle' with scope.name.\n"
            "Metric field should be one of: avg_consumption_vol_per_day, Best_Before, est_consumption_date, predicted_finish_date, stock_status_per.\n"
            f"Known distilleries (examples): {dist_examples}\n"
        )

        user = (
            f"Message: {user_text}\n\n"
            "Return JSON in this schema:\n"
            "{"
            "\"intent\":\"...\","
            "\"scope\":{\"type\":\"all|distillery|bottle\",\"name\":null|\"...\"},"
            "\"metric\":null|\"avg_consumption_vol_per_day|Best_Before|est_consumption_date|predicted_finish_date|stock_status_per\","
            "\"top_n\":1"
            "}"
        )

        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            system_instruction=system
        )
        conversation = [types.Content(role="user", parts=[types.Part.from_text(text=user)])]
        resp = ai_client.models.generate_content(model="gemini-2.0-flash", contents=conversation, config=config)
        raw = _strip_json(getattr(resp, "text", "") or "")
        if not raw:
            return None

        # sometimes model returns extra text; extract first {...}
        if not raw.lstrip().startswith("{"):
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                raw = m.group(0)

        plan = json.loads(raw)
        if not isinstance(plan, dict):
            return None
        # minimal validation
        if "intent" not in plan:
            return None
        return plan
    except Exception as e:
        logging.warning(f"Gemini planner failed: {e}")
        return None

# ---- DF schema context ----
def build_df_schema_context(df: pd.DataFrame, max_examples: int = 12) -> dict:
    ctx = {"columns": []}
    if df is None or df.empty:
        return ctx

    for col in df.columns:
        ser = df[col]
        dtype = str(ser.dtype)
        entry = {"name": col, "dtype": dtype}

        nonnull = ser.dropna()

        # --------------------------
        # TEXT (object)
        # --------------------------
        if dtype == "object":
            vals = (
                nonnull
                .astype(str)
                .map(lambda x: x.strip())
                .loc[lambda s: s != ""]
                .unique()
                .tolist()
            )
            if vals:
                entry["examples"] = vals[:max_examples]

        # --------------------------
        # NUMERIC (int/float/bool)
        # --------------------------
        elif pd.api.types.is_numeric_dtype(ser):
            nums = pd.to_numeric(nonnull, errors="coerce").dropna()
            if not nums.empty:
                # examples
                ex = nums.unique().tolist()[:max_examples]
                entry["examples"] = ex

                # min/max (מאוד עוזר ל-Gemini להבין שזה ABV/price/etc)
                try:
                    entry["min"] = float(nums.min())
                    entry["max"] = float(nums.max())
                except Exception:
                    pass

        # --------------------------
        # DATETIME
        # --------------------------
        elif pd.api.types.is_datetime64_any_dtype(ser):
            # stringify a few values
            vals = nonnull.astype(str).unique().tolist()
            if vals:
                entry["examples"] = vals[:max_examples]

        ctx["columns"].append(entry)

    return ctx

# ==========================================
# Gemini Hard Fallback (free text answer)
# ==========================================
async def gemini_fallback_answer(user_text: str, df: pd.DataFrame) -> str:
    """
    FINAL fallback.
    Used ONLY if:
    - No deterministic intent matched
    - No df_query plan worked
    - No planner plan worked

    Gemini gets schema context + light sample and answers naturally.
    """

    try:
        schema = build_df_schema_context(df)
        sample_rows = df.head(5).to_dict(orient="records") if df is not None else []

        system = (
            "You are an intelligent whisky collection assistant.\n"
            "You have access to the user's whisky collection schema and sample rows.\n"
            "If the question relates to the collection, base your answer on the provided schema/sample.\n"
            "If it's general whisky knowledge, answer normally.\n"
            "Be concise and accurate.\n"
        )

        payload = {
            "user_message": user_text,
            "schema": schema,
            "sample_rows": sample_rows
        }

        config = types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=system
        )

        resp = ai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=json.dumps(payload, ensure_ascii=False))]
                )
            ],
            config=config
        )

        return (getattr(resp, "text", "") or "").strip()

    except Exception as e:
        logging.warning(f"Gemini fallback failed: {e}")
        return "לא הצלחתי להבין לגמרי את הבקשה. תנסה לנסח אחרת 🙏"    

import json
import re
import pandas as pd




# ---- Gemini Router: decides df_query vs intent vs smalltalk ----
def gemini_route(user_text: str, df: pd.DataFrame) -> dict | None:
    try:
        schema = build_df_schema_context(df)

        system = (
            "You convert user questions into a STRICT JSON query plan over a pandas DataFrame.\n"
            "Return JSON ONLY.\n"
            "You MUST use only columns that exist in the provided schema.\n"
            "Never write SQL.\n"
            "When the user asks how many bottles (or Hebrew equivalents like כמה בקבוקי/כמה בקבוקים), prefer counting UNIQUE bottles using nunique on bottle_id if that column exists.\n"
            "Allowed ops: eq, ne, lt, lte, gt, gte, contains, in, is_null, not_null.\n"
            "Allowed agg funcs: count, nunique, sum, avg, min, max.\n"

            # ✅ ADD THIS BLOCK
            "IMPORTANT DOMAIN RULE:\n"
            "Column final_smoky_sweet_score represents a SWEET ↔ SMOKY scale.\n"
            "Scale definition:\n"
            "- LOWER values = MORE SWEET / LESS SMOKY\n"
            "- HIGHER values = MORE SMOKY / LESS SWEET\n"

            "Interpret user intent semantically, including Hebrew terms.\n"

            "Sorting rules:\n"
            "- 'most sweet', 'הכי מתוק', 'מתיקות גבוהה', 'עדין', 'לא מעושן', 'פירותי ועדין' -> order_by final_smoky_sweet_score direction asc\n"
            "- 'most smoky', 'הכי מעושן', 'כבד', 'עשן חזק', 'עשן קיצוני', 'peat heavy' -> order_by final_smoky_sweet_score direction desc\n"
            "- 'least smoky', 'הכי פחות מעושן' -> direction asc\n"
            "- 'least sweet', 'הכי פחות מתוק' -> direction desc\n"

            "If the question implies EXTREME (most / least), always apply sorting + limit 1.\n"
            "Never reverse this scale logic.\n"
            "If the user asks about sweet/smoky and final_smoky_sweet_score exists, prefer using it for ordering.\n"
            "Limit must be between 1 and 50.\n"
        )

        user_payload = {
            "message": user_text,
            "schema": schema,
            "allowed_intents": ["update", "count", "recommend_now", "recommend_soon", "stock"],
            "df_plan_schema": {
                "action": "df_query",
                "select": ["<col>", "..."],
                "filters": [{"col": "<col>", "op": "eq|ne|lt|lte|gt|gte|contains|in|is_null|not_null", "value": "<any or null>"}],
                "group_by": ["<col>", "..."],
                "aggregations": [{"col": "<col|*>", "func": "count|nunique|sum|avg|min|max", "as": "<alias>"}],
                "order_by": [{"col": "<col>", "direction": "asc|desc"}],
                "limit": 10
            },
            "output_schema": {
                "route": "df_query|intent|smalltalk",
                "intent": "update|count|recommend_now|recommend_soon|stock|null",
                "df_plan": "object|null",
                "confidence": 0.0,
                "need_clarification": False,
                "clarifying_question": ""
            }
        }

        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            system_instruction=system
        )

        resp = ai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=json.dumps(user_payload, ensure_ascii=False))]
                )
            ],
            config=config
        )

        raw = (getattr(resp, "text", "") or "").strip()
        if not raw:
            return None

        # safety: extract JSON object if wrapped
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        raw = m.group(0) if m else raw

        out = json.loads(raw)
        return out if isinstance(out, dict) else None

    except Exception as e:
        logging.warning(f"gemini_route failed: {e}")
        return None


# ---- DF Query Executor (deterministic) ----
_ALLOWED_OPS = {"eq","ne","lt","lte","gt","gte","contains","in","is_null","not_null"}
_ALLOWED_AGG = {"count","nunique","sum","avg","min","max"}

def execute_df_query_plan(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    if df is None:
        raise ValueError("DF is None")
    if not isinstance(plan, dict) or plan.get("action") != "df_query":
        raise ValueError("Invalid df_query plan")

    cols = set(df.columns.tolist())

    limit = int(plan.get("limit") or 10)
    limit = max(1, min(limit, 50))

    sub = df.copy()

    # filters
    filters_ = plan.get("filters") or []
    if isinstance(filters_, list):
        for f in filters_:
            if not isinstance(f, dict):
                continue
            col = f.get("col")
            op = f.get("op")
            val = f.get("value", None)

            if col not in cols or op not in _ALLOWED_OPS:
                continue

            s = sub[col]

            if op == "is_null":
                sub = sub[s.isna()]
            elif op == "not_null":
                sub = sub[s.notna()]
            elif op == "contains":
                sub = sub[s.astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "in":
                if not isinstance(val, list):
                    continue
                sub = sub[s.isin(val)]
            elif op in {"lt","lte","gt","gte"}:
                left = pd.to_numeric(s, errors="coerce")
                right = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
                if pd.isna(right):
                    continue
                if op == "lt":
                    sub = sub[left < right]
                elif op == "lte":
                    sub = sub[left <= right]
                elif op == "gt":
                    sub = sub[left > right]
                else:
                    sub = sub[left >= right]
            else:
                if op == "eq":
                    sub = sub[s.astype(str) == str(val)]
                elif op == "ne":
                    sub = sub[s.astype(str) != str(val)]

    # groupby + aggregations
    group_by = plan.get("group_by") or []
    aggs = plan.get("aggregations") or []
    if isinstance(group_by, list) and isinstance(aggs, list) and group_by and aggs:
        gb_cols = [c for c in group_by if isinstance(c, str) and c in cols]

        named_aggs = {}
        for a in aggs:
            if not isinstance(a, dict):
                continue
            func = a.get("func")
            col = a.get("col")
            alias = a.get("as") or f"{func}_{col}"

            if func not in _ALLOWED_AGG:
                continue

            if col == "*" and func == "count":
                # count rows: choose any column
                any_col = next(iter(cols))
                named_aggs[alias] = (any_col, "count")
            elif col in cols:
                ffunc = "mean" if func == "avg" else func
                named_aggs[alias] = (col, ffunc)

        if gb_cols and named_aggs:
            sub = sub.groupby(gb_cols, dropna=False).agg(**named_aggs).reset_index()

    # select
    select = plan.get("select") or []
    if isinstance(select, list):
        select = [c for c in select if isinstance(c, str) and c in sub.columns]
        if select:
            sub = sub[select]

    # order_by
    order_by = plan.get("order_by") or []
    if isinstance(order_by, list) and order_by:
        ob = order_by[0] if order_by else {}
        c = ob.get("col")
        d = (ob.get("direction") or "asc").lower()
        if c in sub.columns:
            sub = sub.sort_values(c, ascending=(d != "desc"))

    return sub.head(limit)


def _df_to_telegram_text(df: pd.DataFrame, max_chars: int = 3500) -> str:
    """Compact, safe formatting for Telegram."""
    if df is None:
        return ""
    try:
        if df.empty:
            return "לא מצאתי תוצאות לפי הבקשה."

        # If it's a single value -> return cleanly
        if df.shape[0] == 1 and df.shape[1] == 1:
            col = str(df.columns[0])
            val = df.iloc[0, 0]
            if pd.isna(val):
                return f"{col}: -"
            return f"{col}: {val}"

        # If single row -> key/value list
        if df.shape[0] == 1 and df.shape[1] <= 8:
            r = df.iloc[0].to_dict()
            lines = []
            for k, v in r.items():
                vv = "-" if pd.isna(v) else v
                lines.append(f"{k}: {vv}")
            out = "\n".join(lines)
        else:
            out = df.to_string(index=False)

        out = out.strip()
        if len(out) > max_chars:
            out = out[: max_chars - 20].rstrip() + "\n... (truncated)"
        return out
    except Exception:
        try:
            out = df.to_string(index=False)
            if len(out) > max_chars:
                out = out[: max_chars - 20].rstrip() + "\n... (truncated)"
            return out
        except Exception:
            return "לא הצלחתי להציג את התוצאה."

# ===========================
# Rule-based inventory Q&A (fast, deterministic)
# ===========================

_HEB_WINE = ("יין", "יינות")
_HEB_WHISKY = ("וויסקי", "ויסקי")
_HEB_RED = ("אדום", "אדומה")
_HEB_WHITE = ("לבן", "לבנה")

def _text_has_any(t: str, words) -> bool:
    tt = _normalize_text(t)
    return any(_normalize_text(w) in tt for w in words)

def _is_count_bottles_question(t: str) -> bool:
    tt = _normalize_text(t)
    return ("כמה" in tt) and ("בקבוק" in tt or "בקבוקים" in tt)

def _is_percent_of_total_question(t: str) -> bool:
    tt = _normalize_text(t)
    # covers: "מה אחוז X מכלל האלכוהול", "מה %", "מה היחס", "מתוך סה"כ"
    return ("אחוז" in tt) or ("%" in t) or ("מכלל" in tt) or ("מתוך" in tt) or ("סהכ" in tt) or ('סה"כ' in t)

def _filter_by_alcohol_type_keywords(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "alcohol_type" not in df.columns:
        return df.iloc[0:0]
    s = df["alcohol_type"].astype(str).fillna("").str.lower()
    mask = False
    for kw in keywords:
        mask = mask | s.str.contains(kw, regex=False)
    return df[mask].copy()

def _infer_category_df(user_text: str, df: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    tt = _normalize_text(user_text)

    # Whisky / whiskey family
    if _text_has_any(tt, _HEB_WHISKY) or "whisky" in tt or "whiskey" in tt:
        # Be generous: many datasets store "Single Malt Whisky", etc.
        kws = ["whisky", "whiskey", "scotch", "bourbon", "rye", "single malt", "malt", "blended"]
        return "וויסקי", _filter_by_alcohol_type_keywords(df, kws)

    # Wine family
    if _text_has_any(tt, _HEB_WINE) or "wine" in tt:
        # red / white refinement
        if _text_has_any(tt, _HEB_RED) or "red" in tt:
            return "יין אדום", _filter_by_alcohol_type_keywords(df, ["red wine"])
        if _text_has_any(tt, _HEB_WHITE) or "white" in tt:
            return "יין לבן", _filter_by_alcohol_type_keywords(df, ["white wine"])
        # generic wine (includes dessert, sparkling, etc.)
        return "יין", _filter_by_alcohol_type_keywords(df, ["wine"])

    return None, df.iloc[0:0]

def _count_unique_bottles(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if "bottle_id" in df.columns:
        return int(df["bottle_id"].nunique(dropna=True))
    # fallback
    return int(len(df))

def rule_based_inventory_answer(user_text: str, df: pd.DataFrame) -> str | None:
    """Deterministic answers for common portfolio inventory questions.
    Returns a short Hebrew answer or None if not applicable.
    """
    if df is None or df.empty:
        return None

    if not _is_count_bottles_question(user_text) and not _is_percent_of_total_question(user_text):
        return None

    # ✅ NEW: if user asked "how many bottles of <X>" and <X> is NOT a category (wine/whisky),
    # don't answer total; let the distillery matcher handle it.
    ent = _normalize_text(_extract_entity_for_count(user_text) or "")
    if ent:
        is_whisky = ("וויסקי" in ent) or ("ויסקי" in ent) or ("whisky" in ent) or ("whiskey" in ent)
        is_wine   = ("יין" in ent) or ("wine" in ent)
        is_generic_total = ent in ("אלכוהול", "alcohol", "הכל", "כולם", "all", "total")

        if (not is_whisky) and (not is_wine) and (not is_generic_total):
            return None

    label, sub = _infer_category_df(user_text, df)

    # Total bottles question: "כמה בקבוקים יש לי" without specifying category
    tt = _normalize_text(user_text)
    if _is_count_bottles_question(user_text) and (label is None):
        total = _count_unique_bottles(df)
        return f"יש לך **{total}** בקבוקים בסך הכל."

    # Category count
    if _is_count_bottles_question(user_text) and label is not None:
        cnt = _count_unique_bottles(sub)
        return f"יש לך **{cnt}** בקבוקי {label}."

    # Percent of total
    if _is_percent_of_total_question(user_text) and label is not None:
        total = _count_unique_bottles(df)
        cnt = _count_unique_bottles(sub)
        if total <= 0:
            return "אין לי מספיק נתונים כדי לחשב אחוזים כרגע."
        pct = round((cnt / total) * 100.0, 1)
        return f"בקבוקי {label} הם **{pct}%** מכלל האלכוהול שלך (**{cnt}/{total}** בקבוקים)."

    return None


def gemini_make_df_query_plan(user_text: str, df: pd.DataFrame, focus: dict | None = None) -> dict | None:
    try:
        schema = build_df_schema_context(df)
        COLUMN_GLOSSARY = {
            "avg_consumption_vol_per_day": [
                "ממוצע שתייה", "כמה אני שותה ממנו בממוצע", "בממוצע", "popular", "פופולריות", "כמה נשתה", "צריכה יומית", "ml ליום"
            ],
            "latest_consumption_time": [
                "  ממנו לאחרונה", "מתי  לאחרונה", "פעם אחרונה", "last drink", "אחרון", "תאריך שתייה אחרון"
            ],
            "predicted_finish_date": [
                "מתי הוא צפוי להיגמר", "מתי ייגמר", "יסתיים", "finish date", "מתי נגמר"
            ],
            "est_consumption_date": [
                "מתי לשתות ממנו", "מתי כדאי לשתות", "המלצה מתי לשתות", "recommend date", "דראם הבא מתי", "מה לשתות עכשיו"
            ],
            "Best_Before": [
                "עד מתי כדאי לשתות אותו", "עד מתי כדאי לשתות ממנו", "best before", "תוקף", "מומלץ עד", "לפני שיתחמצן"
            ],
            "orignal_volume": [
                "מה הנפח שלו", "כמה ml", "נפח", "volume", "700", "1000", "גודל בקבוק"
            ],
            # בונוסים שימושיים:
            "current_status": ["כמה נשאר", "אחוז נשאר", "remaining", "left", "סטוק", "מלאי"],
            "alcohol_percentage": ["אלכוהול אחוז", "abv", "strength", "אלכוהול"],
            "age": ["גיל", "בן כמה", "age statement"],
            "price": ["מחיר", "כמה עלה", "עלות", "₪"],
            "casks_aged_in": ["חבית", "חביות", "cask", "aged in"],
            "nose": ["nose", "נוז", "ארומות", "ריח"],
            "palette": ["palate", "פלטה", "טעמים", "taste"]
        }
        
        
        system = (
            "You convert user questions into a STRICT JSON query plan over a pandas DataFrame.\n"
            "Return JSON ONLY.\n"
            "You MUST use only columns that exist in the provided schema.\n"
            "Never write SQL.\n\n"

            "COLUMN SELECTION RULES (VERY IMPORTANT):\n"
            "You are given a COLUMN_GLOSSARY that maps user intents/synonyms to specific columns.\n"
            "When the question matches a glossary entry, you MUST use the mapped column.\n"
            "Prefer the most specific column available (e.g., predicted_finish_date vs est_consumption_date).\n"
            "If the user asks a direct 'field question', put that column in `select`.\n"
            "If the user asks 'how many' or asks for an aggregate, use `aggregations`.\n\n"
            
            "PRIORITY RULE (CRITICAL):\n"
            "- If the question contains taste/aroma/cask intent, NEVER treat it as a bottle-name matching task.\n"
            "- Do NOT propose close name candidates. Build a DataFrame query plan instead.\n"            
            "TEXT SEARCH RULES (CRITICAL):\n"
            
            "- If the user asks for bottles by flavors/tastes (Hebrew: 'בטעם', 'בטעמי', 'טעמים', 'טעמי', 'שוקולד', 'קפה', etc.),\n"
            "  you MUST filter using: palette contains <keyword(s)>.\n"
            "- If the user asks for bottles by aromas/smells (Hebrew: 'ארומה', 'ארומות', 'ריח', 'נוז'),\n"
            "  you MUST filter using: nose contains <keyword(s)>.\n"
            "- If the user asks for bottles by cask/aging (Hebrew: 'חבית', 'חביות', 'יישון', 'שרי', 'בשרי', or English: 'sherry', 'cask'),\n"
            "  you MUST filter using: casks_aged_in contains <keyword(s)>.\n"
            "\n"
            "KEYWORD EXTRACTION RULES:\n"
            "- Extract the requested keywords from the user message (e.g., 'שוקולד וקפה' -> ['שוקולד','קפה']).\n"
            "- If the user uses 'או' -> use OR logic: multiple filters that broaden results.\n"
            "- If the user uses 'וגם'/'שניהם' -> use AND logic: multiple filters that narrow results.\n"
            "\n"
            "SHERRY EXPANSION (IMPORTANT):\n"
            "- If the user asks for 'שרי' or 'sherry', also consider matching common sherry terms in casks_aged_in:\n"
            "  ['sherry','oloroso','px','pedro ximenez','ximenez','ximénez'].\n"
            "\n"
            "LISTING RULE:\n"
            "- For list questions ('איזה וויסקי', 'תן לי כל הבקבוקים', 'show me bottles'), select minimal identity columns:\n"
            "  distillery, bottle_name, bottle_id (if exists). Limit 10-20.\n"
        
            "OUTPUT CONSTRAINTS (CRITICAL):\n"
            "Never use select=['*'] unless the user explicitly asks for 'all details / everything / כל הפרטים / כל הנתונים'.\n"
            "By default, select ONLY the minimal columns needed to answer the question (usually 1-3 columns).\n"
            "If the user asks a single metric (ABV, age, price, volume, last drink date, predicted finish, best before, avg consumption), select exactly that column.\n\n"

            "FOCUS RULES (VERY IMPORTANT):\n"
            "If focus.bottle_id is provided OR the user uses pronouns like 'שלו/הוא/זה/בו', "
            "you MUST add a filter on bottle_id == focus.bottle_id. "
            "If bottle_id is missing from schema, filter by full_name contains focus.full_name.\n\n"

            "FREE-TEXT FLAVOR EXTRACTION (CRITICAL):\n"
            "- When the user asks: 'איזה בקבוקים יש לי בטעמי X,Y,Z' or similar,\n"
            "  treat X,Y,Z as FREE-TEXT keywords (do NOT require predefined lists).\n"
            "- Extract keywords as the terms that appear AFTER these markers:\n"
            "  ['בטעמי', 'בטעם', 'טעמים של', 'עם טעמים של'].\n"
            "- Split keywords by separators: comma ',', Hebrew 'ו' (and), '/', '&', and the word 'או'.\n"
            "- Remove generic words like: ['וויסקי','בקבוקים','בקבוקי','טעמים','בטעם','בטעמי','של','עם','יש','לי'].\n"
            "- Then build filters on column 'palette' using op='contains' for each keyword.\n"
            "- If the user uses 'או' -> set filter_logic='OR'.\n"
            "- If the user uses 'וגם' or 'שניהם' -> set filter_logic='AND'.\n"
            
            "Counting rule:\n"
            "When the user asks how many bottles (or Hebrew like כמה בקבוקי/כמה בקבוקים), "
            "prefer counting UNIQUE bottles using nunique on bottle_id if that column exists.\n\n"

            "Allowed ops: eq, ne, lt, lte, gt, gte, contains, in, is_null, not_null.\n"
            "Allowed agg funcs: count, nunique, sum, avg, min, max.\n"
            "Limit must be between 1 and 50.\n"

            "IDENTITY COLUMNS:\n"
            "- When returning multiple bottles, always include bottle_id (if exists), bottle_name, distillery.\n"
            "- Do NOT select large text columns unless needed.\n"
        )

        user = {
            "message": user_text,
            "focus": focus,
            "schema": schema,
            "column_glossary": COLUMN_GLOSSARY,   # <-- חדש ומאוד חשוב
            "output_schema": {
                "action": "df_query",
                "need_clarification": False,
                "clarifying_question": "",
                "select": ["<col>", "..."],
                "filter_logic": "AND|OR",
                "filters": [{"col": "<col>", "op": "<op>", "value": "<any or null>"}],
                "group_by": ["<col>", "..."],
                "aggregations": [{"col": "<col|*>", "func": "<agg>", "as": "<alias>"}],
                "order_by": [{"col": "<col>", "direction": "asc|desc"}],
                "limit": 10
            }
        }

        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            system_instruction=system
        )
        resp = ai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=json.dumps(user, ensure_ascii=False))])],
            config=config
        )

        raw = _strip_json(getattr(resp, "text", "") or "")
        if not raw:
            return None
        if not raw.lstrip().startswith("{"):
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            raw = m.group(0) if m else raw

        plan = json.loads(raw)
        if not isinstance(plan, dict) or plan.get("action") != "df_query":
            return None
        return plan
    except Exception as e:
        logging.warning(f"Gemini df_query planner failed: {e}")
        return None
    
    
def _looks_like_df_analytics_question(t: str) -> bool:
    t = _normalize_text(t)
    # אנליטיקה/חישובים/השוואות/טופים/פילוחים
    cues = [
        "כמה", "ספור", "מספר", "count",
        "ממוצע", "average", "avg",
        "אחוז", "%", "חלק", "share", "מתוך", "percent",
        "הכי", "top", "מקסימום", "מינימום", "max", "min",
        "לפי", "פילוח", "התפלגות", "distribution",
        "יותר מ", "פחות מ", "בין", "מעל", "מתחת",
        "השווה", "לעומת", "versus",
    ]
    return any(c in t for c in cues)

async def try_gemini_df_query_answer(user_text: str, df: pd.DataFrame, context) -> str | None:
    user_text = (user_text or "").strip()
    if not user_text or df is None or df.empty:
        return None

    focus = None
    bid = context.user_data.get("focus_bottle_id")
    full = context.user_data.get("focus_full_name")
    if bid:
        focus = {"bottle_id": int(bid), "full_name": full}

    df_question = _looks_like_df_analytics_question(user_text)

    # 1) Planner-first (soft)
    try:
        if not should_skip_planner(user_text):
            plan = gemini_make_df_query_plan(user_text, df, focus=focus)
            plan = enforce_palette_free_text(plan, user_text)

            if plan and isinstance(plan, dict):
                if "order_by'" in plan and "order_by" not in plan:
                    plan["order_by"] = plan.pop("order_by'")

                try:
                    plan = normalize_plan_columns(plan, df)
                except Exception:
                    pass

                res = execute_df_query_plan(df, plan)

                # ✅ return ONLY if meaningful
                if res is not None:
                    if isinstance(res, pd.DataFrame) and not res.empty:
                        return format_df_answer(res, plan)  # או _df_to_telegram_text(res)
                    if isinstance(res, pd.Series) and not res.empty:
                        return _df_to_telegram_text(res.to_frame().T)

                # ❗ אם זו שאלה אנליטית — אל תיפול לגנרי
                if df_question:
                    return None

    except Exception as e:
        logging.warning(f"DF plan execution failed: {e}")
        if df_question:
            return None

    # 2) Free-text fallback ONLY when it's NOT an analytics/DF question
    if not df_question:
        try:
            return await gemini_fallback_answer(user_text, df)
        except Exception as e:
            logging.warning(f"Gemini fallback failed: {e}")
            return None

    return None

    
_ALLOWED_OPS = {"eq","ne","lt","lte","gt","gte","contains","in","is_null","not_null"}
_ALLOWED_AGG = {"count","nunique","sum","avg","min","max"}

    

# ==========================================
# Gemini fallback (optional)
# ==========================================
def _build_gemini_config(schema_text: str):
    tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode='NONE'))
    return types.GenerateContentConfig(
        tool_config=tool_config,
        system_instruction=schema_text,
        temperature=0.1
    )

# ==========================================
# Telegram handlers
# ==========================================


def _fmt_date(x) -> str | None:
    try:
        if pd.isna(x):
            return None
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        return str(dt.date())
    except Exception:
        return None



def build_stock_reply(row: pd.Series) -> str:
    """Create a friendly stock status reply for a bottle.
    IMPORTANT: We intentionally avoid Telegram Markdown/HTML parse modes here,
    to prevent 'Can't parse entities' errors from bottle names (e.g. dots, dashes, underscores).
    """
    name = str(row.get("full_name") or row.get("bottle_name") or "").strip()

    # % left
    per = None
    try:
        per = float(row.get("stock_status_per")) if pd.notnull(row.get("stock_status_per")) else None
    except Exception:
        per = None

    # original volume (ml)
    vol = None
    try:
        vol = float(row.get("orignal_volume")) if pd.notnull(row.get("orignal_volume")) else None
    except Exception:
        vol = None

    # predicted finish date from forecast
    pfd = _fmt_date(row.get("predicted_finish_date"))

    lines = [f"🥃 {name}"]

    if per is not None:
        lines.append(f"📊 נשאר בערך: {round(per, 1)}%")
        if vol is not None:
            est_ml = (per / 100.0) * vol
            lines.append(f"🧪 שזה בערך: {round(est_ml, 1)} ml מתוך {int(vol)}ml")
    else:
        lines.append("📊 אין לי כרגע אחוז מלאי (stock_status_per) לבקבוק הזה.")

    if pfd:
        lines.append(f"📅 תאריך סיום חזוי (predicted_finish_date): {pfd}")

    return "\n".join(lines)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("pending_update", None)
    context.user_data.pop("pending_count", None)
    context.user_data.pop("pending_stock", None)

    await update.message.reply_text(
        "מוכן ✅\n"
        "אפשר לשאול למשל:\n"
        "• כמה בקבוקי גלנפידיך יש לי?\n"
        "• שתיתי 60ml Glenfiddich 15\n\n"
        "אם אני לא בטוח בשם – אציע התאמה ואבקש אישור לפני עדכון."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Stage-aware handler: can receive TEXT or PHOTO
    stage = _get_add_stage(context)
    r = None


    # If we're waiting for a label photo, accept PHOTO messages
    if stage == "await_label_photo":
        # Allow cancel even while waiting for a photo
        _t = ""
        if update.message:
            _t = update.message.text or update.message.caption or ""
        if _normalize_text(_t) in [_normalize_text(x) for x in _CANCEL_WORDS]:
            context.user_data.pop("pending_update", None)
            context.user_data.pop("pending_count", None)
            context.user_data.pop("pending_stock", None)
            _clear_add_flow(context)
            await update.message.reply_text("סבבה, ביטלתי. שלח שאלה חדשה 🙂")
            return
        if update.message and update.message.photo:
            try:
                photo = update.message.photo[-1]  # highest resolution
                tg_file = await context.bot.get_file(photo.file_id)
                bio = await tg_file.download_as_bytearray()
                image_bytes = bytes(bio)
                # Best-effort mime type
                mime_type = "image/jpeg"
                scan_raw = _gemini_label_scan(image_bytes=image_bytes, mime_type=mime_type)
                scan_raw = sanitize_scan_raw(scan_raw)          # ✅ חדש - חובה
                
                logging.info(f"SCAN NOSE normalized: {scan_raw.get('nose')}")
                logging.info(f"SCAN PALATE normalized: {scan_raw.get('palate')}")
                # Load current vocab DF for mapping
                active_df = get_all_data_as_df()
                if active_df is None:
                    active_df = pd.DataFrame()

                payload = _apply_controlled_vocab(scan_raw, active_df)

                # Store payload and move to price
                _set_add_payload(context, payload)
                _set_add_stage(context, "await_price")

                await update.message.reply_text("✅ סריקה הושלמה. הנה מה שחילצתי:")
                await update.message.reply_text(_format_add_summary(payload))
                await update.message.reply_text("מה המחיר ששילמת? (רק מספר, למשל 350)")
                return
            except Exception as e:
                logging.exception(e)
                await update.message.reply_text(f"❌ שגיאה בסריקה: {e}")
                return

        # Still waiting for photo
        await update.message.reply_text("שלח בבקשה תמונה של התווית (צילום ברור).")
        return

    user_text = (update.message.text or "").strip() if update.message else ""
    if not user_text:
        return


    # ----- HARD RULE: sweet/smoky & delicate/rich extremes -----
    df = get_all_data_as_df()
    
    
    ans = try_handle_extremes_sweet_smoky_rich_delicate(user_text, df)
    if ans:
        await update.message.reply_text(ans)
        return    
    
    
    # Global cancel
    if _normalize_text(user_text) in [_normalize_text(x) for x in _CANCEL_WORDS]:
        context.user_data.pop("pending_update", None)
        context.user_data.pop("pending_count", None)
        context.user_data.pop("pending_stock", None)
        _clear_add_flow(context)
        await update.message.reply_text("סבבה, ביטלתי את הפעולה הקודמת. שלח שאלה חדשה 🙂")

        return

    # ===========================
    # Add Bottle flow (text stages)
    # ===========================
    stage = _get_add_stage(context)

    if stage == "await_price":
        m = re.search(r"(\d+(?:\.\d+)?)", user_text.replace(",", ""))
        if not m:
            await update.message.reply_text("תן מחיר כמספר (למשל 350).")
            return
        price_paid = float(m.group(1))
        p = _get_add_payload(context)
        p["price_paid"] = round(price_paid, 2)
        _set_add_payload(context, p)
        _set_add_stage(context, "await_discount_q")
        await update.message.reply_text("הייתה הנחה? (כן/לא)")
        return

    if stage == "await_discount_q":
        t = _normalize_text(user_text)
        if any(_normalize_text(x) == t for x in _CONFIRM_NO) or t in ("לא", "no"):
            p = _get_add_payload(context)
            p["was_discounted"] = False
            p["discount_amount"] = None
            p["price_full"] = p.get("price_paid")
            _set_add_payload(context, p)
            _set_add_stage(context, "await_gift_q")
            await update.message.reply_text("האם מדובר במתנה? (כן/לא)")
            return
        if any(_normalize_text(x) == t for x in _CONFIRM_YES) or t in ("כן", "yes"):
            _set_add_stage(context, "await_discount_amount")
            await update.message.reply_text("כתוב את גובה ההנחה: למשל 10% או 50₪")
            return
        await update.message.reply_text("ענה בבקשה כן/לא לגבי הנחה.")
        return

    if stage == "await_discount_amount":
        kind, val = _parse_discount(user_text)
        if kind is None:
            await update.message.reply_text("לא הבנתי. כתוב למשל 10% או 50₪")
            return
        p = _get_add_payload(context)
        paid = float(p.get("price_paid") or 0)
        p["was_discounted"] = True
        if kind == "percent":
            pct = max(min(val, 95.0), 0.0)
            full = paid / (1.0 - pct/100.0) if pct < 100 else paid
            p["discount_amount"] = f"{pct}%"
            p["price_full"] = round(full, 2)
        else:
            amt = max(val, 0.0)
            p["discount_amount"] = f"{round(amt,2)}₪"
            p["price_full"] = round(paid + amt, 2)
        _set_add_payload(context, p)
        _set_add_stage(context, "await_gift_q")
        await update.message.reply_text("האם מדובר במתנה? (כן/לא)")
        return

    if _looks_like_add_bottle(user_text):
        context.chat_data["add_stage"] = "await_label_photo"
        await update.message.reply_text("שלח לי תמונה של התווית.")
        return

    if stage == "await_gift_q":
        t = _normalize_text(user_text)
        if any(_normalize_text(x) == t for x in _CONFIRM_YES) or t in ("כן", "yes"):
            p = _get_add_payload(context)
            p["was_a_gift"] = True
            _set_add_payload(context, p)
        elif any(_normalize_text(x) == t for x in _CONFIRM_NO) or t in ("לא", "no"):
            p = _get_add_payload(context)
            p["was_a_gift"] = False
            _set_add_payload(context, p)
        else:
            await update.message.reply_text("ענה בבקשה כן/לא לגבי מתנה.")
            return

        p = _get_add_payload(context)
        _set_add_stage(context, "await_confirm_insert")
        await update.message.reply_text("מעולה. זה הסיכום לפני הכנסה למאגר:")
        await update.message.reply_text(_format_add_summary(p))
        await update.message.reply_text("לאשר הכנסת הבקבוק? (כן/לא)")
        return

    if stage == "await_confirm_insert":
        t = _normalize_text(user_text)
        if any(_normalize_text(x) == t for x in _CONFIRM_YES) or t in ("כן", "yes"):
            try:
                p = _get_add_payload(context)
                new_id = insert_new_bottle_from_payload(p)
                _clear_add_flow(context)
                await update.message.reply_text(f"✅ הבקבוק נוסף! bottle_id={new_id}")
            except Exception as e:
                logging.exception(e)
                await update.message.reply_text(f"❌ הכנסת הבקבוק נכשלה: {e}")
            return
        if any(_normalize_text(x) == t for x in _CONFIRM_NO) or t in ("לא", "no"):
            _set_add_stage(context, "await_edit_fields")
            await update.message.reply_text(
                "סבבה. שלח תיקונים בפורמט:\n"
                "field=value, field=value\n"
                "דוגמאות: age=12, region=Islay, alcohol_percentage=46"
            )
            return
        await update.message.reply_text("ענה בבקשה כן/לא.")
        return

    if stage == "await_edit_fields":
        p = _get_add_payload(context)
        raw = user_text
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        for part in parts:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            # numeric fields
            if k in ("age", "alcohol_percentage", "orignal_volume"):
                try:
                    p[k] = float(v) if "." in v else int(v)
                except Exception:
                    p[k] = v
            elif k in ("nose", "palette", "casks_aged_in"):
                # allow comma-separated values inside
                vals = [s.strip() for s in re.split(r"[;|/]", v) if s.strip()]
                p[k] = vals
            elif k in ("limited_edition", "special_bottling", "was_a_gift", "was_discounted"):
                p[k] = _normalize_text(v) in ("כן", "yes", "true", "1")
            else:
                p[k] = v

        # re-apply vocab mapping for controlled fields
        active_df = get_all_data_as_df()
        if active_df is None:
            active_df = pd.DataFrame()
        # map scalar controlled fields
        p["alcohol_type"] = _map_to_closest(p.get("alcohol_type"), _unique_from_scalar_col(active_df, "alcohol_type"))
        p["origin_country"] = _map_to_closest(p.get("origin_country"), _unique_from_scalar_col(active_df, "origin_country"))
        p["region"] = _map_to_closest(p.get("region"), _unique_from_scalar_col(active_df, "region"))
        p["casks_aged_in"] = _map_list_to_options(p.get("casks_aged_in"), _unique_from_array_col(active_df, "casks_aged_in"))
        p["nose"] = _map_list_to_options(p.get("nose"), _unique_from_array_col(active_df, "nose"))
        p["palette"] = _map_list_to_options(p.get("palette"), _unique_from_array_col(active_df, "palette"))

        _set_add_payload(context, p)
        _set_add_stage(context, "await_confirm_insert")
        await update.message.reply_text("עודכן. זה הסיכום המעודכן:")
        await update.message.reply_text(_format_add_summary(p))
        await update.message.reply_text("לאשר הכנסת הבקבוק? (כן/לא)")
        return


    # 1) Pending disambiguation flows (COUNT / POPULAR SCOPE / STOCK)
    pending_stock = context.user_data.get("pending_stock")
    if pending_stock:
        # numeric pick (1-3)
        m_pick = re.match(r"^\s*(\d)\s*$", user_text)
        if m_pick:
            idx = int(m_pick.group(1)) - 1
            cands = pending_stock.get("candidates", [])
            if 0 <= idx < len(cands):
                chosen = cands[idx]
                df = get_all_data_as_df()
                active_df = df[df["stock_status_per"] > 0].copy()
                sub = active_df[active_df["bottle_id"] == chosen["bottle_id"]]
                if sub.empty:
                    context.user_data.pop("pending_stock", None)
                    await update.message.reply_text("לא מצאתי את הבקבוק הזה במלאי הפעיל. נסה שוב.")
                    return
                row = sub.iloc[0]
                reply = build_stock_reply(row)
                _set_focus_bottle(context, row)
                context.user_data.pop("pending_stock", None)
                await update.message.reply_text(reply)
                return

        # free-text resolve
        tries = int(pending_stock.get("tries", 0)) + 1
        pending_stock["tries"] = tries
        if tries >= 3:
            context.user_data.pop("pending_stock", None)
            await update.message.reply_text("ביטלתי כי לא הצלחנו לזהות את הבקבוק. שלח שוב עם שם מדויק יותר 🙂")
            return

        df = get_all_data_as_df()
        active_df = df[df["stock_status_per"] > 0].copy()
        match = find_best_bottle_match(user_text, active_df)
        if match.get("best_name") and float(match.get("score") or 0) >= 0.70:
            chosen_id = int(match["bottle_id"])
            sub = active_df[active_df["bottle_id"] == chosen_id]
            if sub.empty:
                context.user_data.pop("pending_stock", None)
                await update.message.reply_text("לא מצאתי את הבקבוק הזה במלאי הפעיל.")
                return
            row = sub.iloc[0]
            reply = build_stock_reply(row)
            _set_focus_bottle(context, row)
            context.user_data.pop("pending_stock", None)
            await update.message.reply_text(reply)
            return

        # still not resolved
        await update.message.reply_text("לא הצלחתי לזהות. בחר מספר 1-3 או כתוב את שם הבקבוק שוב (מזקקה + שם).")
        return

    pending_count = context.user_data.get("pending_count")
    if pending_count:
        mode = pending_count.get("mode", "count")  # "count" or "popular_scope"
        # numeric pick (1-3)
        m_pick = re.match(r"^\s*(\d)\s*$", user_text)
        if m_pick:
            idx = int(m_pick.group(1)) - 1
            cands = pending_count.get("candidates", [])
            if 0 <= idx < len(cands):
                dist = cands[idx]["distillery"]
                context.user_data.pop("pending_count", None)

                df = get_all_data_as_df()
                active_df = df[df["stock_status_per"] > 0].copy()
                sub = active_df[active_df["distillery"].astype(str) == str(dist)]

                if mode == "popular_scope":
                    # compute popular within this distillery
                    if sub.empty:
                        await update.message.reply_text("לא מצאתי בקבוקים פעילים למזקקה הזו.")
                        return
                    if "avg_consumption_vol_per_day" not in sub.columns or sub["avg_consumption_vol_per_day"].dropna().empty:
                        await update.message.reply_text("אין לי נתוני avg_consumption_vol_per_day זמינים כרגע.")
                        return
                    sub["avg_consumption_vol_per_day"] = pd.to_numeric(sub["avg_consumption_vol_per_day"], errors="coerce")
                    sub = sub[pd.notnull(sub["avg_consumption_vol_per_day"])]
                    if sub.empty:
                        await update.message.reply_text("אין ערכים תקינים של avg_consumption_vol_per_day למזקקה הזו.")
                        return
                    top = sub.sort_values("avg_consumption_vol_per_day", ascending=False).head(1).iloc[0]
                    await update.message.reply_text(
                        f"🏆 הבקבוק הכי 'פופולרי' של {dist} לפי Avg Consumption / Day הוא:\n"
                        f"🥃 {top.get('full_name')}\n"
                        f"📈 {round(float(top.get('avg_consumption_vol_per_day') or 0), 2)} ml/day"
                    )
                    return

                # normal count mode
                cnt = int(sub["bottle_id"].nunique())
                await update.message.reply_text(f"יש לך {cnt} בקבוקים פעילים של {dist}.")
                return

        # free-text resolve (user typed a name)
        tries = int(pending_count.get("tries", 0)) + 1
        pending_count["tries"] = tries
        if tries >= 3:
            context.user_data.pop("pending_count", None)
            await update.message.reply_text("ביטלתי כי לא הצלחנו לזהות את המזקקה. שלח שוב 🙂")
            return

        df = get_all_data_as_df()
        active_df = df[df["stock_status_per"] > 0].copy()
        dist_match = find_best_distillery_match(user_text, active_df)
        if dist_match.get("best") and float(dist_match.get("score") or 0) >= 0.62:
            dist = dist_match["best"]
            # reuse same flow by simulating pick
            pending_count["candidates"] = [{"distillery": dist, "score": dist_match.get("score", 0)}]
            pending_count["tries"] = 0
            context.user_data["pending_count"] = pending_count
            # call ourselves with "1" style: just answer directly
            context.user_data.pop("pending_count", None)
            sub = active_df[active_df["distillery"].astype(str) == str(dist)]
            if mode == "popular_scope":
                if "avg_consumption_vol_per_day" not in sub.columns or sub["avg_consumption_vol_per_day"].dropna().empty:
                    await update.message.reply_text("אין לי נתוני avg_consumption_vol_per_day זמינים כרגע.")
                    return
                sub["avg_consumption_vol_per_day"] = pd.to_numeric(sub["avg_consumption_vol_per_day"], errors="coerce")
                sub = sub[pd.notnull(sub["avg_consumption_vol_per_day"])]
                if sub.empty:
                    await update.message.reply_text("אין ערכים תקינים של avg_consumption_vol_per_day למזקקה הזו.")
                    return
                top = sub.sort_values("avg_consumption_vol_per_day", ascending=False).head(1).iloc[0]
                await update.message.reply_text(
                    f"🏆 הבקבוק הכי 'פופולרי' של {dist} לפי Avg Consumption / Day הוא:\n"
                    f"🥃 {top.get('full_name')}\n"
                    f"📈 {round(float(top.get('avg_consumption_vol_per_day') or 0), 2)} ml/day"
                )
                return
            cnt = int(sub["bottle_id"].nunique())
            await update.message.reply_text(f"יש לך {cnt} בקבוקים פעילים של {dist}.")
            return

        await update.message.reply_text("לא הצלחתי להבין לאיזו מזקקה התכוונת. בחר 1-3 או כתוב את השם שוב.")
        return
    
    
# 1) Pending confirmation flow
    pending = context.user_data.get("pending_update")
    if pending:
        t_norm = _normalize_text(user_text)

        # numeric pick
        m = re.match(r"^\s*(\d)\s*$", user_text)
        if m and "candidates" in pending:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(pending["candidates"]):
                chosen = pending["candidates"][idx]
                pending["bottle_id"] = chosen["bottle_id"]
                pending["full_name"] = chosen["full_name"]
                context.user_data["pending_update"] = pending
                await update.message.reply_text(
                    f"סבבה. התכוונת ל:\n🥃 {pending['full_name']}\n"
                    f"לעדכן שתייה של {pending['amount_ml']}ml? (כן/לא)"
                )
                return

        if t_norm in [_normalize_text(x) for x in _CONFIRM_YES]:
            df = get_all_data_as_df()
            active_df = df[df["stock_status_per"] > 0]

            inventory_dict = {}
            for _, r in active_df.iterrows():
                inventory_dict[int(r["bottle_id"])] = {
                    "name": r["full_name"],
                    "stock": float(r["stock_status_per"]),
                    "vol": float(r["orignal_volume"]) if pd.notnull(r.get("orignal_volume")) else 700.0,
                    "old_nose": list(r["nose"]) if isinstance(r.get("nose"), list) else [],
                    "old_palette": list(r["palette"]) if isinstance(r.get("palette"), list) else [],
                    "old_abv": float(r["alcohol_percentage"]) if pd.notnull(r.get("alcohol_percentage")) else 0.0,
                }

            ok, msg = execute_drink_update(int(pending["bottle_id"]), int(pending["amount_ml"]), inventory_dict)
            _set_focus_bottle(context, {'bottle_id': int(pending['bottle_id']), 'full_name': pending.get('full_name','')})
            context.user_data.pop("pending_update", None)
            context.user_data.pop("pending_count", None)
            context.user_data.pop("pending_stock", None)
            await update.message.reply_text(msg)
            return

        if t_norm in [_normalize_text(x) for x in _CONFIRM_NO]:
            context.user_data.pop("pending_update", None)
            context.user_data.pop("pending_count", None)
            context.user_data.pop("pending_stock", None)
            await update.message.reply_text(
                "סבבה, לא מעדכן.\n"
                "תשלח שוב את ההודעה עם שם מדויק יותר, או פשוט כתוב את שם הבקבוק."
            )
            return

        # fallback: user typed something else -> treat as new bottle term and try again quickly
        # fallback: user typed something else -> try resolving from free-text answer.
        tries = int(pending.get("tries", 0)) + 1
        pending["tries"] = tries
        if tries >= 3:
            context.user_data.pop("pending_update", None)
            await update.message.reply_text("ביטלתי את הפעולה כי לא הצלחנו לזהות את הבקבוק. שלח שוב עם מזקקה + שם/גיל 🙂")
            return
        new_term = user_text
        df = get_all_data_as_df()
        active_df = df[df["stock_status_per"] > 0]
        match = find_best_bottle_match(new_term, active_df)
        if match["best_name"] and match["score"] >= 0.70:
            candidates = match["candidates"]
            lines = [f"{i+1}. {c['full_name']} (score {c['score']})" for i, c in enumerate(candidates[:3])]
            context.user_data["pending_update"] = {
                "amount_ml": pending["amount_ml"],
                "bottle_id": match["bottle_id"],
                "full_name": match["best_name"],
                "candidates": candidates[:3],
            }
            await update.message.reply_text(
                "מצאתי התאמות אפשריות. תבחר מספר 1-3 או כתוב 'כן' כדי לבחור את הראשונה:\n" + "\n".join(lines)
            )
            return
        else:
            await update.message.reply_text("לא הצלחתי לזהות את הבקבוק. תן עוד קצת פרטים (מזקקה + שם/גיל).")
            return
        
        
    df = get_all_data_as_df()
    if df is None:
        df = pd.DataFrame()


    fast = _try_fast_portfolio_answer(user_text, df)
    if fast:
        await update.message.reply_text(fast)
        return


    # 2) Fresh data for deterministic intents
    try:
        df = get_all_data_as_df()
        active_df = df[df["stock_status_per"] > 0].copy()



        # 2) Legacy handlers (מה שעבד לך פעם)
        # - have_query / find_best_bottle_match / recommend וכו'
        # ...
        
        # ===========================
        # Bottle-specific: flavors / casks (MUST be before Gemini df_query)
        # ===========================
        # A) extremes only when asking "הכי ..."
        if _is_extremes_question(user_text):
            ans = try_handle_extremes_sweet_smoky_rich_delicate(user_text, df)
            if ans:
                await update.message.reply_text(ans)
                return

        # B) focus bottle flavor questions only for "כמה מתוק הוא / האם הוא עדין / הוא עשיר?"
        if _is_focus_flavor_question(user_text):
            ans = try_handle_focus_bottle_flavor_questions(user_text, df, context)
            if ans:
                await update.message.reply_text(ans)
                return
        # A) Flavors of a bottle
        if _looks_like_flavors_of_bottle_query(user_text):
            term = user_text
            term = re.sub(r"(?i)\b(מה|איזה|הטעמים|טעמים|טעם|ארומה|ארומות|nose|palate|palette|שלו|שלה|של|בבקבוק|בקבוק)\b", " ", term)
            term = re.sub(r"\s+", " ", term).strip()

            # follow-up mode -> focused bottle
            if _is_focus_placeholder(term):
                focus_row = _get_focus_bottle_row(active_df, context)
                if focus_row is not None:
                    _set_focus_bottle(context, focus_row)  # ✅ keep focus consistent
                    await update.message.reply_text(build_flavors_reply(focus_row), parse_mode="Markdown")
                    return
                    return
                await update.message.reply_text("על איזה בקבוק תרצה את הטעמים? (לדוגמה: 'מה הטעמים של Glenfiddich 15')")
                return

            # explicit bottle name
            match = find_best_bottle_match(term, active_df)
            if not match.get("best_name") or float(match.get("score") or 0) < 0.70:
                await update.message.reply_text("לא הצלחתי לזהות את הבקבוק. נסה לכתוב מזקקה + שם הבקבוק.")
                return

            chosen = match.get("candidates", [])[0]
            sub = active_df[active_df["bottle_id"] == chosen["bottle_id"]]
            if sub.empty:
                await update.message.reply_text("לא מצאתי את הבקבוק הזה במלאי הפעיל.")
                return

            row = sub.iloc[0]
            _set_focus_bottle(context, row)
            await update.message.reply_text(
    build_flavors_reply(row),
    parse_mode="Markdown"
)
            return

        # B) Casks of a bottle ("איזה חבית הוא")
        if _looks_like_casks_of_bottle_query(user_text):
            term = user_text
            term = re.sub(r"(?i)\b(מה|איזה|חבית|חביות|cask|casks|aged|in|שלו|שלה|של|בבקבוק|בקבוק|החבית|החביות)\b", " ", term)
            term = re.sub(r"\s+", " ", term).strip()

            if _is_focus_placeholder(term):
                focus_row = _get_focus_bottle_row(active_df, context)
                if focus_row is not None:
                    _set_focus_bottle(context, focus_row)  # ✅ keep focus consistent
                    await update.message.reply_text(build_casks_reply(focus_row), parse_mode="Markdown")
                    return
                await update.message.reply_text("על איזה בקבוק תרצה לדעת את החביות? (לדוגמה: 'איזה חבית הוא Glenmorangie 10')")
                return

            match = find_best_bottle_match(term, active_df)
            if not match.get("best_name") or float(match.get("score") or 0) < 0.70:
                await update.message.reply_text("לא הצלחתי לזהות את הבקבוק. נסה לכתוב מזקקה + שם הבקבוק.")
                return

            chosen = match.get("candidates", [])[0]
            sub = active_df[active_df["bottle_id"] == chosen["bottle_id"]]
            if sub.empty:
                await update.message.reply_text("לא מצאתי את הבקבוק הזה במלאי הפעיל.")
                return

            row = sub.iloc[0]
            _set_focus_bottle(context, row)
            await update.message.reply_text(
    build_casks_reply(row),
    parse_mode="Markdown"
)
            return

        # C) Age of a bottle ("מה הגיל שלו")
        if _looks_like_age_of_bottle_query(user_text):
            term = user_text
            term = re.sub(r"(?i)\b(מה|כמה|בן|גיל|שנים|age|aged|שלו|שלה|של|בבקבוק|בקבוק)\b", " ", term)
            term = re.sub(r"\s+", " ", term).strip()

            # follow-up mode -> focused bottle
            if _is_focus_placeholder(term):
                focus_row = _get_focus_bottle_row(active_df, context)
                if focus_row is not None:
                    _set_focus_bottle(context, focus_row)
                    await update.message.reply_text(build_age_reply(focus_row), parse_mode="Markdown")
                    return
                await update.message.reply_text("על איזה בקבוק תרצה לדעת את הגיל? (לדוגמה: 'מה הגיל של Glenfiddich 15')")
                return

            # explicit bottle name
            match = find_best_bottle_match(term, active_df)
            if not match.get("best_name") or float(match.get("score") or 0) < 0.70:
                await update.message.reply_text("לא הצלחתי לזהות את הבקבוק. נסה לכתוב מזקקה + שם הבקבוק.")
                return

            chosen = match.get("candidates", [])[0]
            sub = active_df[active_df["bottle_id"] == chosen["bottle_id"]]
            if sub.empty:
                await update.message.reply_text("לא מצאתי את הבקבוק הזה במלאי הפעיל.")
                return

            row = sub.iloc[0]
            _set_focus_bottle(context, row)
            await update.message.reply_text(build_age_reply(row), parse_mode="Markdown")
            return
        
        # build inventory dict once when needed

        # 2) Analytics / research questions (no Gemini)

        # Smart analytics via Gemini DF planner (portfolio shares, flexible ratios, etc.)
        # We route percent-of-total questions here to avoid confusing them with "remaining stock" bottle questions.
        if _looks_like_portfolio_share_query(user_text):
            reply = await try_gemini_df_query_answer(user_text, df, context)
            if reply:
                await update.message.reply_text(reply)
                return


        if _looks_like_popular_query(user_text):
            # Flexible scope via Gemini planner (e.g., "most popular of M&H")
            plan = gemini_make_plan(user_text, active_df)
            scope_type = None
            scope_name = None
            if plan and str(plan.get("intent", "")).lower() == "popular":
                sc = plan.get("scope") or {}
                scope_type = (sc.get("type") or "").lower()
                scope_name = (sc.get("name") or "").strip() if sc.get("name") else None

            sub = active_df.copy()
            scope_label = "אצלך"

            # If scoped to a distillery, resolve with fuzzy matching and filter
            if scope_type == "distillery" and scope_name:
                dist_match = find_best_distillery_match(scope_name, active_df)
                if dist_match.get("best") and float(dist_match.get("score") or 0) >= 0.62:
                    dist = dist_match["best"]
                    sub = sub[sub["distillery"].astype(str) == str(dist)]
                    scope_label = f"של {dist}"
                else:
                    # couldn't resolve the scope -> ask clarification
                    cands = (dist_match.get("candidates") or [])[:3]
                    if cands:
                        lines = [f"{i+1}. {c['distillery']}" for i, c in enumerate(cands)]
                        context.user_data["pending_count"] = {"candidates": cands, "tries": 0, "mode": "popular_scope"}
                        await update.message.reply_text(
                            "לא הייתי בטוח לאיזו מזקקה התכוונת בשאלה על 'הכי פופולרי'.\n"
                            "בחר 1-3 או כתוב את שם המזקקה:\n" + "\n".join(lines)
                        )
                        return
                    await update.message.reply_text("לא הצלחתי לזהות את המזקקה שביקשת בשאלה על 'הכי פופולרי'.")
                    return

            if sub.empty:
                await update.message.reply_text("לא מצאתי בקבוקים פעילים בתחום שביקשת (אולי אין מלאי פעיל למזקקה הזו).")
                return

            if "avg_consumption_vol_per_day" not in sub.columns or sub["avg_consumption_vol_per_day"].dropna().empty:
                await update.message.reply_text("אין לי נתוני Forecast (avg_consumption_vol_per_day) זמינים כרגע.")
                return

            sub["avg_consumption_vol_per_day"] = pd.to_numeric(sub["avg_consumption_vol_per_day"], errors="coerce")
            sub = sub[pd.notnull(sub["avg_consumption_vol_per_day"])]
            if sub.empty:
                await update.message.reply_text("אין ערכים תקינים של avg_consumption_vol_per_day בתחום שביקשת.")
                return

            top = sub.sort_values("avg_consumption_vol_per_day", ascending=False).head(1).iloc[0]
            await update.message.reply_text(
                f"🏆 הבקבוק הכי 'פופולרי' {scope_label} לפי Avg Consumption / Day הוא:\n"
                f"🥃 {top.get('full_name')}\n"
                f"📈 {round(float(top.get('avg_consumption_vol_per_day') or 0), 2)} ml/day"
            )
            _set_focus_bottle(context, top)
            return

        if _looks_like_oxidized_query(user_text):
                    term = user_text

                    # ניקוי מילים "לא רלוונטיות" (אבל לא מסתמכים על זה בלבד)
                    term = re.sub(r"(?i)\b(מה|מהו|מה זה|סטטוס|חמצון|החמצון|מחומצן|best|before|בסט|ביפור|שלו|שלה|של|תוקף|תאריך|פג)\b", " ", term)
                    term = re.sub(r"\s+", " ", term).strip()

                    # ✅ קריטי: Normalize כדי להפוך "שלו?" ל-"שלו" ו-"?" ל-""
                    term = _normalize_text(term)

                    # ✅ קריטי: השתמש בפונקציה הכללית שלך (זה בדיוק מה שפתר לך "כמה נשאר בו")
                    if _is_focus_placeholder(term):
                        focus_row = _get_focus_bottle_row(active_df, context)
                        if focus_row is None:
                            await update.message.reply_text("על איזה בקבוק אתה שואל? תציין שם בקבוק או תשאל קודם על בקבוק ספציפי.")
                            return

                        bb = focus_row.get("Best_Before", None)
                        bb_dt = _safe_to_datetime(bb)
                        if pd.isna(bb_dt):
                            await update.message.reply_text("אין לי Best Before מוגדר לבקבוק הזה.")
                            return
                        
                        from datetime import datetime

                        today = datetime.now().date()  # עדיף UTC כדי להתאים ל-BQ
                        bb_date = bb_dt.date()

                        warning_line = ""
                        if bb_date < today:
                            warning_line = "\n⚠️ שים לב! הבקבוק מאבד טעמים, מומלץ לסיים בהקדם!"

                        await update.message.reply_text(
                            f"🧪 סטטוס חמצון (Best Before):\n"
                            f"🥃 {focus_row.get('full_name')}\n"
                            f"📅 Best Before: {bb_date}"
                            f"{warning_line}"
                        )
                        return
                    # Explicit bottle name path (אם בכל זאת כתבת שם)
                    match = find_best_bottle_match(term, active_df)
                    if not match.get("best_name") or float(match.get("score") or 0) < 0.70:
                        await update.message.reply_text("לא הצלחתי לזהות את הבקבוק. נסה לכתוב מזקקה + שם הבקבוק.")
                        return

                    chosen = match["candidates"][0]
                    sub = active_df[active_df["bottle_id"] == chosen["bottle_id"]]
                    if sub.empty:
                        await update.message.reply_text("לא מצאתי את הבקבוק הזה במלאי הפעיל.")
                        return

                    row = sub.iloc[0]
                    _set_focus_bottle(context, row)

                    bb = row.get("Best_Before", None)
                    bb_dt = _safe_to_datetime(bb)
                    if pd.isna(bb_dt):
                        await update.message.reply_text(f"לא מוגדר Best Before לבקבוק: {row.get('full_name')}")
                        return

                    from datetime import datetime

                    today = datetime.now().date()  # עדיף UTC כדי להתאים ל-BQ
                    bb_date = bb_dt.date()

                    warning_line = ""
                    if bb_date < today:
                        warning_line = "\n⚠️ שים לב! הבקבוק מאבד טעמים, מומלץ לסיים בהקדם!"

                    await update.message.reply_text(
                        f"🧪 סטטוס חמצון (Best Before):\n"
                        f"🥃 {row.get('full_name')}\n"
                        f"📅 Best Before: {str(bb_dt.date())}"
                        f"{warning_line}"
                    )
                    return


        # Stock query: percent/ml left in a specific bottle
        if _looks_like_stock_query(user_text):
            # Try extracting bottle term by removing common stock words
            term = user_text
            term = re.sub(r"(?i)\b(וכמה|כמה|אחוז|%|נשאר|נשאר לי|מהבקבוק|מה|לי|של|בבקבוק|בקבוק|current|stock|remaining|left|inventory|מלאי)\b", " ", term)
            term = re.sub(r"\s+", " ", term).strip()
            # Follow-up mode: user didn't specify bottle name (or left only pronoun) -> use focused bottle
            if _is_focus_placeholder(term):
                focus_row = _get_focus_bottle_row(active_df, context)
                if focus_row is not None:
                    await update.message.reply_text(build_stock_reply(focus_row))
                    return
                await update.message.reply_text(
                    "על איזה בקבוק תרצה לדעת כמה נשאר? (לדוגמה: 'כמה אחוז נשאר לי מהבקבוק Glenfiddich 15')"
                )
                return

            match = find_best_bottle_match(term, active_df)
            if not match.get("best_name") or float(match.get("score") or 0) < 0.70:
                await update.message.reply_text(
                    "לא הצלחתי לזהות את הבקבוק. נסה לכתוב מזקקה + שם הבקבוק (ואם יש גיל/גרסה – אפילו יותר טוב)."
                )
                return

            candidates = match.get("candidates", [])[:3]
            # If ambiguous, ask
            if len(candidates) >= 2 and (candidates[0]["score"] - candidates[1]["score"]) < 0.05:
                context.user_data["pending_stock"] = {"candidates": candidates, "tries": 0}
                lines = [f"{i+1}. {c['full_name']} (score {c['score']})" for i, c in enumerate(candidates)]
                await update.message.reply_text(
                    "מצאתי כמה התאמות אפשריות. על איזה בקבוק התכוונת?\n"
                    "בחר 1-3 או כתוב את השם:\n" + "\n".join(lines)
                )
                return

            chosen = candidates[0]
            sub = active_df[active_df["bottle_id"] == chosen["bottle_id"]]
            if sub.empty:
                await update.message.reply_text("לא מצאתי את הבקבוק הזה במלאי הפעיל.")
                return
            row = sub.iloc[0]
            _set_focus_bottle(context, row)
            await update.message.reply_text(build_stock_reply(row))
            return


        # Recommend SOON: choose bottles with Best_Before within next ~3 months (90 days)
        if _looks_like_recommend_soon_query(user_text):
            sub = active_df.copy()
            if "Best_Before" not in sub.columns:
                await update.message.reply_text("אין לי שדה Best_Before בנתוני ה-Forecast כרגע.")
                return

            sub["Best_Before_dt"] = sub["Best_Before"].apply(_safe_to_datetime)
            sub = sub[pd.notnull(sub["Best_Before_dt"])]
            if sub.empty:
                await update.message.reply_text("לא מצאתי אצלך בקבוקים עם Best Before מוגדר.")
                return

            today = pd.Timestamp.now(tz=None).normalize()
            sub["days_to_bb"] = (sub["Best_Before_dt"].dt.normalize() - today).dt.days

            # window: 0..90 days
            window = sub[(sub["days_to_bb"] >= 0) & (sub["days_to_bb"] <= 90)].sort_values("days_to_bb")
            if window.empty:
                await update.message.reply_text("אין כרגע בקבוקים שמתקרבים ל-Best Before ב-3 החודשים הקרובים.")
                return

            top = window.head(1).iloc[0]
            await update.message.reply_text(
                "⏳ מומלץ לשתות בקרוב (Best Before בתוך 3 חודשים):\n"
                f"🥃 {top.get('full_name')}\n"
                f"📅 Best Before: {str(top.get('Best_Before_dt').date())}\n"
                f"🕒 עוד {int(top.get('days_to_bb'))} ימים"
            )
            _set_focus_bottle(context, top)
            return

        if _looks_like_recommend_now_query(user_text):
            sub = active_df.copy()
            # Prefer est_consumption_date if present, else predicted_finish_date
            date_col = None
            if "est_consumption_date" in sub.columns and sub["est_consumption_date"].notna().any():
                date_col = "est_consumption_date"
            elif "predicted_finish_date" in sub.columns and sub["predicted_finish_date"].notna().any():
                date_col = "predicted_finish_date"
            if not date_col:
                await update.message.reply_text("אין לי כרגע תאריכי Forecast (est_consumption_date / predicted_finish_date) זמינים.")
                return

            sub["target_dt"] = sub[date_col].apply(_safe_to_datetime)
            sub = sub[pd.notnull(sub["target_dt"])]
            if sub.empty:
                await update.message.reply_text("לא מצאתי תאריכים תקינים ב-Forecast.")
                return

            today = pd.Timestamp.now().normalize()
            sub["is_overdue"] = sub["target_dt"].dt.normalize() < today

            pick = sub.sort_values(["is_overdue", "target_dt"], ascending=[False, True]).head(1).iloc[0]

            # ✅ פה: לקבע פוקוס על הבקבוק שהומלץ עכשיו
            _set_focus_bottle(context, pick)

            suffix = " (עבר/דחוף)" if bool(pick.get("is_overdue")) else ""
            await update.message.reply_text(
                f"🥃 הדראם הבא המומלץ לפי Forecast (הכי קרוב/דחוף) הוא:\n"
                f"✅ {pick.get('full_name')}\n"
                f"📅 {date_col}: {str(pick.get('target_dt').date())}{suffix}"
            )
            return

        inventory_dict = None

        # -----------------------------
        # HAVE QUERY (YES/NO)
        # -----------------------------
        if _looks_like_have_query(user_text):
            term = _extract_entity_for_have(user_text).strip()
            if not term:
                await update.message.reply_text("מה לחפש? למשל: 'האם יש לי Glenfiddich 15?'")
                return

            # First: bottle match (best for specific editions like "Project XX")
            m = find_best_bottle_match(term, active_df)
            best_name = m.get("best_name")
            best_score = float(m.get("score") or 0)

            if best_name and best_score >= 0.78 and _token_overlap_ok(term, best_name, min_overlap=0.70):
                # ✅ קבע פוקוס על הבקבוק שמצאת
                # best_name זה שם, אבל יש לך גם candidate עם bottle_id בתוך m["candidates"]
                chosen = (m.get("candidates") or [None])[0]

                if chosen and chosen.get("bottle_id") is not None:
                    sub_focus = active_df[active_df["bottle_id"] == chosen["bottle_id"]]
                    if not sub_focus.empty:
                        _set_focus_bottle(context, sub_focus.iloc[0])

                await update.message.reply_text(f"כן 🙂 יש לך את:\n✅ {best_name}")
                return

            cands = (m.get("candidates") or [])[:5]
            if cands:
                lines = [f"{i+1}. {c.get('full_name')} (score {c.get('score')})" for i, c in enumerate(cands)]
                await update.message.reply_text(
                    "לא הייתי בטוח שהתכוונת בדיוק לזה. מצאתי כמה קרובים:\n" + "\n".join(lines) +
                    "\n\nתענה במספר (1-5) או תכתוב את השם המדויק."
                )
                return
            
            
            # Second: distillery match (best for 'm&h', 'glendiffich' typos, etc.)
            dm = find_best_distillery_match(term, active_df)
            best_dist = dm.get("best")
            dist_score = float(dm.get("score") or 0)

            if best_dist and dist_score >= 0.72:
                sub = active_df[active_df["distillery"].astype(str) == str(best_dist)]
                cnt = int(sub["bottle_id"].nunique())
                if cnt > 0:
                    await update.message.reply_text(f"כן 🙂 יש לך **{cnt}** בקבוקים פעילים של {best_dist}.")
                else:
                    await update.message.reply_text(f"לא. אין לך כרגע בקבוקים פעילים של {best_dist}.")
                return

            # Not confident -> propose top candidates (bottle)
            cands = (m.get("candidates") or [])[:3]
            if cands:
                lines = [f"{i+1}. {c.get('full_name')} (score {c.get('score')})" for i, c in enumerate(cands)]
                await update.message.reply_text(
                    "לא הייתי בטוח למה התכוונת. יכול להיות שהתכוונת לאחד מאלה?\n" + "\n".join(lines)
                )
                return

            await update.message.reply_text("לא מצאתי התאמה במלאי הפעיל. נסה לכתוב מזקקה + שם/גיל 🙂")
            return

        # -----------------------------
        # PERCENT / SHARE QUERIES (must run even without "כמה")
        # -----------------------------
        if _is_percent_of_total_question(user_text) or _looks_like_portfolio_share_query(user_text):
            rb = rule_based_inventory_answer(user_text, df)
            if rb:
                await update.message.reply_text(rb)
                return

            # If RB couldn't infer category, try Gemini DF planner as fallback
            reply = await try_gemini_df_query_answer(user_text, df, context)
            if reply:
                await update.message.reply_text(reply)
                return
        # -----------------------------
        # COUNT ROUTING (FIXED ORDER)
        # -----------------------------
        if _looks_like_count_query(user_text):
            # 0) Deterministic inventory Q&A first (category/total/percent, etc.)
            rb = rule_based_inventory_answer(user_text, df)
            if rb:
                await update.message.reply_text(rb)
                return

            # 1) Try fuzzy-distillery count ONLY if entity exists and match is strong.
            #    This prevents "וויסקי/יין/אלכוהול" from triggering distillery prompts.
            # 1) Fuzzy distillery count when user provided an entity (m&h / Glenfiddich / etc.)
            ent = _extract_entity_for_count(user_text).strip()
            if ent:
                dist_match = find_best_distillery_match(ent, active_df)
                best_dist = dist_match.get("best")
                score = float(dist_match.get("score") or 0)

                # ✅ v13-like threshold (more forgiving)
                if (not best_dist) or (score < 0.62):
                    cands = (dist_match.get("candidates") or [])[:3]
                    if not cands:
                        await update.message.reply_text("לא מצאתי מזקקה דומה במלאי הפעיל.")
                        return
                    lines = [f"{i+1}. {c['distillery']} (score {c['score']})" for i, c in enumerate(cands)]
                    await update.message.reply_text(
                        "לא הייתי בטוח לאיזו מזקקה התכוונת.\n"
                        "תכתוב שם מדויק יותר או בחר אחד:\n" + "\n".join(lines)
                    )
                    return

                # confident -> answer
                sub = active_df[active_df["distillery"].astype(str) == str(best_dist)]
                cnt = int(sub["bottle_id"].nunique())

                sample = (
                    sub["bottle_name"]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .head(8)
                )

                if sample.empty:
                    await update.message.reply_text(f"יש לך {cnt} בקבוקים פעילים של {best_dist}.")
                    return

                details = "\n".join([f"• {name} ×{int(n)}" for name, n in sample.items()])
                more = f"\n(+ עוד {cnt - 8} נוספים)" if cnt > 8 else ""
                await update.message.reply_text(
                    f"יש לך {cnt} בקבוקים פעילים של {best_dist}.\n\n{details}{more}"
                )
                return

            # 2) Otherwise, Gemini DF planner for flexible analytics
            reply = await try_gemini_df_query_answer(user_text, df, context)
            if reply:
                await update.message.reply_text(reply)
                return

        # 2b) Update query: fuzzy bottle + confirmation when not exact
        if _looks_like_update(user_text):
            amount_ml = _extract_amount_ml(user_text)
            if not amount_ml or amount_ml <= 0:
                await update.message.reply_text("כמה מ״ל שתית/מזגת? (למשל: 'שתיתי 60ml Glenfiddich 15')")
                return

            ent = _extract_entity_for_update(user_text)
            ent = ent.strip()
            if not ent:
                focus_row = _get_focus_bottle_row(active_df, context)
                if focus_row is not None:
                    ent = str(focus_row.get("full_name") or "").strip()
                else:
                    await update.message.reply_text("איזה בקבוק בדיוק? תכתוב מזקקה + שם/גיל (או תשאל קודם על בקבוק ואז תכתוב 'שתיתי 60ml').")
                    return

            match = find_best_bottle_match(ent, active_df)
            if not match["best_name"] or match["score"] < 0.70:
                # show top candidates anyway (3) to allow pick
                cands = match["candidates"][:3]
                if not cands:
                    await update.message.reply_text("לא מצאתי התאמה במלאי הפעיל. תכתוב את השם קצת יותר ברור.")
                    return
                lines = [f"{i+1}. {c['full_name']} (score {c['score']})" for i, c in enumerate(cands)]
                context.user_data["pending_update"] = {"amount_ml": amount_ml, "candidates": cands}
                await update.message.reply_text(
                    "לא מצאתי התאמה חד-משמעית.\n"
                    "תבחר מספר 1-3:\n" + "\n".join(lines)
                )
                return

            # prepare inventory dict only now
            inventory_dict = {}
            for _, r in active_df.iterrows():
                inventory_dict[int(r["bottle_id"])] = {
                    "name": r["full_name"],
                    "stock": float(r["stock_status_per"]),
                    "vol": float(r["orignal_volume"]) if pd.notnull(r.get("orignal_volume")) else 700.0,
                    "old_nose": list(r["nose"]) if isinstance(r.get("nose"), list) else [],
                    "old_palette": list(r["palette"]) if isinstance(r.get("palette"), list) else [],
                    "old_abv": float(r["alcohol_percentage"]) if pd.notnull(r.get("alcohol_percentage")) else 0.0,
                }

            best_name = match["best_name"]
            bottle_id = match["bottle_id"]

            # REQUIREMENT: if not exact, ask before update
            if _normalize_text(ent) != _normalize_text(best_name):
                context.user_data["pending_update"] = {
                    "amount_ml": amount_ml,
                    "bottle_id": bottle_id,
                    "full_name": best_name,
                    "candidates": match["candidates"][:3],
                }
                lines = [f"{i+1}. {c['full_name']} (score {c['score']})" for i, c in enumerate(match["candidates"][:3])]
                await update.message.reply_text(
                    f"התכוונת ל:\n🥃 {best_name}\n"
                    f"לעדכן שתייה של {amount_ml}ml?\n"
                    f"ענה 'כן' כדי לעדכן, 'לא' כדי לבטל, או בחר 1-3:\n" + "\n".join(lines)
                )
                return

            ok, msg = execute_drink_update(int(bottle_id), int(amount_ml), inventory_dict)
            _set_focus_bottle(context, {'bottle_id': int(bottle_id), 'full_name': best_name})
            await update.message.reply_text(msg)
            return
        
        # 0) hard rules (שתיתי/הוספה/תמונה וכו') נשארים כרגיל
        # ...


        # --- FINAL: Gemini as the default engine (Option 2) ---
        df = get_all_data_as_df()

        try:
            reply = await try_gemini_df_query_answer(user_text, df, context)
            if reply:
                await update.message.reply_text(reply)
                return

            await update.message.reply_text("לא הצלחתי להבין לגמרי את הבקשה. תנסה לנסח אחרת 🙏")
            return

        except Exception as e:
            logging.warning(f"Gemini DF engine error: {e}")
            await update.message.reply_text("לא הצלחתי להבין לגמרי את הבקשה. תנסה לנסח אחרת 🙏")
            return
        
        
    except Exception as e:
        logging.exception("Error in handle_message")
        await update.message.reply_text(f"❌ שגיאה:\n{e}")
        


if __name__ == "__main__":
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))


    print("Whisky Telegram agent running (deterministic fuzzy inventory + Gemini DF analytics fallback)...")
    application.run_polling()
