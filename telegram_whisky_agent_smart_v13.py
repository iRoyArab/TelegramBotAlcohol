import logging

BOT_VERSION = "v56"
import time
import re
import json
import uuid
import ast
from datetime import datetime, date
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
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
Consumption_table = "consumption_table"

TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
HISTORY_TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{HISTORY_TABLE_ID}"
FORECAST_TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{FORECAST_TABLE_ID}"
VIEW_REF = f"{PROJECT_ID}.{DATASET_ID}.{VIEW_ID}"
CONS_REF = f"{PROJECT_ID}.{DATASET_ID}.{Consumption_table}"


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
    'Sweet-Citrucy': (1.51, 2.6),
    'Citrucy-Spicy': (2.61, 3.6),
    'Coffee Like - Sea Salt': (3.61, 4.5),
    'Minerals - Sulfur': (4.51, 5.5),
    'Ash - BBQ Smoke': (5.51, 7.5),
    'Heavy Peat - Medicinal Smoke': (7.51, 100.0)
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
    # Strip Unicode direction/control marks that Telegram injects into Hebrew text
    # (e.g. \u200f = RTL mark, \u200e = LTR mark, \u202a-\u202e = embedding marks)
    cleaned = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", (s or ""))
    return re.sub(r"\s+", " ", cleaned.strip().lower())

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


# ══════════════════════════════════════════════════════
#  VFM  (Value For Money) – זיהוי ומענה
# ══════════════════════════════════════════════════════

def _is_vfm_question(text: str) -> bool:
    """
    מזהה שאלות על VFM, לדוגמה:
    - מה ה-VFM שלו?
    - מה רמת ה-VFM של גלנפידיך?
    - מי הכי VFM?
    - מה הכי פחות VFM?
    - מה הכי VFM ממזקקת X?
    """
    if not text:
        return False
    t = text.strip().lower()
    return bool(re.search(r"\bvfm\b", t, re.IGNORECASE))


def _fmt_vfm_line(i: int, dist: str, bottle: str, vfm: float) -> str:
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
    dist   = (dist   or "-").strip()
    bottle = (bottle or "-").strip()
    return f"{medal} {dist} – {bottle}  ·  VFM: {vfm:.2f}"


def try_handle_vfm_questions(
    user_text: str,
    active_df: pd.DataFrame,
    context: ContextTypes.DEFAULT_TYPE
) -> str | None:
    """
    מטפל בשלושה סוגי שאלות VFM:

    1. שאלה על בקבוק בפוקוס / שם ספציפי:
       "מה ה-VFM שלו?" / "מה ה-VFM של Glenfiddich 15?"
    2. שאלה כללית על כל האוסף:
       "מי הכי VFM?" / "מה הכי פחות VFM?"
    3. שאלה על קבוצה (focus_list / מזקקה / שתיתי לאחרונה):
       "מה הכי VFM ממזקקת Ardbeg?" / "מה הכי VFM מהבקבוקים ששתיתי לאחרונה?"
    """
    if not user_text:
        return None

    t = user_text.strip().lower()

    # ── סוג 1: בקבוק ספציפי / פוקוס ──
    # כולל ביטויי גוף שלישי (שלו / הוא / זה / בו / עליו) או "מה ה-VFM של <name>"
    is_focus_query = bool(re.search(r"\b(שלו|הוא|שלה|בו|עליו|זה|שלהם)\b", t)) or \
                     bool(re.search(r"(vfm\s*(של|ל))\b", t, re.IGNORECASE))

    # ── סוג 2: הכי / הכי פחות VFM (כללי) ──
    want_best  = bool(re.search(r"(הכי\s*vfm|vfm\s*הכי\s*גבוה|הכי\s*שווה\s*כסף|best\s*vfm)", t, re.IGNORECASE))
    want_worst = bool(re.search(r"(הכי\s*פחות\s*vfm|vfm\s*הכי\s*נמוך|worst\s*vfm|הכי\s*פחות\s*שווה)", t, re.IGNORECASE))

    sep = "━━━━━━━━━━━━━━━━━━"

    if active_df is None or active_df.empty:
        return "אין לי נתונים כרגע."

    if "rvfm" not in active_df.columns:
        return "⚠️ עמודת rvfm לא נמצאה בנתונים."

    df_vfm = active_df.copy()
    df_vfm["rvfm"] = pd.to_numeric(df_vfm["rvfm"], errors="coerce")
    df_vfm = df_vfm.dropna(subset=["rvfm"])

    if df_vfm.empty:
        return "⚠️ אין ערכי VFM תקינים בנתונים."

    # ─── סוג 1: פוקוס / שם ספציפי ───
    if is_focus_query and not want_best and not want_worst:
        row = _get_focus_bottle_row(active_df, context)
        if row is None:
            row = find_best_bottle_match(user_text, active_df)
            if isinstance(row, dict):
                # find_best_bottle_match מחזירה dict עם candidates
                candidates = row.get("candidates", [])
                if candidates and float(row.get("score", 0)) >= 0.6:
                    bid = candidates[0]["bottle_id"]
                    hit = active_df[active_df["bottle_id"] == bid]
                    row = hit.iloc[0] if not hit.empty else None
                else:
                    row = None

        if row is None:
            return "על איזה בקבוק מדובר? כתוב שם בקבוק/מזקקה, או שאל קודם שאלה שמחזירה בקבוק."

        vfm_val = pd.to_numeric(row.get("rvfm"), errors="coerce")
        if pd.isna(vfm_val):
            dist   = str(row.get("distillery") or "-")
            bottle = str(row.get("bottle_name") or row.get("full_name") or "-")
            return f"⚠️ אין ערך VFM זמין עבור {dist} – {bottle}."

        dist   = str(row.get("distillery") or "-").strip()
        bottle = str(row.get("bottle_name") or row.get("full_name") or "-").strip()

        # פרשנות איכותית לפי סקאלה 1-10
        if vfm_val >= 9.5:
            verdict = "🟢 מחיר מציאה – חובה לרכוש!"
        elif vfm_val >= 7.201:
            verdict = "🟢 מחיר סבבה  – רכישה מומלצת"
        elif vfm_val >= 6.01:
            verdict = "🟡 מחיר הוגן – ממוצע בסדר"
        elif vfm_val >= 3.0:
            verdict = "🟠 יקר, אך שווה יחסית – לשקול בכובד ראש"
        else:
            verdict = "🔴 לא שווה בכלל – לא מומלץ לרכישה"

        return (
            f"{sep}\n💰 VFM – ערך לכסף\n{sep}\n\n"
            f"{dist} – {bottle}\n\n"
            f"📊 ציון VFM: {vfm_val:.2f} / 10\n"
            f"{verdict}\n\n"
            f"ℹ️ סקאלה: 9.5–10 מציאה · 7.2–9.5 שווה · 6–7.2 הוגן · 3–6 יקר · 1–3 לא שווה"
        )

    # ─── סוג 2 / 3: הכי / הכי פחות ───
    if not want_best and not want_worst:
        return None

    # האם יש focus_list שרלוונטי לשאלה?
    working_df = df_vfm
    scope_label = "כל האוסף שלך"

    list_df = _get_focus_list_df(active_df, context)
    if list_df is not None and not list_df.empty and _is_list_followup(user_text):
        list_df["rvfm"] = pd.to_numeric(list_df["rvfm"], errors="coerce")
        working_df  = list_df.dropna(subset=["rvfm"])
        scope_label = f"בקבוקי {context.user_data.get('focus_list_label', 'הרשימה')}"
    else:
        dist_scope = _extract_distillery_scope_from_extremes(user_text, df_vfm)
        if dist_scope is not None and not dist_scope.empty:
            working_df  = dist_scope
            scope_label = f"בקבוקי {str(dist_scope['distillery'].iloc[0])}"

    if working_df.empty:
        return "⚠️ אין ערכי VFM תקינים בקבוצה המבוקשת."

    k = 3
    if want_best:
        top = working_df.nlargest(k, "rvfm")
        title = f"💰 טופ {k} הכי שווי כסף – {scope_label}"
        hint  = "🟢 גבוה יותר = שווה כסף יותר"
    else:
        top = working_df.nsmallest(k, "rvfm")
        title = f"💸 טופ {k} הכי פחות שווי כסף – {scope_label}"
        hint  = "🔴 נמוך יותר = יקר יחסית לאיכות"

    avg_vfm = float(df_vfm["rvfm"].mean())
    lines = []
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        vfm_val = float(r["rvfm"])
        if vfm_val > 9.5:
            tag = "🟢 מציאה"
        elif vfm_val > 7.2:
            tag = "🟢 שווה"
        elif vfm_val > 6:
            tag = "🟡 הוגן"
        elif vfm_val > 3:
            tag = "🟠 יקר יחסית"
        else:
            tag = "🔴 לא שווה"
        lines.append(
            f"{_fmt_vfm_line(i, str(r.get('distillery', '-')), str(r.get('bottle_name', '-')), vfm_val)}  ({tag})"
        )

    return (
        f"{sep}\n{title}\n{sep}\n\n"
        + "\n".join(lines)
        + f"\n\n📌 ממוצע VFM באוסף: {avg_vfm:.2f} / 10\n"
        + hint
        + "\nℹ️ סקאלה: 1–3 לא שווה · 3–5 יקר · 5–6.5 הוגן · 6.5–8 שווה · 8–10 מציאה"
    )



# ==========================================
# Group Extremes: VFM / Best Before / Stock / ABV on distillery/focus-list scope
# Examples:
#   "מי מביניהם הכי VFM?"
#   "מה מביניהם מומלץ לשתות הכי ממוקדם?"
#   "באיזה בקבוק מבין Glenmorangie נשאר הכי קצת?"
#   "מה מבין M&H הכי אלכוהולי?"
# ==========================================

_GROUP_VFM_RE = re.compile(r"\b(vfm|שווי?\s*כסף|value\s*for\s*money)\b", re.IGNORECASE)
_GROUP_BEST_BEFORE_RE = re.compile(
    r"(best\s*before|ממוקדם|הכי\s*מוקדם|לשתות\s*(?:הכי\s*)?בקרוב|פג\s*תוקף|מתי\s*להספיק)",
    re.IGNORECASE
)
_GROUP_STOCK_LOW_RE = re.compile(
    r"(נשאר\s*הכי\s*קצת|הכי\s*(?:מעט|קצת|ריק|כמעט\s*נגמר)|עומד\s*להסתיים|"
    r"least\s*(?:remaining|left)|almost\s*(?:empty|gone)|מלאי\s*הכי\s*(?:נמוך|קטן|דל))",
    re.IGNORECASE
)
_GROUP_ABV_MAX_RE = re.compile(
    r"(הכי\s*אלכוהולי|הכי\s*(?:חזק|גבוה).*abv|abv.*הכי\s*(?:גבוה|חזק)|highest\s*abv|most\s*alcohol|אחוז\s*אלכוהול\s*הכי\s*גבוה)",
    re.IGNORECASE
)


def _detect_group_extreme_intent(user_text: str) -> str | None:
    if _GROUP_VFM_RE.search(user_text):
        return "vfm"
    if _GROUP_BEST_BEFORE_RE.search(user_text):
        return "best_before"
    if _GROUP_STOCK_LOW_RE.search(user_text):
        return "stock_low"
    if _GROUP_ABV_MAX_RE.search(user_text):
        return "abv_max"
    return None


def try_handle_group_extremes(user_text: str, scope_df: "pd.DataFrame", scope_label: str) -> "str | None":
    if scope_df is None or scope_df.empty:
        return None
    intent = _detect_group_extreme_intent(user_text)
    if not intent:
        return None
    sep = "━━━━━━━━━━━━━━━━━━"

    if intent == "vfm":
        df_w = scope_df.copy()
        if "rvfm" not in df_w.columns:
            return f"📋 מתוך בקבוקי {scope_label}\n⚠️ אין ערכי VFM זמינים."
        df_w["rvfm"] = pd.to_numeric(df_w["rvfm"], errors="coerce")
        df_w = df_w.dropna(subset=["rvfm"])
        if df_w.empty:
            return f"📋 מתוך בקבוקי {scope_label}\n⚠️ אין ערכי VFM זמינים לקבוצה זו."
        top = df_w.nlargest(3, "rvfm")
        lines = []
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            v = float(r["rvfm"])
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
            name = str(r.get("full_name") or r.get("bottle_name") or "-")
            tag = ("🟢 מציאה" if v >= 8.01 else "🟢 שווה" if v >= 6.501 else
                   "🟡 הוגן" if v >= 5.01 else "🟠 יקר יחסית" if v >= 3.0 else "🔴 לא שווה")
            lines.append(f"{medal} {name}  ·  VFM: {v:.2f}  ({tag})")
        return (f"{sep}\n💰 הכי שווי כסף – {scope_label}\n{sep}\n\n" +
                "\n".join(lines) +
                "\n\nℹ️ סקאלה: 1–3 לא שווה · 3–5 יקר · 5–6.5 הוגן · 6.5–8 שווה · 8–10 מציאה")

    if intent == "best_before":
        df_w = scope_df.copy()
        if "Best_Before" not in df_w.columns:
            return f"📋 מתוך בקבוקי {scope_label}\n⚠️ אין שדה Best Before."
        df_w["_bb_dt"] = df_w["Best_Before"].apply(_safe_to_datetime)
        df_w = df_w.dropna(subset=["_bb_dt"])
        if df_w.empty:
            return f"📋 מתוך בקבוקי {scope_label}\n⚠️ אין ערכי Best Before זמינים."
        df_w = df_w.sort_values("_bb_dt")
        top = df_w.head(3)
        today = pd.Timestamp.now(tz=None).normalize()
        lines = []
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
            name = str(r.get("full_name") or r.get("bottle_name") or "-")
            bb_date = r["_bb_dt"].normalize()
            days = int((bb_date - today).days)
            urgency = ("⚠️ עבר!" if days < 0 else "🔴 דחוף!" if days <= 14 else
                       "🟠 בקרוב" if days <= 60 else "🟡 עוד זמן")
            lines.append(f"{medal} {name}\n    📅 {str(bb_date.date())} · עוד {days} ימים {urgency}")
        return (f"{sep}\n⏳ הכי ממוקדם לשתייה – {scope_label}\n{sep}\n\n" +
                "\n".join(lines))

    if intent == "stock_low":
        df_w = scope_df.copy()
        if "stock_status_per" not in df_w.columns:
            return f"📋 מתוך בקבוקי {scope_label}\n⚠️ אין נתוני מלאי."
        df_w["_per"] = pd.to_numeric(df_w["stock_status_per"], errors="coerce")
        df_w = df_w.dropna(subset=["_per"])
        df_w = df_w[df_w["_per"] > 0]
        if df_w.empty:
            return f"📋 מתוך בקבוקי {scope_label}\n⚠️ אין נתוני מלאי זמינים."
        top = df_w.nsmallest(3, "_per")
        lines = []
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
            name = str(r.get("full_name") or r.get("bottle_name") or "-")
            per = float(r["_per"])
            vol = None
            try:
                v = r.get("orignal_volume")
                if pd.notnull(v):
                    vol = float(v)
            except Exception:
                pass
            ml_str = f"  (~{round((per/100)*vol)}ml)" if vol else ""
            pfd = _fmt_date(r.get("predicted_finish_date"))
            pfd_str = f"\n    📅 צפי סיום: {pfd}" if pfd else ""
            urgency = ("🔴 כמעט נגמר!" if per <= 10 else "🟠 מעט נשאר" if per <= 25 else "🟡 חצי")
            lines.append(f"{medal} {name}\n    📊 נשאר: {round(per, 1)}%{ml_str} {urgency}{pfd_str}")
        return (f"{sep}\n📉 הכי מעט נשאר – {scope_label}\n{sep}\n\n" +
                "\n".join(lines))

    if intent == "abv_max":
        df_w = scope_df.copy()
        if "alcohol_percentage" not in df_w.columns:
            return f"📋 מתוך בקבוקי {scope_label}\n⚠️ אין נתוני ABV."
        df_w["_abv"] = pd.to_numeric(df_w["alcohol_percentage"], errors="coerce")
        df_w = df_w.dropna(subset=["_abv"])
        if df_w.empty:
            return f"📋 מתוך בקבוקי {scope_label}\n⚠️ אין נתוני ABV זמינים."
        top = df_w.nlargest(3, "_abv")
        lines = []
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
            name = str(r.get("full_name") or r.get("bottle_name") or "-")
            abv = float(r["_abv"])
            tag = ("🔥 Cask Strength" if abv >= 55 else "💪 חזק" if abv >= 46 else "✅ סטנדרטי")
            lines.append(f"{medal} {name}  ·  {abv:.1f}%  {tag}")
        return (f"{sep}\n🔥 הכי אלכוהולי – {scope_label}\n{sep}\n\n" +
                "\n".join(lines) +
                "\n\nℹ️ <46% סטנדרטי · 46–55% חזק · 55%+ Cask Strength")

    return None


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
           t3.avg_consumption_vol_per_day, t3.est_consumption_date, t3.avg_days_between_updates, t3.predicted_finish_date, t3.latest_consumption_time, t3.Best_Before,
           t3.rvfm,
       COALESCE(t4_exact.total_volume_consumed, t4_latest.total_volume_consumed) AS total_volume_consumed,
       COALESCE(t4_exact.total_drams, t4_latest.total_drams)                     AS total_drams,
       t5.known_drinkers

    FROM `{TABLE_REF}` t1
    LEFT JOIN `{VIEW_REF}` t2 ON t1.bottle_id = t2.bottle_id
    LEFT JOIN `{FORECAST_TABLE_REF}` t3 ON t1.bottle_id = t3.bottle_id
    LEFT JOIN `{CONS_REF}` as t4_exact ON t1.bottle_id =  t4_exact.bottle_id AND t3.latest_consumption_time =  t4_exact.update_date
    
    LEFT JOIN (
        SELECT bottle_id,
            total_volume_consumed,
            total_drams
        FROM `{CONS_REF}`
        QUALIFY ROW_NUMBER() OVER (PARTITION BY bottle_id ORDER BY update_date DESC) = 1
    ) t4_latest 
        ON t1.bottle_id = t4_latest.bottle_id
        AND t4_exact.bottle_id IS NULL

    LEFT JOIN (
        SELECT ARRAY_AGG(DISTINCT d IGNORE NULLS ORDER BY d) AS known_drinkers
        FROM `{HISTORY_TABLE_REF}`,
        UNNEST(drinker_name) AS d
        WHERE d IS NOT NULL AND TRIM(d) != ''
    ) t5 ON TRUE
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



def normalize_to_list(x):
    import numpy as np
    if x is None:
        return []
    # numpy array (BigQuery REPEATED columns come as ndarray via pandas)
    if isinstance(x, np.ndarray):
        return [str(v).strip() for v in x.tolist() if str(v).strip()]
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
# Focus List — רשימת בקבוקים מהשאלה האחרונה
# נשמרת אחרי "כמה בקבוקים של X" כדי לאפשר
# follow-up כמו "מה מהם הכי יקר / הכי מתוק"
# ==========================================
def _set_focus_list(context: ContextTypes.DEFAULT_TYPE, df_sub: "pd.DataFrame", label: str = ""):
    """שמור רשימת bottle_ids ותוית הקשר (שם מזקקה / קטגוריה)."""
    try:
        ids = df_sub["bottle_id"].dropna().astype(int).tolist()
        context.user_data["focus_list_ids"]   = ids
        context.user_data["focus_list_label"] = label
    except Exception:
        pass

def _get_focus_list_df(active_df: "pd.DataFrame", context) -> "pd.DataFrame | None":
    """החזר DataFrame מסונן לפי focus_list הנוכחי, או None אם אין."""
    ids = context.user_data.get("focus_list_ids")
    if not ids:
        return None
    try:
        sub = active_df[active_df["bottle_id"].astype(int).isin(ids)]
        return sub if not sub.empty else None
    except Exception:
        return None

def _clear_focus_list(context):
    context.user_data.pop("focus_list_ids", None)
    context.user_data.pop("focus_list_label", None)

# מילות trigger לשאלות follow-up על רשימה
_LIST_FOLLOWUP_TRIGGERS = (
    "מהם", "מאלו", "מביניהם", "מבינהם", "מהרשימה",
    "מה מ", "מי מ", "איזה מ", "איזה מהם", "איזה מאלו",
    "from them", "from those", "among them", "of those", "of them",
    "which of", "which one",
    # group extreme follow-ups — רק עם כינוי scope מפורש (מביניהם/מהם/מאלו)
    # "הכי vfm" לבד ללא כינוי scope לא אמור להיחשב follow-up על רשימה
    "מביניהם הכי",
    "הכי vfm מביניהם", "הכי vfm מהם", "הכי vfm מאלו",
    "הכי אלכוהולי מביניהם", "הכי אלכוהולי מהם",
    "הכי ממוקדם מביניהם", "הכי מוקדם מביניהם",
    "נשאר הכי קצת מביניהם", "הכי מעט מביניהם", "עומד להסתיים מביניהם",
)

def _is_list_followup(user_text: str) -> bool:
    """האם השאלה מתייחסת לרשימת הבקבוקים האחרונה?"""
    t = _normalize_text(user_text)
    return any(trigger in t for trigger in _LIST_FOLLOWUP_TRIGGERS)

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

_UPDATE_FLOW_TRIGGERS = (
    "עדכן בקבוק", "עדכון בקבוק", "עדכן שתיה", "עדכן שתייה",
    "update bottle", "log drink", "log dram",
)

def _looks_like_update_trigger(text: str) -> bool:
    t = _normalize_text(text)
    return any(_normalize_text(tr) in t for tr in _UPDATE_FLOW_TRIGGERS)

def _get_update_flow(context) -> dict | None:
    return context.user_data.get("update_flow")

def _set_update_flow(context, data: dict):
    context.user_data["update_flow"] = data

def _clear_update_flow(context):
    context.user_data.pop("update_flow", None)

def _get_known_drinkers_from_df() -> list[str]:
    """Extract distinct drinker names from the cached DF (known_drinkers column)."""
    try:
        df = get_all_data_as_df()
        if df is None or df.empty or "known_drinkers" not in df.columns:
            return []
        val = df["known_drinkers"].dropna().iloc[0] if not df["known_drinkers"].dropna().empty else None
        if val is None:
            return []
        return normalize_to_list(val)
    except Exception as e:
        logging.warning(f"Could not extract drinker names from DF: {e}")
        return []

def _build_drinker_keyboard(known: list[str], selected: list[str]) -> "InlineKeyboardMarkup":
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = []
    for name in known:
        tick = "✅" if name in selected else "◻️"
        buttons.append([InlineKeyboardButton(f"{tick} {name}", callback_data=f"drk_toggle:{name}")])
    buttons.append([InlineKeyboardButton("➕ הוסף שם חדש", callback_data="drk_add_new")])
    buttons.append([InlineKeyboardButton("✔️ אישור", callback_data="drk_confirm")])
    return InlineKeyboardMarkup(buttons)


def _looks_like_count_query(text: str) -> bool:
    t = _normalize_text(text)
    return any(h in t for h in _COUNT_HINTS) and (any(h in t for h in _BOTTLE_HINTS) or any(h in t for h in _HAVE_HINTS))


# ── Extremes-on-distillery: "מה הכי מתוק מבקבוקי Glenfiddich" ──
_EXTREMES_GROUP_SCOPE_RE = re.compile(
    r"(מ(?:בקבוקי|בקבוקים\s*של|ביניהם|אלו|הם|בקוקי|קבוקי|קבוקים)|"
    r"מה(?:בקבוקים|בקבוקי)\s*של|"
    r"בקבוק\s+\w|"                          # "מה הבקבוק Glenmorangie הכי..."
    r"של\s+מזקקת?|"
    r"של\s+\S|"                             # "הכי מתוק של M&H / של Glenfiddich"
    r"from\s+(?:the\s+)?bottles?\s+of|among\s+(?:the\s+)?)",
    re.IGNORECASE
)

def _extract_distillery_scope_from_extremes(user_text: str, active_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    אם שאלת extremes מציינת מזקקה ספציפית (למשל "מה הכי מתוק מבקבוקי M&H"),
    מחזיר DataFrame מסונן לאותה מזקקה.
    מחזיר None אם לא זוהתה מזקקה ספציפית.
    """
    if not _EXTREMES_GROUP_SCOPE_RE.search(user_text):
        return None

    # נסה fuzzy match על שם מזקקה בטקסט
    # קודם ננקה מילי "הכי מתוק / מעושן" כדי לקבל שם מזקקה נקי
    clean = re.sub(
        r"(מה|הכי|מתוק|מעושן|עדין|עשיר|סמיך|מ(?:בקבוקי|בקבוקים|ביניהם|אלו)|"
        r"שלי|של|ה|בקבוק|בקבוקי|sweet|smoky|delicate|rich|\?|!)",
        " ", user_text, flags=re.IGNORECASE
    )
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None

    dist_match = find_best_distillery_match(clean, active_df)
    if dist_match.get("best") and float(dist_match.get("score") or 0) >= 0.60:
        dist = dist_match["best"]
        sub = active_df[active_df["distillery"].astype(str) == str(dist)]
        return sub if not sub.empty else None
    return None


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

        if "עשירים/סמיכים" in title or "עדינים" in title:
            over_score = max_2
        else:
            over_score = max_score
            
        for idx in range(n):
            r = rows.iloc[idx]
            score = float(r.get(score_col))
            lines.append(_fmt_line(idx + 1, str(r.get("distillery", "-")), str(r.get("bottle_name", "-")), score, over_score))

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
    # שאלות מידע שמכילות מילות עדכון אבל הן לא עדכון
    if re.search(r"^מתי\b", t):
        return False
    if re.search(r"\b(לאחרונה|אחרון|last time|when did|מתי שתיתי|history)\b", t):
        return False
    return any(h in t for h in _UPDATE_HINTS)


# ==========================================
# History time-range query (bottles I drank in last N days)
# ==========================================

def _looks_like_history_timerange_query(text: str) -> bool:
    """
    Detects queries like:
    - "איזה בקבוקים שתיתי מהם בשבוע האחרון?"
    - "מה שתיתי ב-10 ימים האחרונים?"
    - "אילו בקבוקים שתיתי בחודש האחרון?"
    """
    t = _normalize_text(text)
    has_drink = bool(re.search(r"(שתיתי|טעמתי|שתה)", t))
    has_time  = bool(re.search(
        r"(שבוע\s*האחרון|חודש\s*האחרון|\d+\s*ימים?\s*אחרונים?|ימים?\s*האחרונים?|"
        r"last\s*\d*\s*(day|week|month|days|weeks)|בשבוע|בחודש)", t))
    return has_drink and has_time


def _extract_days_from_timerange(text: str) -> int:
    """
    Extracts number of days from time-range expressions.
    - "שבוע אחרון" -> 7
    - "חודש אחרון" -> 30
    - "10 ימים אחרונים" -> 10
    - "last 5 days" -> 5
    - "last week" -> 7
    - "last month" -> 30
    """
    t = _normalize_text(text)

    # explicit number of days: "10 ימים" / "10 days"
    m = re.search(r"(\d+)\s*(ימים?|days?)", t)
    if m:
        return int(m.group(1))

    # week
    if re.search(r"(שבוע|week)", t):
        return 7

    # month
    if re.search(r"(חודש|month)", t):
        return 30

    # fallback
    return 7


def query_bottles_drunk_in_last_n_days(n_days: int, df: pd.DataFrame) -> list[dict]:
    cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(days=n_days)
    
    filtered = df[df["updating_time"] >= cutoff]
    
    grouped = (
        filtered
        .groupby(["distillery", "bottle_name"], dropna=True)
        .agg(
            total_volume_ml=("total_volume_consumed", "sum"),
            total_drams=("total_drams", "sum")
        )
        .reset_index()
        .sort_values(["distillery", "bottle_name"])
    )
    
    return grouped.round({"total_volume_ml": 1}).to_dict(orient="records")


def build_history_timerange_reply(n_days: int, bottles: list[dict]) -> str:
    sep = "━━━━━━━━━━━━━━━━━━"
    if not bottles:
        return f"{sep}\n🗓️ ב-{n_days} הימים האחרונים\n{sep}\n\nלא שתיתי בקבוקים בתקופה זו 🤷"

    lines = []
    for b in bottles:
        distillery   = b.get("distillery") or "?"
        bottle_name  = b.get("bottle_name", "")
        volume       = b.get("total_volume_ml", 0)
        drams        = int(b.get("total_drams", 0))
        lines.append(f"🥃 {distillery} – {bottle_name}\n   📏 {volume} מ\"ל | ב- 🥛 {drams} כוסות")

    body = "\n\n".join(lines)
    total_drams  = sum(int(b.get("total_drams", 0)) for b in bottles)
    total_volume = round(sum(b.get("total_volume_ml", 0) for b in bottles), 1)

    return (
        f"{sep}\n"
        f"🗓️ בקבוקים ששתיתי ב-{n_days} הימים האחרונים\n"
        f"{sep}\n\n"
        f"{body}\n\n"
        f"{sep}\n"
        f"סה\"כ: {len(bottles)} בקבוקים שונים | {total_drams} דרמים | {total_volume} מ\"ל"
    )

def _extract_amount_ml(text: str) -> int | None:
    """
    Returns ml amount from text.
    If only glass/dram count given (e.g. '2 כוסות'), converts to ml (x30).
    If nothing found -> returns None (caller will use default 30ml / 1 dram).
    """
    t = text.replace("מ״ל", "ml").replace('מ"ל', "ml").replace("מ''ל", "ml")

    # explicit ml amount: "60ml", "60 ml"
    m = re.search(r"(\d{1,4})\s*(ml)\b", t, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    # glass/dram count: "2 כוסות", "3 דראמים", "2 glasses", "2 drams"
    m_glasses = re.search(
        r"(\d{1,2})\s*(כוס(?:ות)?|דראם(?:ים)?|glass(?:es)?|dram(?:s)?)",
        t, flags=re.IGNORECASE
    )
    if m_glasses:
        return int(m_glasses.group(1)) * 30

    # bare number (not a year, not an age)
    m2 = re.search(r"\b(\d{1,4})\b", t)
    if m2:
        n = int(m2.group(1))
        if n < 1900:
            return n

    return None


def _extract_glass_count(text: str, amount_ml: int) -> int:
    """
    Returns number of drams/glasses to log in drams_counter.
    Priority:
      1. Explicit glass/dram count in text ("2 כוסות", "3 דראמים")
      2. Derive from ml: round(amount_ml / 30), min 1
      3. Default: 1
    """
    t = text.replace("מ״ל", "ml").replace('מ"ל', "ml").replace("מ''ל", "ml")

    m = re.search(
        r"(\d{1,2})\s*(כוס(?:ות)?|דראם(?:ים)?|glass(?:es)?|dram(?:s)?)",
        t, flags=re.IGNORECASE
    )
    if m:
        return max(1, int(m.group(1)))

    return max(1, round(amount_ml / 30))

def _extract_palette_from_text(text: str) -> list[str] | None:
    """
    Extracts palette/taste flavors from text if the user mentioned tasting notes.
    Trigger keywords: טעימה, טעמתי, טעם, טעמים, palate, palette, taste, flavor, פלט
    Returns a list of flavor strings, or None if not mentioned.
    Example: "שתיתי 60ml, טעימה: chocolate, coffee, citrus" -> ["chocolate","coffee","citrus"]
    """
    t = text

    # Find trigger keyword and capture everything after it
    m = re.search(
        r"(?:טעימה|טעמתי|טעמים|טעם|palate|palette|taste|flavor|פלט)\s*[:\-–]?\s*(.+?)(?:\.|$|\n|ארומה|nose|אחוז|abv|%)",
        t, flags=re.IGNORECASE
    )
    if not m:
        return None

    raw = m.group(1).strip()
    if not raw:
        return None

    # Split by common separators: comma, slash, semicolon, Hebrew ו
    parts = re.split(r"[,/;]+|\bו\b", raw)
    result = [p.strip().strip("'\".,") for p in parts if p.strip()]
    return result if result else None


def _extract_nose_from_text(text: str) -> list[str] | None:
    """
    Extracts nose/aroma notes from text if the user mentioned aromas.
    Trigger keywords: ארומה, ריח, נוז, nose, aroma
    Returns a list of aroma strings, or None if not mentioned.
    Example: "ארומה: vanilla, honey, oak" -> ["vanilla","honey","oak"]
    """
    t = text

    m = re.search(
        r"(?:ארומה|ריח|נוז|nose|aroma)\s*[:\-–]?\s*(.+?)(?:\.|$|\n|טעימה|palate|palette|אחוז|abv|%)",
        t, flags=re.IGNORECASE
    )
    if not m:
        return None

    raw = m.group(1).strip()
    if not raw:
        return None

    parts = re.split(r"[,/;]+|\bו\b", raw)
    result = [p.strip().strip("'\".,") for p in parts if p.strip()]
    return result if result else None


def _extract_abv_from_text(text: str) -> float | None:
    """
    Extracts a new ABV percentage from text if the user mentioned it.
    Patterns: "50% אלכוהול", "ABV 46", "עומד על 43%", "אחוז אלכוהול 46"
    Returns float or None if not mentioned.
    """
    t = text

    patterns = [
        r"(\d{2,3}(?:\.\d)?)\s*%\s*(?:אלכוהול|alc|abv)",
        r"(?:abv|אלכוהול|alc)\s*[:\-–]?\s*(\d{2,3}(?:\.\d)?)\s*%?",
        r"עומד על\s+(\d{2,3}(?:\.\d)?)\s*%",
        r"אחוז אלכוהול\s+(\d{2,3}(?:\.\d)?)",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1))
            # sanity: ABV must be between 20 and 95
            if 20.0 <= val <= 95.0:
                return val
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

_FOCUS_BACK_PRONOUNS_RE = re.compile(
    r"\b(הוא|בו|בה|אותו|אותה|עליו|עליה|שלו|שלה|זה|הזה|הבקבוק הזה)\b",
    re.IGNORECASE
)

def _is_general_portfolio_query(user_text: str) -> bool:
    """
    Returns True if this looks like a general portfolio question
    (filtering all bottles by flavor/cask/etc.) — NOT a follow-up
    about the last specific bottle.

    Logic:
    - Must match looks_like_text_intent (has flavor/cask keywords)
    - Must NOT contain focus-back pronouns (הוא / בו / שלו / זה...)
    - Must NOT contain a specific bottle name from the DB
      (we can't check DB here, so we use a simple heuristic:
       if no pronoun and looks_like_text_intent -> treat as general)
    """
    if not looks_like_text_intent(user_text):
        return False
    # if the text has a pronoun pointing back to a focus bottle -> not general
    if _FOCUS_BACK_PRONOUNS_RE.search(user_text or ""):
        return False
    return True


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
                for tok in normalize_to_list(it):  # FIX: was outside inner loop
                    tok = tok.strip()
                    if tok:
                        out.add(tok)
        else:
            # scalar from DB may be raw string like "['Honey']" — normalize it
            for tok in normalize_to_list(str(x)):
                tok = tok.strip()
                if tok:
                    out.add(tok)
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
    lines.append(f"Casks: {fmt_list(p.get('casks_aged_in'))}")
    lines.append(f"Nose: {fmt_list(p.get('nose'))}")
    lines.append(f"Palette: {fmt_list(p.get('palette'))}")
    lines.append(f"Volume (ml): {p.get('orignal_volume','-')}")
    lines.append(f"Rarity: {p.get('rarity') or '-'}")
    lines.append(f"Special bottling: {p.get('special_bottling')}")
    lines.append(f"Limited edition: {p.get('limited_edition')}")
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

    # If already list/tuple -> flatten each element
    if isinstance(x, (list, tuple)):
        out = []
        for it in x:
            out.extend(_normalize_vision_list(it))
        return out

    s = str(x).strip()
    if not s:
        return []

    # Case: looks like a bracketed list (full string is "[...]")
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        # First try to extract quoted tokens: "['Honey']" -> ["Honey"]
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
        # Each part may still be a bracket-wrapped token like "['Honey']" — clean it
        cleaned = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if (p.startswith("[") and p.endswith("]")) or (p.startswith("(") and p.endswith(")")):
                cleaned.extend(_normalize_vision_list(p))
            else:
                cleaned.append(p.strip("'\""))
        return [c for c in cleaned if c]

    # Single token — strip any stray bracket/quote wrappers
    s = s.strip("[]()").strip("'\"").strip()
    return [s] if s else []

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
2. Identify distillery and origin. Recognize logos (e.g. 'M&H' -> 'Milk & Honey', Israel).
3. Analyze cask type.
4. Infer sensory profile: if missing, generate accurate keywords based on cask + distillery + climate.

### PHASE 2: STRICT OPERATIONAL PROTOCOLS
- Exact match only. Do not conflate versions.
- If unknown and cannot be inferred, return null.
- Volumes in ml.
- limited: true if 'Single Cask' or 'Small Batch'.
- special: true if 'Distillery Exclusive' or 'Single Cask'.

### PHASE 3: RARITY CLASSIFICATION
Classify rarity into exactly one of these options:
- "Core Range" — standard permanent release, widely available
- "Limited Edition" — time-limited or seasonal release (look for "Limited Edition", "Special Release", year statements without cask number, or numbered bottles like 029/150)
- "Small Batch" — explicitly states "Small Batch"
- "Local Distillery Core Range" — standard release from a small/local distillery (e.g. Israeli, Taiwanese, Indian)
- "Private Release Blend" — private bottling, blended whisky
- "Single Cask" — single cask without specific bottler branding (look for Cask No., Butt No., Barrel No., or similar cask identifiers)
- "Private Release Single Cask" — single cask from an independent bottler
- "Distillery Edition Single Cask" — single cask released directly by the distillery

Inference rules:
1. If you see a cask number (e.g. "Cask No. 123", "Butt No. 5", "Barrel #42") AND a year -> "Single Cask" or "Distillery Edition Single Cask"
2. If you see a bottle number out of total (e.g. "029/150", "Bottle 12 of 200") -> "Limited Edition" unless also a single cask
3. If both cask number AND numbered bottles -> "Distillery Edition Single Cask"
4. If "Small Batch" appears explicitly -> "Small Batch"
5. If "Private" or independent bottler name appears -> "Private Release Single Cask" or "Private Release Blend"
6. Default for unknown local distilleries -> "Local Distillery Core Range"
7. Default for known major distilleries with no special markers -> "Core Range"

### PHASE 4: OUTPUT SCHEMA
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
  "rarity": string,
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

RARITY_OPTIONS = [
    "Core Range",
    "Limited Edition",
    "Small Batch",
    "Local Distillery Core Range",
    "Private Release Blend",
    "Single Cask",
    "Private Release Single Cask",
    "Distillery Edition Single Cask",
]

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
    dist_opts = _unique_from_scalar_col(active_df, "distillery")

    out["alcohol_type"] = _map_to_closest(out.get("alcohol_type"), alcohol_type_opts)
    out["origin_country"] = _map_to_closest(out.get("origin_country"), country_opts)
    out["region"] = _map_to_closest(out.get("region"), region_opts)

    # Normalize distillery: prefer existing name in DB over Gemini's version
    raw_dist = out.get("distillery")
    if raw_dist and dist_opts:
        mapped_dist = _map_to_closest(raw_dist, dist_opts)
        # Only override if the match is reasonably close (Levenshtein will pick best)
        if mapped_dist:
            out["distillery"] = mapped_dist

    out["casks_aged_in"] = _map_list_to_options(normalize_to_list(out.get("casks")), casks_opts)
    out["nose"] = _map_list_to_options(normalize_to_list(out.get("nose")), nose_opts)
    out["palette"] = _map_list_to_options(normalize_to_list(out.get("palate")), pal_opts)

    # normalize booleans
    out["special_bottling"] = bool(out.get("special"))
    out["limited_edition"] = bool(out.get("limited"))

    # Rarity: validate against controlled vocab; keep Gemini's value if it matches, else None
    raw_rarity = out.get("rarity")
    if raw_rarity and isinstance(raw_rarity, str):
        # exact match first
        if raw_rarity in RARITY_OPTIONS:
            out["rarity"] = raw_rarity
        else:
            # fuzzy fallback to closest option
            mapped = _map_to_closest(raw_rarity, RARITY_OPTIONS)
            out["rarity"] = mapped if mapped else None
    else:
        out["rarity"] = None

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

    val_rarity = _sql_str_or_null(p.get("rarity"))

    sql = f"""
    BEGIN
    DECLARE ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP();

    INSERT INTO `{TABLE_REF}`
    (bottle_id, bottle_name, distillery, alcohol_type, origin_country, region,
    casks_aged_in, nose, palette, age, alcohol_percentage, price,
    limited_edition, special_bottling, was_a_gift, stock_status_per, full_or_empy,
    orignal_volume, date_of_purchase, opening_date, bottle_counter,
    was_discounted, discount_amount, discounted_price, time_of_registration,
    updating_time, rarity)
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
        ts,
        {val_rarity}
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
        "candidates": [ {"full_name":..., "bottle_id":..., "score":...}, ... up to 5],
        "duplicates": [ {"full_name":..., "bottle_id":..., "stock_status_per":...}, ... ]
          -- populated when multiple distinct bottle_ids share the same best full_name
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
        stock = r.get("stock_status_per", None)
        scored.append((full, int(r["bottle_id"]), s, stock))

    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:5]
    candidates = [{"full_name": a, "bottle_id": b, "score": round(c, 3)} for a, b, c, _ in top]
    if not candidates:
        return {"best_name": None, "bottle_id": None, "score": 0.0, "candidates": [], "duplicates": []}

    best = candidates[0]

    # Detect duplicate bottle_ids sharing the same full_name at the top score
    best_name_norm = _normalize_text(best["full_name"])
    best_score = best["score"]
    duplicates = [
        {"full_name": a, "bottle_id": b, "stock_status_per": stk}
        for a, b, c, stk in scored
        if _normalize_text(a) == best_name_norm and round(c, 3) >= best_score - 0.01
    ]
    # Only flag as duplicates when there are truly multiple distinct bottle_ids
    if len(duplicates) <= 1:
        duplicates = []

    return {
        "best_name": best["full_name"],
        "bottle_id": best["bottle_id"],
        "score": best["score"],
        "candidates": candidates,
        "duplicates": duplicates,
    }
# ==========================================
# Update execution
# ==========================================
def execute_drink_update(
    bottle_id: int,
    amount_ml: int,
    inventory_dict: dict,
    glasses_cnt: int = 1,
    new_palette: list | None = None,
    new_nose: list | None = None,
    new_abv: float | None = None,
    drinkers: list | None = None,
):
    if bottle_id not in inventory_dict:
        return False, "❌ לא מצאתי את ה-ID הזה במלאי."
    print(bottle_id)
    b_data = inventory_dict[bottle_id]
    vol = b_data["vol"] or 700
    drank_per = (amount_ml / vol) * 100
    new_stock_per = round(max(b_data["stock"] - drank_per, 0), 2)
    is_empty = str(new_stock_per == 0.0).lower()

    # ── Palette: use new if provided, else keep old ──
    f_palette = normalize_to_list(new_palette) if new_palette else normalize_to_list(b_data["old_palette"])

    # ── Nose: use new if provided, else keep old ──
    f_nose = normalize_to_list(new_nose) if new_nose else normalize_to_list(b_data["old_nose"])

    # ── ABV: validate — can only go down ──
    old_abv = float(b_data["old_abv"]) if b_data["old_abv"] else 0.0
    if new_abv is not None:
        if new_abv >= old_abv:
            return False, (
                f"⚠️ לא ניתן לעדכן אחוז אלכוהול גבוה יותר!\n"
                f"הערך הנוכחי הוא {old_abv}% — הערך שהזנת ({new_abv}%) גבוה ממנו או שווה לו.\n"
                f"אלכוהול רק יורד עם הזמן. תבדוק שוב את הערך."
            )
        f_abv = new_abv
    else:
        f_abv = old_abv

    safe_name   = _escape_sql_str(b_data["name"])
    sql_nose    = sql_array(f_nose)
    sql_palette = sql_array(f_palette)

    if drinkers:
            clean_drinkers = [str(d).strip() for d in drinkers if str(d).strip()]
            sql_drinker = sql_array(clean_drinkers)
    else:
        sql_drinker = "[]"

    sql = f"""
    -- ===== DECLARES FIRST =====
    DECLARE upd_ts TIMESTAMP;
    DECLARE upd_date DATE;
    DECLARE consumption_match BOOL;
    DECLARE forecasted_bid INT64;
    DECLARE depletion_match BOOL;
    DECLARE consumption_flag_str STRING;
    DECLARE depletion_flag_str STRING;

    -- ===== SET =====
    SET upd_ts   = CURRENT_TIMESTAMP();
    SET upd_date = DATE(upd_ts);

    SET consumption_match = EXISTS (
        SELECT 1 FROM `{FORECAST_TABLE_REF}`
        WHERE DATE(est_consumption_date) = upd_date
        LIMIT 1
    );
    SET forecasted_bid = (
        SELECT bottle_id FROM `{FORECAST_TABLE_REF}`
        WHERE DATE(est_consumption_date) = upd_date
        LIMIT 1
    );

    SET depletion_match = FALSE;
    IF {new_stock_per} = 0 THEN
        SET depletion_match = EXISTS (
            SELECT 1 FROM `{FORECAST_TABLE_REF}`
            WHERE DATE(predicted_finish_date) = upd_date
            LIMIT 1
        );
    END IF;

    SET consumption_flag_str = IF(consumption_match, 'True', 'False');
    SET depletion_flag_str   = IF(depletion_match,  'True', 'False');

    -- ===== UPDATE MAIN TABLE =====
    UPDATE `{TABLE_REF}`
    SET stock_status_per  = {new_stock_per},
        full_or_empy      = {is_empty},
        updating_time     = upd_ts
    WHERE bottle_id = {bottle_id};

    -- ===== INSERT HISTORY =====
    INSERT INTO `{HISTORY_TABLE_REF}`
    (
        update_id, bottle_id, bottle_name, stock_status_per,
        update_time, drams_counter, nose, palette, alc_pre,
        consumption_flag, depletion_flag, forecasted_bottle_id, drinker_name
    )
    VALUES (
        (SELECT COALESCE(MAX(update_id), 0) + 1 FROM `{HISTORY_TABLE_REF}`),
        {bottle_id},
        '{safe_name}',
        {new_stock_per},
        upd_ts,
        {glasses_cnt},
        {sql_nose},
        {sql_palette},
        {f_abv},
        consumption_flag_str,
        depletion_flag_str,
        IF(consumption_match, forecasted_bid, NULL),
        {sql_drinker}
    );
    """

    job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
    bq_client.query(sql, job_config=job_config).result()

    # invalidate cache
    global CACHE_DATA
    CACHE_DATA["df"] = None

    # ── Build confirmation message ──
    lines = [
        f"✅ עדכון בוצע!",
        f"🥃 {b_data['name']}",
        f"📉 ירד ל-{new_stock_per}% (הפחתה של ~{round(drank_per, 2)}%)",
        f"🥃 דראמים שנרשמו: {glasses_cnt}",
    ]
    if drinkers:
        lines.append(f"👤 שתה/ו: {', '.join(drinkers)}")
    if new_palette:
        lines.append(f"🍫 Palette עודכן: {', '.join(f_palette)}")
    if new_nose:
        lines.append(f"👃 Nose עודכן: {', '.join(f_nose)}")
    if new_abv is not None:
        lines.append(f"🔢 ABV עודכן: {f_abv}%")

    return True, "\n".join(lines)


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
    
    # ✅ תמיד עבוד על בקבוקים פעילים בלבד
    if "stock_status_per" in df.columns:
        df = df[pd.to_numeric(df["stock_status_per"], errors="coerce").fillna(0) > 0].copy()

    if df.empty:
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
                "ממוצע שתייה", "כמה אני שותה ממנו בממוצע", "בממוצע יומי",
                "popular", "פופולריות", "כמה נשתה", "צריכה יומית", "ml ליום",
                "average consumption", "avg consumption", "daily consumption"
            ],
            "avg_days_between_updates": [
                "כל כמה זמן", "כל כמה ימים", "כל כמה שבועות",
                "כמה זמן בין שתייה לשתייה", "תדירות שתייה", "בממוצע כל כמה",
                "how often", "frequency", "days between", "avg days between",
            ],
            "latest_consumption_time": [
                "מתי שתיתי ממנו לאחרונה", "מתי שתיתי ממנו",
                "מתי טעמתי ממנו לאחרונה", "מתי טעמתי ממנו",
                "ממנו לאחרונה", "מתי לאחרונה", "שתיתי ממנו לאחרונה",
                "פעם אחרונה", "שתייה אחרונה", "תאריך שתייה אחרון",
                "last drink", "last time", "last tasted", "last consumed",
                "when did i last", "most recent drink", "אחרון"
            ],
            "predicted_finish_date": [
                "מתי הוא צפוי להיגמר", "מתי ייגמר", "מתי יגמר",
                "מתי הבקבוק ייגמר", "מתי יסתיים", "יסתיים",
                "finish date", "predicted finish", "מתי נגמר",
                "צפי גמר", "מתי ייגמר הבקבוק"
            ],
            "est_consumption_date": [
                "מתי לשתות ממנו", "מתי כדאי לשתות", "המלצה מתי לשתות",
                "recommend date", "דראם הבא מתי", "מה לשתות עכשיו",
                "מתי כדאי לפתוח", "est consumption", "estimated date"
            ],
            "Best_Before": [
                "עד מתי כדאי לשתות אותו", "עד מתי כדאי לשתות ממנו",
                "best before", "תוקף", "מומלץ עד", "לפני שיתחמצן",
                "עד מתי", "תאריך תפוגה", "oxidation", "חמצון"
            ],
            "orignal_volume": [
                "מה הנפח שלו", "כמה ml בבקבוק", "נפח הבקבוק", "נפח",
                "volume", "גודל בקבוק", "original volume",
                "700ml", "1000ml", "700", "1000", "כמה מ\"ל"
            ],
            "current_status": [
                "כמה נשאר", "כמה נשאר ממנו", "אחוז שנשאר", "אחוז נשאר",
                "remaining", "left", "סטוק", "מלאי",
                "how much left", "how much is left", "percent left",
                "כמה יש ממנו", "כמה יש לי ממנו"
            ],
            "alcohol_percentage": [
                "אלכוהול", "אחוז אלכוהול", "abv", "strength",
                "כמה אחוז", "כמה אלכוהול", "alcohol percentage",
                "how strong", "proof"
            ],
            "age": [
                "גיל", "בן כמה", "כמה שנים", "age statement",
                "aged", "how old", "ישן כמה שנים", "שנות יישון"
            ],
            "price": [
                "מחיר", "כמה עלה", "עלות", "₪", "שקל", "שקלים",
                "price", "cost", "how much did it cost", "כמה שילמתי"
            ],
            "casks_aged_in": [
                "חבית", "חביות", "cask", "aged in", "יישון",
                "בחבית", "שרי", "sherry", "bourbon", "בורבון",
                "oloroso", "pedro ximenez", "oak", "עץ"
            ],
            "nose": [
                "nose", "נוז", "ארומות", "ריח", "ריחות",
                "aroma", "aromas", "smell", "sniff"
            ],
            "palette": [
                "palate", "פלטה", "טעמים", "טעם", "taste",
                "flavors", "flavours", "flavor profile"
            ],
        }

        # ── דטרמיניסטי: מזהים hint_column לפני שGemini מחליט ──
        def _resolve_hint_column(text: str, glossary: dict) -> str | None:
            t = re.sub(r"\s+", " ", text.strip().lower())
            best_col, best_len = None, 0
            for col, keywords in glossary.items():
                for kw in keywords:
                    kw_norm = re.sub(r"\s+", " ", kw.strip().lower())
                    if kw_norm and kw_norm in t and len(kw_norm) > best_len:
                        best_col, best_len = col, len(kw_norm)
            return best_col

        hint_col = _resolve_hint_column(user_text, COLUMN_GLOSSARY)

        system = (
            "You convert user questions into a STRICT JSON query plan over a pandas DataFrame.\n"
            "Return JSON ONLY.\n"
            "You MUST use only columns that exist in the provided schema.\n"
            "Never write SQL.\n\n"

            "COLUMN SELECTION RULES (CRITICAL — HIGHEST PRIORITY):\n"
            "You are given a COLUMN_GLOSSARY that maps user intents/synonyms to exact column names.\n"
            "You are also given hint_column: if it is not null, you MUST use it as the primary select column. No exceptions.\n"
            "Even if you think another column is more relevant, if hint_column is set — use it.\n"
            "The glossary was resolved deterministically before you were called. Trust it completely.\n"
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

            "FOCUS RULES (CRITICAL — READ IN ORDER):\n"
            "Rule 1: If focus_bottle is None → do NOT filter by bottle_id. Answer for the ENTIRE collection.\n"
            "Rule 2: If focus_bottle is provided AND the user uses pronouns (שלו/הוא/זה/בו/עליו/אותו) "
            "→ add a filter: bottle_id == focus.bottle_id.\n"
            "Rule 3: If focus_bottle is provided BUT the question contains no pronouns "
            "(e.g. 'which bottles have Chocolate flavor?') → IGNORE the focus. Answer for the ENTIRE collection.\n"
            "Rule 4: If bottle_id is missing from schema, filter by full_name contains focus.full_name.\n\n"

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
            "column_glossary": COLUMN_GLOSSARY,
            "hint_column": hint_col,   # דטרמיניסטי — Gemini חייב להשתמש בזה אם לא None
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
    
    
async def try_gemini_df_query_answer(user_text: str, df: pd.DataFrame, context) -> str | None:
    """Use Gemini to build a strict df_query plan and execute it. Returns a user reply, or None."""
    focus = None
    bid = context.user_data.get("focus_bottle_id")
    full = context.user_data.get("focus_full_name")
    if bid:
        focus = {"bottle_id": int(bid), "full_name": full}

    plan = gemini_make_df_query_plan(user_text, df, focus=focus)
    

    if plan and isinstance(plan, dict):
        try:
            res = execute_df_query_plan(df, plan)
            return _df_to_telegram_text(res)
        except Exception as e:
            logging.warning(f"DF plan execution failed: {e}")
            # fall through to natural-language fallback
    # final fallback: free text answer (still grounded by schema sample)
    try:
        return await gemini_fallback_answer(user_text, df)
    except Exception as e:
        logging.warning(f"Gemini fallback failed: {e}")
        return None
    

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

# ==========================================
# שאלות ספציפיות על שדה בקבוק מסוים
# לדוגמה: "מה אחוז האלכוהול של Glenmorangie A Tale of Spices?"
#          "מה ה-rarity של Milk & Honey Biblical Origin?"
#          "מתי שתיתי Spey Tenne לאחרונה?"
#          "האם Glenfiddich Project XX מתוק?"
# ==========================================

# Detect patterns like "X של <bottle>" or "האם <bottle> Y?"
_SPECIFIC_BOTTLE_ABV_RE = re.compile(
    r"(אחוז\s*(?:אלכוהול)?|\babv\b|alcohol\s*percentage|כמה\s*אלכוהול|\bproof\b)",
    re.IGNORECASE
)
_SPECIFIC_BOTTLE_RARITY_RE = re.compile(
    r"\b(rarity|נדירות|נדיר|רריטי)\b",
    re.IGNORECASE
)
_SPECIFIC_BOTTLE_LAST_DRINK_RE = re.compile(
    r"(מתי\s*שתיתי|מתי\s*טעמתי|שתיתי.*לאחרונה|last\s*time\s*i\s*drank|last\s*drink|last\s*tasted)",
    re.IGNORECASE
)
_SPECIFIC_BOTTLE_SWEETNESS_RE = re.compile(
    r"(מתוק|מעושן|עדין|עשיר|סמיך|sweet|smoky|delicate|rich|האם.*מתוק|האם.*מעושן)",
    re.IGNORECASE
)
_SPECIFIC_BOTTLE_REMAINING_RE = re.compile(
    r"(כמה\s*נשאר|כמה\s*%|אחוז\s*נשאר|remaining|how\s*much\s*left|left\s*in)",
    re.IGNORECASE
)
_SPECIFIC_BOTTLE_VFM_RE = re.compile(
    r"\bvfm\b",
    re.IGNORECASE
)
_SPECIFIC_BOTTLE_FREQ_RE = re.compile(
    r"(כל\s*כמה|תדירות|באיזו\s*תדירות|how\s*often|frequency)",
    re.IGNORECASE
)


def _is_specific_bottle_question(user_text: str) -> bool:
    """
    מזהה שאלה שמכילה שם בקבוק ספציפי + שאלה על שדה כלשהו.
    שאלות כמו:
    - "האם Glenfiddich Project XX מתוק?"
    - "מה אחוז האלכוהול של Glenmorangie A Tale of Spices?"
    - "מה ה-rarity של Milk & Honey Biblical Origin?"
    - "מתי שתיתי Spey Tenne לאחרונה?"
    - "מה הטעמים של M&H Under The Bridge?"
    - "כמה נשאר ב GlenAllachie The 15th?"

    שאלות עם "הכי" (extremes) נדחות — הן שייכות לloגיקת ה-extremes/group.
    """
    if not user_text:
        return False
    t = _normalize_text(user_text)

    # "הכי מתוק / הכי מעושן" = extremes question, NOT a specific bottle question
    if re.search(r"\bהכי\b", t):
        return False

    has_field_intent = bool(
        _SPECIFIC_BOTTLE_ABV_RE.search(user_text) or
        _SPECIFIC_BOTTLE_RARITY_RE.search(user_text) or
        _SPECIFIC_BOTTLE_LAST_DRINK_RE.search(user_text) or
        _SPECIFIC_BOTTLE_SWEETNESS_RE.search(user_text) or
        _SPECIFIC_BOTTLE_REMAINING_RE.search(user_text) or
        _SPECIFIC_BOTTLE_VFM_RE.search(user_text) or
        _looks_like_flavors_of_bottle_query(user_text) or
        _looks_like_age_of_bottle_query(user_text) or
        _looks_like_oxidized_query(user_text)
    )

    if not has_field_intent:
        return False

    # Make sure it doesn't look like a pronoun-only follow-up (no specific bottle name)
    # by checking if there's at least some non-stopword content after removing field keywords
    clean = re.sub(
        r"(מה|האם|מתי|כמה|שתיתי|טעמתי|אחוז|אלכוהול|abv|rarity|נדירות|נשאר|"
        r"טעמים|הטעמים|ארומה|גיל|שנים|מתוק|מעושן|עדין|עשיר|נשאר|של|ב|לאחרונה|"
        r"\\?|!|:|-)",
        " ", _normalize_text(user_text), flags=re.IGNORECASE
    )
    clean = re.sub(r"\s+", " ", clean).strip()

    # If what's left is basically a pronoun only -> not a "specific bottle" question
    if _is_focus_placeholder(clean):
        return False

    return True


def try_handle_specific_bottle_question(
    user_text: str,
    active_df: pd.DataFrame,
    context: ContextTypes.DEFAULT_TYPE
) -> str | None:
    """
    מטפל בשאלות שבהן מוזכר שם בקבוק ספציפי + שדה כלשהו.
    מחזיר תשובה כטקסט, או None אם לא זוהה.
    """
    if not user_text or active_df is None or active_df.empty:
        return None

    # ── נקה את טקסט השאלה מ"מילות שאלה" לפני fuzzy match ──
    # זה מונע מ"מה ה vfm של Glenmorangie The Lasanta" להתאים גרוע
    _FIELD_NOISE_RE = re.compile(
        r"(^|\s)(מה|ה|של|האם|מתי|כמה|שתיתי|טעמתי|אחוז|אלכוהול|abv|vfm|"
        r"rarity|נדירות|נשאר|טעמים|הטעמים|ארומה|גיל|שנים|מתוק|מעושן|"
        r"עדין|עשיר|לאחרונה|ב|את|עם|על|the|של|is|does|what|when|how|much)(\s|$)",
        re.IGNORECASE
    )
    clean_for_match = _FIELD_NOISE_RE.sub(" ", user_text)
    clean_for_match = re.sub(r"[\?\!\:\-\.\,]", " ", clean_for_match)
    clean_for_match = re.sub(r"\s+", " ", clean_for_match).strip()

    # נסה קודם עם טקסט נקי, אחר כך עם הטקסט המלא
    match = find_best_bottle_match(clean_for_match, active_df)
    if not match.get("best_name") or float(match.get("score") or 0) < 0.62:
        match = find_best_bottle_match(user_text, active_df)
    if not match.get("best_name") or float(match.get("score") or 0) < 0.62:
        return None

    chosen = (match.get("candidates") or [None])[0]
    if not chosen or chosen.get("bottle_id") is None:
        return None

    sub = active_df[active_df["bottle_id"] == chosen["bottle_id"]]
    if sub.empty:
        return None

    row = sub.iloc[0]
    full_name = str(row.get("full_name") or row.get("bottle_name") or "").strip()
    sep = "━━━━━━━━━━━━━━━━━━"

    # ── קבע פוקוס ──
    _set_focus_bottle(context, row)

    # ── VFM ──
    if _SPECIFIC_BOTTLE_VFM_RE.search(user_text):
        vfm_val = pd.to_numeric(row.get("rvfm"), errors="coerce")
        if pd.isna(vfm_val):
            return f"⚠️ אין ערך VFM זמין עבור {full_name}."
        if vfm_val >= 9.5:
            verdict = "🟢 מחיר מציאה – חובה לרכוש!"
        elif vfm_val >= 7.201:
            verdict = "🟢 מחיר סבבה  – רכישה מומלצת"
        elif vfm_val >= 6.01:
            verdict = "🟡 מחיר הוגן – ממוצע בסדר"
        elif vfm_val >= 3.0:
            verdict = "🟠 יקר, אך שווה יחסית – לשקול בכובד ראש"
        else:
            verdict = "🔴 לא שווה בכלל – לא מומלץ לרכישה"
        return (
            f"{sep}\n💰 VFM – ערך לכסף\n{sep}\n\n"
            f"{full_name}\n\n"
            f"📊 ציון VFM: {float(vfm_val):.2f} / 10\n"
            f"{verdict}\n\n"
            f"ℹ️ סקאלה: 9.5–10 מציאה · 7.2–9.5 שווה · 6–7.2 הוגן · 3–6 יקר · 1–3 לא שווה"
        )

    # ── החמצון / Best Before ──
    if _looks_like_oxidized_query(user_text):
        bb = row.get("Best_Before")
        bb_dt = _safe_to_datetime(bb)
        if pd.isna(bb_dt):
            return f"🥃 {full_name}\n⚠️ אין לי Best Before מוגדר לבקבוק הזה."
        from datetime import datetime as _dt
        today = _dt.now().date()
        bb_date = bb_dt.date()
        warning = "\n⚠️ שים לב! הבקבוק מאבד טעמים, מומלץ לסיים בהקדם!" if bb_date < today else ""
        return (
            f"{sep}\n🧪 סטטוס חמצון (Best Before)\n{sep}\n\n"
            f"🥃 {full_name}\n"
            f"📅 Best Before: {bb_date}{warning}"
        )

    # ── ABV ──
    if _SPECIFIC_BOTTLE_ABV_RE.search(user_text):
        abv = row.get("alcohol_percentage")
        try:
            abv_val = float(abv) if pd.notnull(abv) else None
        except Exception:
            abv_val = None
        if abv_val is None:
            return f"🥃 {full_name}\n⚠️ אין לי ערך ABV לבקבוק הזה."
        return f"{sep}\n🔢 אחוז אלכוהול\n{sep}\n\n🥃 {full_name}\n💧 ABV: {abv_val}%"

    # ── Rarity ──
    if _SPECIFIC_BOTTLE_RARITY_RE.search(user_text):
        rarity = row.get("rarity") or row.get("Rarity")
        if rarity is None or (isinstance(rarity, float) and pd.isna(rarity)):
            return f"🥃 {full_name}\n⚠️ אין לי ערך Rarity לבקבוק הזה."
        return f"{sep}\n💎 Rarity\n{sep}\n\n🥃 {full_name}\n✨ Rarity: {rarity}"

    # ── מתי שתיתי לאחרונה ──
    if _SPECIFIC_BOTTLE_LAST_DRINK_RE.search(user_text):
        last_time = row.get("latest_consumption_time")
        dt = _safe_to_datetime(last_time)
        if pd.isna(dt):
            return f"🥃 {full_name}\n📅 אין לי רשומת שתייה לבקבוק הזה עדיין."
        return (
            f"{sep}\n🕒 שתייה אחרונה\n{sep}\n\n"
            f"🥃 {full_name}\n"
            f"📅 {str(dt.date())}"
        )

    # ── מתוק / מעושן / עדין / עשיר ──
    if _SPECIFIC_BOTTLE_SWEETNESS_RE.search(user_text):
        t = _normalize_text(user_text)
        ask_sweet_smoky = bool(re.search(r"(מתוק|מעושן|sweet|smoky)", t))
        ask_richness = bool(re.search(r"(עדין|עשיר|סמיך|delicate|rich)", t))

        out_parts = [f"{sep}\n🧾 פרופיל בקבוק\n{sep}\n\n{full_name}\n"]

        if ask_sweet_smoky:
            s_val = _safe_float(row.get("final_smoky_sweet_score"))
            if s_val is None:
                out_parts.append("⚠️ אין מדד מתוק↔עשן לבקבוק הזה.")
            else:
                label = _label_from_ranges(s_val, SWEETNESS_RANGES)
                out_parts.append(
                    f"🍯 מתוק↔עשן: {s_val:.2f}\n"
                    f"🏷️ פרופיל: {label}\n"
                    "ℹ️ נמוך=יותר מתוק · גבוה=יותר מעושן"
                )
        if ask_richness:
            r_val = _safe_float(row.get("final_richness_score"))
            if r_val is None:
                out_parts.append("\n⚠️ אין מדד עדין↔עשיר לבקבוק הזה.")
            else:
                label = _label_from_ranges(r_val, RICHNESS_RANGES)
                out_parts.append(
                    f"\n🌿 עדין↔עשיר: {r_val:.2f}\n"
                    f"🏷️ פרופיל: {label}\n"
                    "ℹ️ נמוך=יותר עדין · גבוה=יותר עשיר"
                )
        return "".join(out_parts).strip()

    # ── כל כמה זמן שותה ──
    if _SPECIFIC_BOTTLE_FREQ_RE.search(user_text):
        freq = row.get("avg_days_between_updates")
        try:
            freq_val = float(freq) if (freq is not None and not (isinstance(freq, float) and pd.isna(freq))) else None
        except Exception:
            freq_val = None
        if freq_val is None:
            return f"{sep}\n📅 תדירות שתייה\n{sep}\n\n🥃 {full_name}\n⚠️ Missing Data!"
        return f"{sep}\n📅 תדירות שתייה\n{sep}\n\n🥃 {full_name}\n🔄 שותה בממוצע כל {freq_val:.1f} ימים"

    # ── טעמים ──
    if _looks_like_flavors_of_bottle_query(user_text):
        return build_flavors_reply(row)

    # ── גיל ──
    if _looks_like_age_of_bottle_query(user_text):
        return build_age_reply(row)

    # ── כמה נשאר ──
    if _SPECIFIC_BOTTLE_REMAINING_RE.search(user_text):
        return build_stock_reply(row)

    return None


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

async def handle_final_confirm_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles upd_confirm:yes / upd_confirm:no InlineKeyboard callbacks
    from the guided update flow summary screen.
    """
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    update_flow = _get_update_flow(context)
    if not update_flow or update_flow.get("stage") != "await_final_confirm":
        await query.edit_message_text("הפעולה פגה. כתוב 'עדכן בקבוק' מחדש.")
        return

    if data == "upd_confirm:yes":
        df_all = get_all_data_as_df()
        active_upd = df_all[df_all["stock_status_per"] > 0].copy()
        inventory_dict = {
            int(r["bottle_id"]): {
                "name": r["full_name"],
                "stock": float(r["stock_status_per"]),
                "vol": float(r["orignal_volume"]) if pd.notnull(r.get("orignal_volume")) else 700.0,
                "old_nose": normalize_to_list(r.get("nose")),
                "old_palette": normalize_to_list(r.get("palette")),
                "old_abv": float(r["alcohol_percentage"]) if pd.notnull(r.get("alcohol_percentage")) else 0.0,
            }
            for _, r in active_upd.iterrows()
        }
        ok, msg = execute_drink_update(
            int(update_flow["bottle_id"]),
            int(update_flow["amount_ml"]),
            inventory_dict,
            update_flow["glasses_cnt"],
            drinkers=update_flow.get("drinkers"),
        )
        bottle_name = update_flow.get("full_name", "הבקבוק")
        _set_focus_bottle(context, {"bottle_id": int(update_flow["bottle_id"]), "full_name": bottle_name})
        _clear_update_flow(context)
        if ok:
            await query.edit_message_text(
                f"{msg}\n\n"
                f"🎉 הבקבוק *{bottle_name}* עודכן בהצלחה!",
                parse_mode="Markdown"
            )
        else:
            await query.edit_message_text(
                f"❌ העדכון של *{bottle_name}* נכשל:\n{msg}",
                parse_mode="Markdown"
            )
        return

    if data == "upd_confirm:no":
        update_flow["stage"] = "await_fix_field"
        _set_update_flow(context, update_flow)
        await query.edit_message_text(
            f"סיכום:\n"
            f"🥃 {update_flow['full_name']}\n"
            f"👤 {', '.join(update_flow.get('drinkers', []))}\n"
            f"📏 {update_flow['amount_ml']} מ\"ל\n"
            f"🥛 {update_flow['glasses_cnt']} כוסות\n\n"
            f"מה לתקן?\n1. בקבוק\n2. שותה\n3. כמות (מ\"ל)\n4. כוסות\n5. ביטול"
        )
        return


async def handle_drinker_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    update_flow = _get_update_flow(context)
    if not update_flow or update_flow.get("stage") != "await_drinker":
        await query.edit_message_text("הפעולה פגה. כתוב 'עדכן בקבוק' מחדש.")
        return

    known    = update_flow.get("known_drinkers", [])
    selected = update_flow.get("selected_drinkers", [])

    if data.startswith("drk_toggle:"):
        name = data[len("drk_toggle:"):]
        if name in selected:
            selected.remove(name)
        else:
            selected.append(name)
        update_flow["selected_drinkers"] = selected
        _set_update_flow(context, update_flow)
        kb = _build_drinker_keyboard(known, selected)
        await query.edit_message_reply_markup(reply_markup=kb)
        return

    if data == "drk_add_new":
        update_flow["adding_new_drinker"] = True
        _set_update_flow(context, update_flow)
        await query.edit_message_text("כתוב את השם החדש:")
        return

    if data == "drk_confirm":
        if not selected:
            await query.answer("בחר לפחות שם אחד.", show_alert=True)
            return
        update_flow["drinkers"] = selected
        update_flow["stage"] = "await_ml"
        _set_update_flow(context, update_flow)
        await query.edit_message_text(
            f"נבחרו: {', '.join(selected)}\n\nכמה מ\"ל נשתה? (מספר בלבד)"
        )
        return


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

                # ── Duplicate check ──────────────────────────────────────
                dup_found = False
                if not active_df.empty and "distillery" in active_df.columns and "bottle_name" in active_df.columns:
                    p_dist   = _normalize_text(payload.get("distillery") or "")
                    p_bottle = _normalize_text(payload.get("bottle_name") or "")
                    if p_dist and p_bottle:
                        for _, row in active_df.iterrows():
                            r_dist   = _normalize_text(str(row.get("distillery") or ""))
                            r_bottle = _normalize_text(str(row.get("bottle_name") or ""))
                            if r_dist == p_dist and r_bottle == p_bottle:
                                dup_found = True
                                break

                # Store payload
                _set_add_payload(context, payload)

                await update.message.reply_text("✅ סריקה הושלמה. הנה מה שחילצתי:")
                await update.message.reply_text(_format_add_summary(payload))

                if dup_found:
                    _set_add_stage(context, "await_duplicate_confirm")
                    await update.message.reply_text(
                        f"⚠️ שים לב: הבקבוק *{payload.get('bottle_name')}* של *{payload.get('distillery')}* "
                        f"כבר קיים במאגר שלך.\n"
                        f"האם ברצונך להוסיף עותק נוסף? (כן/לא)"
                    )
                else:
                    _set_add_stage(context, "await_price")
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


    # ----- HARD RULE: sweet/smoky & delicate/rich extremes + group extremes (VFM/BB/Stock/ABV) -----
    df = get_all_data_as_df()
    active_df_ext = df[df["stock_status_per"] > 0].copy() if df is not None and not df.empty else df

    # ── helper: scope detection (distillery or focus_list) ──
    def _resolve_group_scope():
        """Returns (scope_df, scope_label) or (None, None)."""
        # 1. explicit distillery in text
        dist_scope_df = _extract_distillery_scope_from_extremes(user_text, active_df_ext)
        if dist_scope_df is not None and not dist_scope_df.empty:
            dist_name = str(dist_scope_df.iloc[0].get("distillery", "המזקקה"))
            _set_focus_list(context, dist_scope_df, label=dist_name)
            return dist_scope_df, dist_name
        # 2. follow-up on existing focus_list
        if _is_list_followup(user_text):
            ldf = _get_focus_list_df(active_df_ext, context)
            if ldf is not None and not ldf.empty:
                return ldf, context.user_data.get("focus_list_label", "הרשימה")
        return None, None

    # ── group extremes on a distillery scope (VFM / Best Before / Stock / ABV / sweet-smoky) ──
    scope_df, scope_label = _resolve_group_scope()
    if scope_df is not None:
        # Try new group extremes handler first (VFM, BB, stock, ABV)
        group_ans = try_handle_group_extremes(user_text, scope_df, scope_label)
        if group_ans:
            await update.message.reply_text(group_ans)
            return
        # Fallback to sweet/smoky/rich/delicate extremes on the scope
        sweet_ans = try_handle_extremes_sweet_smoky_rich_delicate(user_text, scope_df)
        if sweet_ans:
            await update.message.reply_text(f"📋 מתוך בקבוקי {scope_label}:\n\n{sweet_ans}")
            return

    # ── general extremes on whole collection ──
    ans = try_handle_extremes_sweet_smoky_rich_delicate(user_text, df)
    if ans:
        await update.message.reply_text(ans)
        return    

    # ----- שאלות ספציפיות על בקבוק בשם מפורש -----
    # לדוגמה: "האם Glenfiddich Project XX מתוק?" / "מה ABV של M&H Biblical?"
    # חייב לרוץ לפני ה-pending flows כי הוא דטרמיניסטי ומהיר,
    # ולפני ה-flavor/cask/stock handlers כי הוא מטפל בכולם.
    if _is_specific_bottle_question(user_text):
        active_df_for_specific = df[df["stock_status_per"] > 0].copy() if df is not None and not df.empty else None
        specific_df = active_df_for_specific if (active_df_for_specific is not None and not active_df_for_specific.empty) else df
        specific_ans = try_handle_specific_bottle_question(user_text, specific_df, context)
        if specific_ans:
            # לטעמים ו-age נשלח עם Markdown, לשאר — plaintext
            if _looks_like_flavors_of_bottle_query(user_text) or _looks_like_age_of_bottle_query(user_text):
                await update.message.reply_text(specific_ans, parse_mode="Markdown")
            else:
                await update.message.reply_text(specific_ans)
            return
    
    
    # Global cancel
    if _normalize_text(user_text) in [_normalize_text(x) for x in _CANCEL_WORDS]:
        context.user_data.pop("pending_update", None)
        context.user_data.pop("pending_count", None)
        context.user_data.pop("pending_stock", None)
        _clear_update_flow(context)
        _clear_add_flow(context)
        await update.message.reply_text("סבבה, ביטלתי את הפעולה הקודמת. שלח שאלה חדשה 🙂")

        return

    # ===========================
    # Add Bottle flow (text stages)
    # ===========================
    stage = _get_add_stage(context)

    if stage == "await_duplicate_confirm":
        t = _normalize_text(user_text)
        if any(_normalize_text(x) == t for x in _CONFIRM_YES) or t in ("כן", "yes"):
            _set_add_stage(context, "await_price")
            await update.message.reply_text("מה המחיר ששילמת? (רק מספר, למשל 350)")
        elif any(_normalize_text(x) == t for x in _CONFIRM_NO) or t in ("לא", "no"):
            _clear_add_flow(context)
            await update.message.reply_text("סבבה, ביטלתי את ההוספה. שלח שאלה חדשה 🙂")
        else:
            await update.message.reply_text("ענה בבקשה כן/לא.")
        return

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
    
    
    # ===========================
    # Guided Update Flow
    # ===========================
    update_flow = _get_update_flow(context)
    if update_flow:
        stage = update_flow.get("stage")
        t_norm = _normalize_text(user_text)

        if t_norm in [_normalize_text(x) for x in _CANCEL_WORDS]:
            _clear_update_flow(context)
            await update.message.reply_text("ביטלתי. שלח שאלה חדשה.")
            return

        if stage == "await_bottle_name":
            df_all = get_all_data_as_df()
            active_upd = df_all[df_all["stock_status_per"] > 0].copy()
            match = find_best_bottle_match(user_text, active_upd)
            if not match["best_name"] or match["score"] < 0.55:
                await update.message.reply_text("לא מצאתי. נסה שוב (מזקקה + שם בקבוק).")
                return

            # ── Disambiguation: multiple bottle_ids with the same name ──
            dups = match.get("duplicates", [])
            if dups:
                update_flow["duplicate_options"] = dups
                update_flow["stage"] = "await_duplicate_select"
                _set_update_flow(context, update_flow)
                lines = [
                    f"{i+1}. 🥃 {d['full_name']}  |  bottle_id: {d['bottle_id']}  |  נשאר: {round(float(d['stock_status_per'] or 0))}%"
                    for i, d in enumerate(dups)
                ]
                await update.message.reply_text(
                    f"מצאתי {len(dups)} בקבוקים עם אותו שם. איזה מהם לעדכן?\n\n"
                    + "\n".join(lines)
                    + "\n\nשלח את המספר המתאים:"
                )
                return

            update_flow["bottle_id"] = match["bottle_id"]
            update_flow["full_name"] = match["best_name"]
            update_flow["stage"] = "await_bottle_confirm"
            _set_update_flow(context, update_flow)
            await update.message.reply_text(f"התכוונת ל: 🥃 {match['best_name']}? (כן / לא)")
            return

        if stage == "await_duplicate_select":
            dups = update_flow.get("duplicate_options", [])
            try:
                choice = int(user_text.strip()) - 1
                if choice < 0 or choice >= len(dups):
                    raise ValueError
            except ValueError:
                await update.message.reply_text(f"שלח מספר בין 1 ל-{len(dups)}.")
                return
            chosen = dups[choice]
            update_flow["bottle_id"] = chosen["bottle_id"]
            update_flow["full_name"] = chosen["full_name"]
            update_flow.pop("duplicate_options", None)
            update_flow["stage"] = "await_drinker"
            update_flow["known_drinkers"] = _get_known_drinkers_from_df()
            update_flow["selected_drinkers"] = []
            _set_update_flow(context, update_flow)
            known = update_flow["known_drinkers"]
            if known:
                kb = _build_drinker_keyboard(known, [])
                await update.message.reply_text(
                    f"✅ נבחר: 🥃 {chosen['full_name']} (ID: {chosen['bottle_id']})\nמי שתה?",
                    reply_markup=kb
                )
            else:
                await update.message.reply_text(
                    f"✅ נבחר: 🥃 {chosen['full_name']} (ID: {chosen['bottle_id']})\nמי שתה? (כתוב שם)"
                )
            return

        if stage == "await_bottle_confirm":
            if t_norm in [_normalize_text(x) for x in _CONFIRM_YES]:
                known = _get_known_drinkers_from_df()
                update_flow["stage"] = "await_drinker"
                update_flow["known_drinkers"] = known
                update_flow["selected_drinkers"] = []
                _set_update_flow(context, update_flow)
                if known:
                    kb = _build_drinker_keyboard(known, [])
                    await update.message.reply_text("מי שתה? (בחר אחד או יותר)", reply_markup=kb)
                else:
                    await update.message.reply_text("מי שתה? (כתוב שם)")
                return
            if t_norm in [_normalize_text(x) for x in _CONFIRM_NO]:
                update_flow["stage"] = "await_bottle_name"
                _set_update_flow(context, update_flow)
                await update.message.reply_text("כתוב את שם הבקבוק שוב:")
                return
            await update.message.reply_text("ענה כן או לא.")
            return

        if stage == "await_drinker":
            # Free-text fallback (no known drinkers, or user adding new name via text)
            if update_flow.get("adding_new_drinker"):
                new_name = user_text.strip()
                if not new_name:
                    await update.message.reply_text("כתוב שם.")
                    return
                selected = update_flow.get("selected_drinkers", [])
                if new_name not in selected:
                    selected.append(new_name)
                update_flow["selected_drinkers"] = selected
                update_flow.pop("adding_new_drinker", None)
                known = update_flow.get("known_drinkers", [])
                if known:
                    kb = _build_drinker_keyboard(known, selected)
                    await update.message.reply_text(f"נוסף: {new_name}\nבחר עוד או לחץ אישור:", reply_markup=kb)
                else:
                    # No known drinkers — go straight to ml
                    update_flow["drinkers"] = selected
                    update_flow["stage"] = "await_ml"
                    _set_update_flow(context, update_flow)
                    await update.message.reply_text("כמה מ\"ל נשתה? (מספר בלבד)")
                return
            # No known drinkers at all — accept free text directly
            drinkers = [d.strip() for d in re.split(r"[,،]", user_text) if d.strip()]
            if not drinkers:
                await update.message.reply_text("כתוב שם שותה.")
                return
            update_flow["drinkers"] = drinkers
            update_flow["stage"] = "await_ml"
            _set_update_flow(context, update_flow)
            await update.message.reply_text("כמה מ\"ל נשתה? (מספר בלבד)")
            return

        if stage == "await_ml":
            m_ml = re.match(r"^\s*(\d{1,4}(?:\.\d+)?)\s*$", user_text.replace(",", "."))
            if not m_ml or float(m_ml.group(1)) <= 0:
                await update.message.reply_text("כתוב מספר בלבד. למשל: 60")
                return
            update_flow["amount_ml"] = int(float(m_ml.group(1)))
            update_flow["stage"] = "await_glasses"
            _set_update_flow(context, update_flow)
            await update.message.reply_text("בכמה כוסות/דראמים? (מספר שלם)")
            return

        if stage == "await_glasses":
            m_g = re.match(r"^\s*(\d{1,3})\s*$", user_text)
            if not m_g or int(m_g.group(1)) <= 0:
                await update.message.reply_text("כתוב מספר שלם גדול מ-0.")
                return
            update_flow["glasses_cnt"] = int(m_g.group(1))
            update_flow["stage"] = "await_final_confirm"
            _set_update_flow(context, update_flow)
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ כן", callback_data="upd_confirm:yes"),
                InlineKeyboardButton("❌ לא", callback_data="upd_confirm:no"),
            ]])
            await update.message.reply_text(
                f"סיכום:\n"
                f"🥃 {update_flow['full_name']}\n"
                f"👤 {', '.join(update_flow.get('drinkers', []))}\n"
                f"📏 {update_flow['amount_ml']} מ\"ל\n"
                f"🥛 {update_flow['glasses_cnt']} כוסות\n\n"
                f"לעדכן?",
                reply_markup=kb
            )
            return

        if stage == "await_final_confirm":
            if t_norm in [_normalize_text(x) for x in _CONFIRM_YES]:
                df_all = get_all_data_as_df()
                active_upd = df_all[df_all["stock_status_per"] > 0].copy()
                inventory_dict = {
                    int(r["bottle_id"]): {
                        "name": r["full_name"],
                        "stock": float(r["stock_status_per"]),
                        "vol": float(r["orignal_volume"]) if pd.notnull(r.get("orignal_volume")) else 700.0,
                        "old_nose": normalize_to_list(r.get("nose")),
                        "old_palette": normalize_to_list(r.get("palette")),
                        "old_abv": float(r["alcohol_percentage"]) if pd.notnull(r.get("alcohol_percentage")) else 0.0,
                    }
                    for _, r in active_upd.iterrows()
                }
                ok, msg = execute_drink_update(
                    int(update_flow["bottle_id"]),
                    int(update_flow["amount_ml"]),
                    inventory_dict,
                    update_flow["glasses_cnt"],
                    drinkers=update_flow.get("drinkers"),
                )
                bottle_name = update_flow.get("full_name", "הבקבוק")
                _set_focus_bottle(context, {"bottle_id": int(update_flow["bottle_id"]), "full_name": bottle_name})
                _clear_update_flow(context)
                if ok:
                    await update.message.reply_text(
                        f"{msg}\n\n"
                        f"🎉 הבקבוק *{bottle_name}* עודכן בהצלחה!",
                        parse_mode="Markdown"
                    )
                else:
                    await update.message.reply_text(
                        f"❌ העדכון של *{bottle_name}* נכשל:\n{msg}",
                        parse_mode="Markdown"
                    )
                return
            if t_norm in [_normalize_text(x) for x in _CONFIRM_NO]:
                update_flow["stage"] = "await_fix_field"
                _set_update_flow(context, update_flow)
                await update.message.reply_text(
                    "מה לתקן?\n1. בקבוק\n2. שותה\n3. כמות (מ\"ל)\n4. כוסות\n5. ביטול"
                )
                return
            await update.message.reply_text("ענה כן או לא.")
            return

        if stage == "await_fix_field":
            m_fix = re.match(r"^\s*([1-5])\s*$", user_text)
            if not m_fix:
                await update.message.reply_text("בחר 1-5.")
                return
            choice = int(m_fix.group(1))
            if choice == 5:
                _clear_update_flow(context)
                await update.message.reply_text("ביטלתי.")
                return
            stage_map = {
                1: ("await_bottle_name", "כתוב שם בקבוק:"),
                2: ("await_drinker", "מי שתה?"),
                3: ("await_ml", "כמה מ\"ל?"),
                4: ("await_glasses", "כמה כוסות?"),
            }
            update_flow["stage"], prompt = stage_map[choice]
            _set_update_flow(context, update_flow)
            await update.message.reply_text(prompt)
            return

        _clear_update_flow(context)

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
                    "old_nose": normalize_to_list(r.get("nose")),
                    "old_palette": normalize_to_list(r.get("palette")),
                    "old_abv": float(r["alcohol_percentage"]) if pd.notnull(r.get("alcohol_percentage")) else 0.0,
                }

            glasses_cnt = int(pending.get("glasses_cnt") or max(1, round(int(pending["amount_ml"]) / 30)))
            ok, msg = execute_drink_update(
                int(pending["bottle_id"]),
                int(pending["amount_ml"]),
                inventory_dict,
                glasses_cnt,
                pending.get("new_palette"),
                pending.get("new_nose"),
                pending.get("new_abv"),
            )
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

        if looks_like_text_intent(user_text):
            # ───────────────────────────────────────────────────────────────
            # KEY FIX: if this looks like a GENERAL portfolio query
            # (e.g. "which bottles have Chocolate flavor?"), we must NOT
            # pass the focus bottle to Gemini — otherwise Gemini will
            # filter by the last specific bottle instead of the whole collection.
            # We only pass focus when the user is asking a follow-up about
            # a specific bottle (detected by pronouns: הוא/בו/שלו/זה...).
            # ───────────────────────────────────────────────────────────────

            # ✅ follow-up על רשימה (מהם/מאלו/מביניהם)? סנן לסאב-DF
            if _is_list_followup(user_text):
                _list_df = _get_focus_list_df(active_df, context)
                if _list_df is not None and not _list_df.empty:
                    _list_label = context.user_data.get("focus_list_label", "הרשימה")
                    plan = gemini_make_df_query_plan(user_text, _list_df, focus=None)
                    if plan:
                        plan = normalize_plan_columns(plan, _list_df)
                        res_df = execute_df_query_plan(_list_df, plan)
                        if res_df is not None and not res_df.empty:
                            await update.message.reply_text(
                                f"📋 מתוך בקבוקי {_list_label}:\n\n" + format_df_answer(res_df, plan)
                            )
                            return
                    # fallback: gemini free-text על הסאב-DF
                    ans = await gemini_fallback_answer(user_text, _list_df)
                    if ans:
                        await update.message.reply_text(f"📋 מתוך בקבוקי {_list_label}:\n\n{ans}")
                        return

            if _is_general_portfolio_query(user_text):
                # General query — NO focus passed to Gemini
                plan = gemini_make_df_query_plan(user_text, active_df, focus=None)
            else:
                # Follow-up about a specific bottle — pass focus
                plan = gemini_make_df_query_plan(user_text, active_df, focus=_get_focus_bottle_row(active_df, context))

            if plan:
                logging.info("DF PLAN: %s", plan)
                plan = normalize_plan_columns(plan, active_df)
                res_df = execute_df_query_plan(active_df, plan)
                if res_df is not None and not res_df.empty:
                    # For general queries: also CLEAR focus so next question starts fresh
                    if _is_general_portfolio_query(user_text):
                        _clear_focus(context)
                    await update.message.reply_text(format_df_answer(res_df, plan))
                    return

        
        # ===========================
        # Bottle-specific: flavors / casks (MUST be before Gemini df_query)
        # ===========================
        # A) extremes only when asking "הכי ..."
        if _is_extremes_question(user_text):
            # ✅ follow-up על רשימה? סנן לפיה
            _extremes_df = df
            _extremes_prefix = ""
            if _is_list_followup(user_text):
                _ldf = _get_focus_list_df(df, context)
                if _ldf is not None and not _ldf.empty:
                    _extremes_df = _ldf
                    _extremes_prefix = f"📋 מתוך בקבוקי {context.user_data.get('focus_list_label', 'הרשימה')}:\n\n"
            else:
                # ✅ מזקקה מוזכרת ישירות בשאלה? ("הכי מתוק של M&H")
                _dist_df = _extract_distillery_scope_from_extremes(user_text, active_df)
                if _dist_df is not None and not _dist_df.empty:
                    _extremes_df = _dist_df
                    _extremes_prefix = f"📋 מתוך בקבוקי {_dist_df.iloc[0].get('distillery', '')}:\n\n"
            ans = try_handle_extremes_sweet_smoky_rich_delicate(user_text, _extremes_df)
            if ans:
                await update.message.reply_text(_extremes_prefix + ans)
                return

        # B) focus bottle flavor questions only for "כמה מתוק הוא / האם הוא עדין / הוא עשיר?"
        if _is_focus_flavor_question(user_text):
            ans = try_handle_focus_bottle_flavor_questions(user_text, df, context)
            if ans:
                await update.message.reply_text(ans)
                return

        # C) VFM questions – focus bottle / general / group
        if _is_vfm_question(user_text):
            ans = try_handle_vfm_questions(user_text, df, context)
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
            reply = await try_gemini_df_query_answer(user_text, active_df, context)
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
            rb = rule_based_inventory_answer(user_text, active_df)
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
            rb = rule_based_inventory_answer(user_text, active_df)
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

                # ✅ שמור focus list לשאלות follow-up
                _set_focus_list(context, sub, label=best_dist)

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
            reply = await try_gemini_df_query_answer(user_text, active_df, context)
            if reply:
                await update.message.reply_text(reply)
                return

        # ─────────────────────────────────────────────────────────────
        # 2b-pre-0) Follow-up על רשימת בקבוקים של מזקקה / קטגוריה
        #  "מה מהם הכי יקר", "מה מאלו הכי מתוק", "איזה מביניהם Sherry"
        # ─────────────────────────────────────────────────────────────
        if _is_list_followup(user_text):
            list_df = _get_focus_list_df(active_df, context)
            if list_df is not None and not list_df.empty:
                label = context.user_data.get("focus_list_label", "הרשימה")
                # הפעל Gemini על תת-הקבוצה בלבד
                try:
                    plan = gemini_make_df_query_plan(user_text, list_df, focus=None)
                    if plan and isinstance(plan, dict):
                        res = execute_df_query_plan(list_df, plan)
                        reply = _df_to_telegram_text(res)
                    else:
                        reply = await gemini_fallback_answer(user_text, list_df)
                    if reply:
                        await update.message.reply_text(
                            f"📋 מתוך בקבוקי {label}:\n\n{reply}"
                        )
                        return
                except Exception as e:
                    logging.warning(f"Focus-list follow-up failed: {e}")
                    # fall through to regular handling

        # 2b-pre) History time-range query: "אילו בקבוקים שתיתי ב-X ימים האחרונים?"
        if _looks_like_history_timerange_query(user_text):
            n_days = _extract_days_from_timerange(user_text)
            try:
                bottles = query_bottles_drunk_in_last_n_days(n_days, df)
                reply = build_history_timerange_reply(n_days, bottles)
            except Exception as e:
                logging.warning(f"History time-range query failed: {e}")
                reply = "❌ לא הצלחתי לשלוף את ההיסטוריה. נסה שוב."
            await update.message.reply_text(reply)
            return
        
        # 2b) Update trigger
        if _looks_like_update_trigger(user_text):
            _set_update_flow(context, {"stage": "await_bottle_name"})
            await update.message.reply_text("איזה בקבוק תרצה לעדכן?")
            return
        
       # --- FINAL: Gemini as the default engine ---
        df = get_all_data_as_df()

        try:
            # Same principle as above: if this is a general portfolio query,
            # call Gemini WITHOUT focus so it doesn't anchor on the last bottle.
            # ✅ follow-up על רשימה? הפעל Gemini על הסאב-DF
            if _is_list_followup(user_text):
                _ldf = _get_focus_list_df(df, context)
                if _ldf is not None and not _ldf.empty:
                    _lbl = context.user_data.get("focus_list_label", "הרשימה")
                    reply = await try_gemini_df_query_answer(user_text, _ldf, context)
                    if reply:
                        await update.message.reply_text(f"📋 מתוך בקבוקי {_lbl}:\n\n{reply}")
                        return

            if _is_general_portfolio_query(user_text):
                # Temporarily clear focus just for this call
                saved_focus_id   = context.user_data.get("focus_bottle_id")
                saved_focus_name = context.user_data.get("focus_full_name")
                saved_focus_dist = context.user_data.get("focus_distillery")
                _clear_focus(context)
                reply = await try_gemini_df_query_answer(user_text, df, context)
                # Restore focus after the call (so bottle-level follow-ups still work)
                # Actually for a general query we want to KEEP focus cleared.
                # If you want to RESTORE instead, uncomment the 3 lines below:
                # context.user_data["focus_bottle_id"]   = saved_focus_id
                # context.user_data["focus_full_name"]   = saved_focus_name
                # context.user_data["focus_distillery"]  = saved_focus_dist
            else:
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
    application.add_handler(CallbackQueryHandler(handle_final_confirm_callback, pattern=r"^upd_confirm:"))
    application.add_handler(CallbackQueryHandler(handle_drinker_callback, pattern=r"^drk_"))
    application.add_handler(MessageHandler(filters.PHOTO & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))


    print("Whisky Telegram agent running (deterministic fuzzy inventory + Gemini DF analytics fallback)...")
    application.run_polling()
