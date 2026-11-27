"""AuroraStays QnA: Streamlit front end for the hotel-policy question service."""
import html
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:7860").rstrip("/")
TIMEOUT = (5, 120)  # (connect, read) seconds
MODELS = ["gpt-4o-mini", "gpt-4o"]
HISTORY_ROWS = 60  # rows drawn in the sidebar; search looks through all loaded rows
EXAMPLES = [
    "Is breakfast free at the Miami Brickell property?",
    "What is the pet fee at ASH-CHI Loop?",
    "Can I cancel an Advance Purchase booking for a refund?",
    "A guest has a service dog. What can the front desk ask?",
    "What can we charge if a guest smokes in the room?",
    "Why does my card still show a hold after checkout?",
]


def logo_svg(size: int, tile: str, ink: str) -> str:
    """The mark: a doorway that reads as an A, under a small aurora arc."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 32 32' "
            f"fill='none' role='img' aria-label='AuroraStays'><rect width='32' height='32' rx='9' fill='{tile}'/>"
            f"<path d='M10 24v-8a6 6 0 0 1 12 0v8M13.2 20h5.6' stroke='{ink}' stroke-width='2.6' "
            f"stroke-linecap='round'/><path d='M8 10a10.5 10.5 0 0 1 16 0' stroke='{ink}' stroke-opacity='.55' "
            f"stroke-width='2' stroke-linecap='round'/></svg>")


st.set_page_config(page_title="AuroraStays QnA", page_icon=logo_svg(64, "#287A5B", "#FFFFFF"),
                   layout="centered", initial_sidebar_state="auto")

# ---------------------------------------------------------------- state
ss = st.session_state
if "theme" not in ss:
    try:
        ss.theme = "dark" if (st.context.theme.type == "dark") else "light"
    except Exception:
        ss.theme = "light"
ss.setdefault("messages", [])
ss.setdefault("pending", None)
ss.setdefault("active_id", None)
ss.setdefault("inflight", None)  # question being answered; survives an interrupted run

try:
    TZ = ZoneInfo(st.context.timezone)  # the browser's time zone
except Exception:
    TZ = datetime.now().astimezone().tzinfo

# ---------------------------------------------------------------- theme
PALETTE = {
    "light": dict(bg="#F7F8F7", surface="#FFFFFF", surface2="#F0F2F0", hover="#E9EDEA", border="#E3E8E4",
                  text="#202722", muted="#5F6962", faint="#626C65", accent="#287A5B", accent_hover="#20674C",
                  accent_soft="#E7F2EC", on_accent="#FFFFFF",
                  ok_fg="#1F6B4E", ok_bg="#E7F2EC", warn_fg="#875810", warn_bg="#F8F0DF",
                  bad_fg="#AD3D3D", bad_bg="#F9EAEA",
                  shadow="0 1px 2px rgba(20,30,24,.04), 0 4px 16px rgba(20,30,24,.04)"),
    "dark": dict(bg="#111512", surface="#191F1B", surface2="#202823", hover="#29332C", border="#29332D",
                 text="#EEF2EE", muted="#A3ADA5", faint="#96A199", accent="#72BD96", accent_hover="#8ACBA8",
                 accent_soft="#1D382A", on_accent="#0E1711",
                 ok_fg="#72BD96", ok_bg="#1D382A", warn_fg="#E6B968", warn_bg="#40331D",
                 bad_fg="#F18A85", bad_bg="#402523",
                 shadow="0 1px 3px rgba(0,0,0,.3)"),
}
FONT = "'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
MONO = "'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def inject_css(mode: str) -> None:
    p = PALETTE[mode]
    vars_ = ";".join(f"--a-{k.replace('_', '-')}:{v}" for k, v in p.items())
    st.html(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{{{vars_};--a-font:{FONT};--a-mono:{MONO};--a-ease:cubic-bezier(.2,.8,.2,1);color-scheme:{mode}}}
/* base */
/* the font itself is set in .streamlit/config.toml, so Streamlit's icon fonts are left alone */
.stApp{{-webkit-font-smoothing:antialiased;font-feature-settings:"cv11","ss01"}}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stBottom"]>div,
[data-testid="stBottomBlockContainer"]{{background:var(--a-bg)!important;color:var(--a-text)}}
[data-testid="stHeader"]{{background:var(--a-bg)!important}}
.stApp p,.stApp li,.stApp label,.stApp span,.stApp h1,.stApp h2,.stApp h3,.stApp summary,
[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"] *{{color:var(--a-text)}}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{{color:var(--a-muted)!important}}
.stApp code{{background:var(--a-surface2);color:var(--a-text);font-family:var(--a-mono)}}
.stApp *{{overflow-wrap:anywhere}}
.block-container{{padding-top:4rem;padding-bottom:1rem;max-width:792px}}
.a-mono{{font-family:var(--a-mono);font-variant-numeric:tabular-nums}}
/* sidebar */
[data-testid="stSidebar"]{{border-right:1px solid var(--a-border);background:var(--a-bg)}}
[data-testid="stSidebar"]>div:first-child,[data-testid="stSidebarContent"]{{background:var(--a-bg)}}
[data-testid="stSidebarUserContent"]{{padding-left:14px;padding-right:14px}}
.a-brand{{display:flex;align-items:center;gap:10px;font-size:16px;font-weight:600;letter-spacing:-.02em;
 color:var(--a-text);padding:2px 4px 6px}}
.a-brand i{{font-style:normal;font-weight:500;color:var(--a-faint)!important}}
.a-group{{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
 color:var(--a-faint)!important;margin:0;padding:16px 6px 8px;line-height:1.2}}
.a-none{{font-size:13px!important;line-height:1.5;color:var(--a-muted)!important;margin:10px 6px}}
/* inputs: Streamlit themes these from its own base, so follow the app's day/night switch */
[data-testid="stChatInput"]>div,[data-testid="stChatInput"] textarea,
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="input"] input,[data-testid="stTextInputRootElement"],
[data-testid="stTextInput"] input,[data-baseweb="popover"] ul,[data-baseweb="popover"] li{{
 background:var(--a-surface)!important;color:var(--a-text)!important;border-color:var(--a-border)!important}}
[data-testid="stChatInput"]>div{{border-radius:16px!important;box-shadow:var(--a-shadow);
 transition:border-color .18s var(--a-ease),box-shadow .18s var(--a-ease)}}
[data-testid="stChatInput"]>div:focus-within{{border-color:var(--a-accent)!important}}
[data-testid="stChatInput"] textarea::placeholder,[data-testid="stTextInput"] input::placeholder{{
 color:var(--a-faint)!important;opacity:1}}
[data-testid="stChatInputSubmitButton"]{{background:var(--a-accent)!important;color:var(--a-on-accent)!important;
 border-radius:10px!important}}
[data-testid="stChatInputSubmitButton"] *{{color:var(--a-on-accent)!important}}
[data-testid="stChatInputSubmitButton"]:disabled{{background:var(--a-surface2)!important}}
[data-testid="stChatInputSubmitButton"]:disabled *{{color:var(--a-faint)!important}}
[data-testid="stTextInputRootElement"]{{border-radius:10px!important}}
[data-testid="stExpander"] details{{background:transparent;border:0!important}}
[data-testid="stExpander"] summary{{background:transparent!important;color:var(--a-muted)!important;
 font-size:13px;padding:6px 8px;border-radius:8px;transition:background .18s var(--a-ease)}}
[data-testid="stExpander"] summary *{{color:var(--a-muted)!important}}
[data-testid="stExpander"] summary p{{font-size:13px}}
[data-testid="stExpander"] summary:hover{{background:var(--a-hover)!important}}
[data-testid="stExpanderDetails"]{{padding:4px 8px 8px}}
[data-testid="stSelectbox"] [role="group"]{{background:var(--a-surface)!important;border-color:var(--a-border)!important}}
[data-testid="stSelectbox"] input,[data-testid="stSelectbox"] button{{color:var(--a-text)!important}}
[role="listbox"],[role="listbox"] [role="option"]{{background:var(--a-surface)!important;color:var(--a-text)!important}}
[role="listbox"] [role="option"]:hover,[role="listbox"] [role="option"][data-focused]{{background:var(--a-hover)!important}}
[data-testid="stTooltipContent"]{{background:var(--a-surface)!important;color:var(--a-text)!important;
 border:1px solid var(--a-border)}}
[data-testid="stTooltipContent"] *{{color:var(--a-text)!important}}
[data-testid="stToast"]{{background:var(--a-surface)!important;border:1px solid var(--a-border);box-shadow:var(--a-shadow)}}
[data-testid="stToast"] *{{color:var(--a-text)!important}}
/* buttons */
.stButton button,.stDownloadButton button{{background:transparent;color:var(--a-text);border:1px solid var(--a-border);
 border-radius:10px;font-size:13px;font-weight:500;min-height:36px;
 transition:background .18s var(--a-ease),border-color .18s var(--a-ease),transform .15s var(--a-ease)}}
.stButton button p,.stDownloadButton button p{{font-size:13px;font-weight:500}}
.stButton button:hover,.stDownloadButton button:hover{{background:var(--a-hover);border-color:var(--a-border);color:var(--a-text)}}
.stButton button:active{{transform:scale(.985)}}
.stButton button:focus-visible{{outline:2px solid var(--a-accent);outline-offset:2px}}
.stButton button[kind="primary"]{{background:var(--a-accent);border-color:var(--a-accent)}}
.stButton button[kind="primary"]:hover{{background:var(--a-accent-hover);border-color:var(--a-accent-hover)}}
.stButton button[kind="primary"] *{{color:var(--a-on-accent)!important}}
.stButton button:disabled,.stDownloadButton button:disabled{{opacity:.5}}
/* icon-only buttons keep a real label for screen readers; it is hidden from view */
.st-key-theme_btn button [data-testid="stMarkdownContainer"],[class*="st-key-del_"] button [data-testid="stMarkdownContainer"]{{
 position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}}
/* top bar */
.st-key-topbar{{position:fixed;top:.6rem;right:1rem;z-index:999991;width:auto!important;
 background:var(--a-bg);border-radius:12px;padding:2px}}
.st-key-topbar .stButton button{{min-height:34px;padding:.2rem .7rem;border-color:transparent}}
/* history rows: dot, one-line question, time underneath */
.st-key-history{{gap:2px!important;margin-bottom:10px}}
[class*="st-key-hrow-"]{{border-radius:10px;transition:background .18s var(--a-ease);gap:0!important;
 flex-wrap:nowrap!important}}
[class*="st-key-hrow-"]:hover,.a-active-row{{background:var(--a-hover)}}
[class*="st-key-hrow-"]>div:first-child{{flex:1 1 auto;min-width:0}}
[class*="st-key-hrow-"]>div:last-child{{flex:0 0 auto;width:auto!important}}
[class*="st-key-open_"] button{{display:grid;grid-template-columns:16px minmax(0,1fr);align-items:center;
 text-align:left;border:0!important;background:transparent!important;padding:7px 4px 7px 8px;min-height:0;
 width:100%;row-gap:1px;transform:none!important}}
[class*="st-key-open_"] button::before{{content:"";width:7px;height:7px;border-radius:50%;background:var(--a-faint);
 grid-column:1;grid-row:1}}
[class*="st-key-hrow-auto-"] [class*="st-key-open_"] button::before{{background:var(--a-ok-fg)}}
[class*="st-key-hrow-review-"] [class*="st-key-open_"] button::before{{background:var(--a-warn-fg)}}
[class*="st-key-hrow-manual-"] [class*="st-key-open_"] button::before{{background:var(--a-bad-fg)}}
[class*="st-key-open_"] button>*{{grid-column:2;grid-row:1;min-width:0;width:100%;display:block}}
[class*="st-key-open_"] button::after{{grid-column:2;grid-row:2;font-family:var(--a-mono);font-size:10.5px;
 color:var(--a-faint);line-height:1.3;white-space:nowrap}}
[class*="st-key-open_"] button *{{min-width:0;max-width:100%}}
[class*="st-key-open_"] button p{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:left;
 font-size:13px;font-weight:450;line-height:1.35;display:block}}
[class*="st-key-hrow-"].a-on [class*="st-key-open_"] button p{{font-weight:600}}
[class*="st-key-del_"] button{{border:0!important;background:transparent!important;min-height:28px;width:28px;
 padding:0;opacity:0;transition:opacity .15s var(--a-ease)}}
[class*="st-key-del_"] button *{{color:var(--a-muted)!important}}
[class*="st-key-hrow-"]:hover [class*="st-key-del_"] button,[class*="st-key-del_"] button:focus-visible{{opacity:1}}
[class*="st-key-del_"] button:hover{{background:var(--a-surface2)!important}}
@media (hover:none){{[class*="st-key-del_"] button{{opacity:.7}}}}
/* empty state */
.a-hero{{text-align:center;padding:2.5rem .5rem .5rem}}
.a-hero h1{{font-size:clamp(24px,5vw,32px);font-weight:600;letter-spacing:-.035em;line-height:1.25;
 margin:18px 0 6px;padding:0;color:var(--a-text)}}
.a-hero p{{color:var(--a-muted)!important;font-size:15px!important;line-height:1.6;max-width:30rem;margin:0 auto}}
.a-try{{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--a-faint)!important;
 text-align:center;margin:28px 0 8px}}
.st-key-examples .stButton button{{background:var(--a-surface);border-radius:12px;padding:10px 14px;
 box-shadow:var(--a-shadow);border-color:transparent;text-align:left}}
.st-key-examples .stButton button:hover{{background:var(--a-surface);border-color:var(--a-accent)}}
.st-key-examples .stButton button p{{font-weight:450;color:var(--a-text)}}
.st-key-examples .stButton button *{{text-align:left;justify-content:flex-start}}
.st-key-examples .stButton button>*{{width:100%}}
.st-key-examples>div{{flex:1 1 calc(50% - .5rem);min-width:240px}}
.st-key-examples .stButton,.st-key-examples .stButton button{{width:100%;justify-content:flex-start}}
[data-testid="stSidebarHeader"]{{height:3rem;padding-bottom:0}}
.a-foot{{text-align:center;color:var(--a-faint)!important;font-size:12px!important;margin-top:28px}}
/* question bubble */
.a-q{{display:flex;flex-direction:column;align-items:flex-end;margin:14px 0 2px}}
.a-q-b{{background:var(--a-surface2);color:var(--a-text);border-radius:16px 16px 4px 16px;padding:9px 14px;
 max-width:85%;font-size:15px;line-height:1.55;white-space:pre-wrap}}
.a-q-t{{font-family:var(--a-mono);font-size:10.5px;color:var(--a-faint);margin:4px 6px 0}}
/* answer card */
[class*="st-key-card_"]{{background:var(--a-surface);border-radius:16px;padding:20px 22px 12px;
 box-shadow:var(--a-shadow);margin:6px 0 10px;animation:a-in .2s var(--a-ease);gap:.3rem!important}}
@keyframes a-in{{from{{opacity:0;transform:translateY(4px)}}to{{opacity:1;transform:none}}}}
[class*="st-key-card_"] [data-testid="stMarkdownContainer"] p{{font-size:15px;line-height:1.7}}
.a-meta{{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin:0 0 10px;font-size:12px}}
.a-pill{{display:inline-flex;align-items:center;gap:7px;padding:5px 11px;border-radius:999px;font-weight:700;
 letter-spacing:.04em;font-size:10.5px;white-space:nowrap}}
.a-pill::before{{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}}
.a-ok{{color:var(--a-ok-fg)!important;background:var(--a-ok-bg)}}
.a-warn{{color:var(--a-warn-fg)!important;background:var(--a-warn-bg)}}
.a-bad{{color:var(--a-bad-fg)!important;background:var(--a-bad-bg)}}
.a-hint{{color:var(--a-muted)!important;font-size:12.5px}}
.a-stat{{margin-left:auto;font-family:var(--a-mono);font-size:11px;color:var(--a-faint)!important;white-space:nowrap}}
sup.a-cite{{display:inline-block;min-width:16px;padding:0 4px;margin:0 1px;border-radius:5px;
 background:var(--a-accent-soft);color:var(--a-accent)!important;font-family:var(--a-mono);font-size:10.5px;
 font-weight:500;line-height:16px;text-align:center;vertical-align:.3em}}
.a-load{{font-size:14px;font-weight:500;background:linear-gradient(90deg,var(--a-faint) 25%,var(--a-text) 50%,
 var(--a-faint) 75%);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;
 color:transparent!important;animation:a-shim 1.6s linear infinite;display:inline-block}}
@keyframes a-shim{{from{{background-position:200% 0}}to{{background-position:-200% 0}}}}
.a-caret{{display:inline-block;width:2px;height:1em;background:var(--a-accent);vertical-align:-.15em;margin-left:2px;
 animation:a-blink 1s steps(2) infinite}}
@keyframes a-blink{{50%{{opacity:0}}}}
/* sources */
.a-src{{background:var(--a-surface2);border-radius:12px;padding:12px 14px;margin:0 0 8px}}
.a-src-h{{display:flex;gap:10px;align-items:flex-start}}
.a-src-n{{flex:0 0 auto;min-width:20px;height:20px;border-radius:6px;background:var(--a-accent-soft);
 color:var(--a-accent)!important;font-family:var(--a-mono);font-weight:500;font-size:11px;line-height:20px;text-align:center}}
.a-src-doc{{flex:1 1 auto;min-width:0;font-weight:600;font-size:13px;line-height:1.4;color:var(--a-text)}}
.a-src-sec{{display:block;font-weight:400;color:var(--a-muted)!important;font-size:12px}}
.a-tag{{flex:0 0 auto;font-size:10.5px;font-weight:600;padding:2px 7px;border-radius:999px;white-space:nowrap}}
.a-tag.on{{color:var(--a-ok-fg)!important;background:var(--a-ok-bg)}}
.a-tag.off{{color:var(--a-faint)!important;background:transparent}}
.a-src-t{{color:var(--a-muted)!important;font-size:13px!important;line-height:1.6;margin:8px 0 0;white-space:pre-wrap}}
.a-rel{{display:flex;align-items:center;gap:8px;margin-top:8px;font-family:var(--a-mono);font-size:10.5px;
 color:var(--a-faint)!important}}
.a-rel>div{{flex:1;height:3px;border-radius:99px;background:var(--a-hover);overflow:hidden}}
.a-rel>div>i{{display:block;height:100%;background:var(--a-accent);border-radius:99px}}
.a-why{{font-size:13px;color:var(--a-muted)!important;line-height:1.6;margin:0;padding-left:18px}}
.a-why li{{color:var(--a-muted)!important;font-size:13px}}
.a-notice{{background:var(--a-bad-bg);border-radius:12px;padding:10px 14px;font-size:14px;color:var(--a-text);margin:4px 0 8px}}
.a-notice b{{color:var(--a-bad-fg)!important}}
.a-notice.warn{{background:var(--a-warn-bg)}} .a-notice.warn b{{color:var(--a-warn-fg)!important}}
.a-dot{{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:7px;vertical-align:middle}}
.a-health{{font-size:12px;color:var(--a-muted)!important;margin:8px 6px 0}}
@media (max-width:640px){{.block-container{{padding-left:1rem;padding-right:1rem;padding-top:3.6rem}}
 .a-hero{{padding-top:1rem}} [class*="st-key-card_"]{{padding:16px 16px 10px}} .a-q-b{{max-width:92%}}
 .st-key-topbar{{right:.6rem}} .a-stat{{margin-left:0}}
 .st-key-examples>div:nth-child(n+5),.a-foot{{display:none}} .a-hero h1{{margin-top:12px}} .a-try{{margin-top:18px}}}}
@media (prefers-reduced-motion:reduce){{.stApp *{{animation:none!important;transition:none!important}}}}
</style>""")


inject_css(ss.theme)
P = PALETTE[ss.theme]
MARK = logo_svg(26, P["accent"], P["on_accent"])


# ---------------------------------------------------------------- time
def local(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, timezone.utc).astimezone(TZ)


def clock(d: datetime) -> str:
    return f"{d.hour % 12 or 12}:{d.minute:02d} {'AM' if d.hour < 12 else 'PM'}"


def when(ts: float) -> tuple:
    """(group, label) for a saved question, by local calendar day."""
    d, today = local(ts), datetime.now(TZ).date()
    age = (today - d.date()).days
    if age <= 0:
        return "Today", clock(d)
    if age == 1:
        return "Yesterday", clock(d)
    if age < 7:
        return "Last 7 days", f"{d:%a} {clock(d)}"
    return "Older", f"{d.day} {d:%b}" + ("" if d.year == today.year else f" {d.year}")


# ---------------------------------------------------------------- backend helpers
@st.cache_data(ttl=15, show_spinner=False)
def get_health(url: str) -> dict:
    try:
        r = requests.get(f"{url}/health", timeout=(3, 5))
        if r.status_code == 200:
            return {"ok": True, **r.json()}
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "Not reachable"}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Timed out"}
    except Exception:
        return {"ok": False, "error": "Unexpected response"}


@st.cache_data(ttl=10, show_spinner=False)
def load_history(url: str):
    """Saved questions, newest first. None when they cannot be loaded."""
    try:
        r = requests.get(f"{url}/v1/history", params={"limit": 300}, timeout=(2, 5))
        items = r.json().get("items") if r.status_code == 200 else None
        return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else None
    except Exception:
        return None


class BackendError(Exception):
    pass


class ServerSaid(BackendError):
    """The backend answered with an error of its own, so asking again will not help."""

    def __init__(self, message, sources=None):
        super().__init__(message)
        self.sources = sources or []


def error_from_response(r: requests.Response) -> str:
    try:
        msg = r.json().get("error", {}).get("message")
    except Exception:
        msg = None
    hint = {400: "The question could not be processed", 422: "The request was not valid",
            503: "The answering service is temporarily unavailable",
            500: "The server hit an error"}.get(r.status_code, "The server returned an error")
    return f"{hint} (HTTP {r.status_code}). {msg or ''}".strip()


def friendly(exc: Exception) -> str:
    if isinstance(exc, BackendError):
        return str(exc)
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "Could not reach the answering service within 5 seconds. Check that the backend is running."
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return "The answer took longer than 2 minutes. Try again, or try a shorter question."
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"Cannot connect to the answering service at {BACKEND_URL}. Is the backend running?"
    return "Something went wrong while getting the answer. Please try again."


def post_json(path: str, payload: dict) -> dict:
    r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        raise BackendError(error_from_response(r))
    try:
        return r.json()
    except ValueError:
        raise BackendError("The server sent a response that could not be read.")


def sse_events(resp: requests.Response):
    event, data = "message", []
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.rstrip("\r")
        if not line:
            if data:
                try:
                    yield event, json.loads("\n".join(data))
                except ValueError:
                    pass
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


def loading(text: str) -> str:
    return f"<span class='a-load'>{text}</span>"


def run_query(query: str, params: dict, answer_slot) -> dict:
    """Stream an answer into answer_slot; fall back to the non-streaming endpoint."""
    payload = {"query": query, "save_history": True, **params}
    msg = {"role": "assistant", "content": "", "sources": [], "confidence": None, "latency": None,
           "request_id": None, "error": None, "streamed": False, "query": query, "ts": time.time()}
    t0 = time.perf_counter()
    got_tokens = False
    try:
        with requests.post(f"{BACKEND_URL}/v1/answer/stream", json=payload, timeout=TIMEOUT,
                           stream=True, headers={"Accept": "text/event-stream"}) as r:
            if r.status_code != 200:
                try:
                    found = r.json().get("sources")
                except Exception:
                    found = None
                raise ServerSaid(error_from_response(r), found if isinstance(found, list) else None)
            if "event-stream" not in r.headers.get("content-type", ""):
                raise BackendError("The server sent a response that could not be read.")
            last = 0.0
            for ev, d in sse_events(r):
                if ev == "sources":
                    msg["sources"] = d.get("sources") or []
                    msg["request_id"] = d.get("request_id")
                    answer_slot.markdown(loading("Writing the answer"), unsafe_allow_html=True)
                elif ev == "token":
                    got_tokens = True
                    msg["content"] += d.get("t", "")
                    now = time.perf_counter()
                    if now - last > 0.04:
                        answer_slot.markdown(render_answer_md(msg["content"]) + "<span class='a-caret'></span>",
                                             unsafe_allow_html=True)
                        last = now
                elif ev == "done":
                    # The final event has the cleaned answer and the sources with their "cited" flags.
                    msg["content"] = d.get("answer") or msg["content"]
                    msg["sources"] = d.get("sources") or msg["sources"]
                    msg["confidence"] = d.get("confidence")
                    msg["latency"] = d.get("latency_ms")
                    msg["cached"] = d.get("cached", False)
                    msg["request_id"] = d.get("request_id") or msg["request_id"]
                    msg["streamed"] = True
                elif ev == "error":
                    raise ServerSaid((d.get("error") or {}).get("message") or "Streaming failed.")
        if not msg["streamed"]:
            raise BackendError("The stream ended before the answer was complete.")
    except Exception as exc:  # noqa: BLE001 - every failure becomes a friendly message
        if got_tokens:
            msg["error"] = friendly(exc) + " The partial answer above may be incomplete."
            msg["confidence"] = msg["confidence"] or {"score": 0, "level": "low", "route": "MANUAL LOOKUP",
                                                      "reasons": ["Answer was interrupted"]}
        elif isinstance(exc, (ServerSaid, requests.exceptions.ReadTimeout)):
            msg["error"] = friendly(exc)
            msg["sources"] = msg["sources"] or getattr(exc, "sources", [])
        else:
            # The stream itself could not be set up: try the plain endpoint once.
            answer_slot.markdown(loading("Trying again"), unsafe_allow_html=True)
            try:
                d = post_json("/v1/answer_with_relevant_chunks", payload)
                msg.update(content=d.get("answer", ""), sources=d.get("sources") or [],
                           confidence=d.get("confidence"), latency=d.get("latency_ms"),
                           request_id=d.get("request_id"), cached=d.get("cached", False))
            except Exception as exc2:  # noqa: BLE001
                msg["error"] = friendly(exc2)
    msg["client_ms"] = int((time.perf_counter() - t0) * 1000)
    return msg


# ---------------------------------------------------------------- rendering
CITE_RE = re.compile(r"\[(\d{1,2})\]")
ROUTES = {"AUTO ANSWER": ("a-ok", "auto", "Ready to share with the guest."),
          "ASSISTED REVIEW": ("a-warn", "review", "Check the sources before sharing."),
          "MANUAL LOOKUP": ("a-bad", "manual", "Look this up with the property team.")}


def render_answer_md(text: str) -> str:
    # "$" is escaped so two prices in one answer are not read as a formula,
    # and "![" so an answer can never make the browser load an outside image.
    safe = text.replace("<", "&lt;").replace(">", "&gt;").replace("![", "!\\[")
    safe = re.sub(r"\\?\$", r"\\$", safe)
    return CITE_RE.sub(r"<sup class='a-cite' title='Source \1'>\1</sup>", safe)


def esc(text) -> str:
    """Escape text for an HTML block. Line breaks become entities, because a
    blank line would end the block and the rest would be read as markdown."""
    return html.escape(str(text)).replace("\r", "").replace("\n", "&#10;")


def plain_label(text) -> str:
    """Show a saved question as plain text in a button (labels are markdown)."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|~$<>:])", r"\\\1", " ".join(str(text).split()))


def asked_at(ts) -> str:
    group, label = when(float(ts))
    if group == "Yesterday":
        return f"Yesterday {label}"
    if group == "Older":
        return f"{label}, {clock(local(float(ts)))}"
    return label


def render_question(text: str, ts) -> None:
    stamp = f"<div class='a-q-t'>{asked_at(ts)}</div>" if ts else ""
    st.markdown(f"<div class='a-q'><div class='a-q-b'>{esc(text)}</div>{stamp}</div>", unsafe_allow_html=True)


def render_meta(m: dict) -> None:
    conf = m.get("confidence") or {}
    route = (conf.get("route") or "").upper()
    if not route:
        return
    cls, _, hint = ROUTES.get(route, ("a-warn", "review", "Reliability unknown."))
    stats = []
    lat = m.get("latency") or {}
    if lat.get("total") is not None:
        stats.append(f"{lat['total'] / 1000:.1f} s")
    if m.get("cached"):
        stats.append("cached")
    stat = f"<span class='a-stat'>{' · '.join(stats)}</span>" if stats else ""
    st.markdown(f"<div class='a-meta'><span class='a-pill {cls}'>{html.escape(route)}</span>"
                f"<span class='a-hint'>{hint}</span>{stat}</div>", unsafe_allow_html=True)


def render_sources(sources: list) -> None:
    if not sources:
        return
    cited = sum(1 for s in sources if s.get("cited"))
    label = f"Sources · {len(sources)}" + (f" · {cited} cited" if cited else "")
    with st.expander(label, icon=":material/menu_book:"):
        for s in sources:
            score = max(0.0, min(1.0, float(s.get("score") or 0)))
            tag = ("<span class='a-tag on'>Cited</span>" if s.get("cited")
                   else "<span class='a-tag off'>Not cited</span>")
            code = html.escape(str(s.get("policy_code") or ""))
            st.markdown(f"""<div class='a-src'><div class='a-src-h'>
<span class='a-src-n'>{int(s.get('id') or 0)}</span>
<span class='a-src-doc'>{html.escape(str(s.get('document') or 'Unknown document'))}
<span class='a-src-sec'>{code + ' · ' if code else ''}{html.escape(str(s.get('section') or ''))}</span></span>
{tag}</div>
<p class='a-src-t'>{esc(s.get('text') or 'No text available.')}</p>
<div class='a-rel'><span>Relevance</span><div><i style='width:{score * 100:.0f}%'></i></div>
<span>{score * 100:.0f}%</span></div></div>""", unsafe_allow_html=True)


def render_why(m: dict) -> None:
    conf = m.get("confidence") or {}
    route = (conf.get("route") or "").upper()
    if not route:
        return
    score = conf.get("score")
    title = f"Why {route.capitalize()}"
    if isinstance(score, (int, float)):
        title += f" · score {float(score):.2f}"
    with st.expander(title, icon=":material/verified_user:"):
        items = "".join(f"<li>{html.escape(str(r))}</li>" for r in (conf.get("reasons") or
                                                                    ["No reasons were given by the server."]))
        lat = m.get("latency") or {}
        if lat.get("total") is not None:
            items += (f"<li>Search {esc(lat.get('retrieval', '?'))} ms, "
                      f"writing {esc(lat.get('generation', '?'))} ms</li>")
        st.markdown(f"<ul class='a-why'>{items}</ul>", unsafe_allow_html=True)


def send_feedback(idx: int) -> None:
    m = ss.messages[idx]
    val = ss.get(f"fb_{idx}")
    if val is None or not m.get("request_id"):
        return
    rating = "up" if val == 1 else "down"
    try:
        r = requests.post(f"{BACKEND_URL}/v1/feedback", timeout=(5, 15), json={
            "request_id": m["request_id"], "query": m.get("query", ""),
            "answer": m.get("content", ""), "rating": rating})
        ok = r.status_code == 200
    except Exception:
        ok = False
    if ok:
        m["feedback"] = rating
        load_history.clear()
    else:
        ss[f"fb_{idx}"] = None  # do not leave a thumb selected that was not saved
    st.toast("Thanks, feedback saved." if ok else "Feedback could not be sent right now.")


def render_assistant(m: dict, idx: int, show_sources: bool) -> None:
    with st.container(key=f"card_{idx}"):
        if m.get("content"):
            render_meta(m)
            st.markdown(render_answer_md(m["content"]), unsafe_allow_html=True)
        if m.get("error"):
            st.markdown(f"<div class='a-notice'><b>Couldn't complete the answer.</b> {html.escape(m['error'])} "
                        "Your question is kept above, so you can ask it again.</div>", unsafe_allow_html=True)
        if show_sources and (m.get("content") or m.get("error")):
            render_sources(m.get("sources") or [])
        if m.get("content"):
            render_why(m)
            if m.get("request_id"):
                st.feedback("thumbs", key=f"fb_{idx}", on_change=send_feedback, args=(idx,))


def conversation_markdown() -> str:
    out = [f"# AuroraStays QnA conversation\n\n_Exported {datetime.now(TZ):%Y-%m-%d %H:%M}_\n"]
    for m in ss.messages:
        if m["role"] == "user":
            asked = f" ({local(m['ts']):%Y-%m-%d %H:%M})" if m.get("ts") else ""
            out.append(f"\n## Q: {m['content']}{asked}\n")
        else:
            out.append(m.get("content") or f"_Error: {m.get('error')}_")
            c = m.get("confidence") or {}
            if c:
                out.append(f"\n**Route:** {c.get('route')} (score {float(c.get('score') or 0):.2f})")
            for s in m.get("sources") or []:
                out.append(f"- [{s.get('id')}] {s.get('document')}: {s.get('policy_code')}, "
                           f"{s.get('section')} (relevance {float(s.get('score') or 0):.2f})")
            out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- actions
def toggle_theme():
    ss.theme = "light" if ss.theme == "dark" else "dark"


def clear_feedback_keys():
    for k in [k for k in ss.keys() if str(k).startswith("fb_")]:
        del ss[k]


def new_conversation():
    ss.messages, ss.active_id = [], None
    clear_feedback_keys()


def ask(q: str):
    ss.pending = q


def open_saved(item_id: str):
    """Show a saved question with its answer and sources again."""
    try:
        r = requests.get(f"{BACKEND_URL}/v1/history/{item_id}", timeout=(3, 8))
        if r.status_code != 200:
            raise BackendError(error_from_response(r))
        d = r.json()
        if not isinstance(d, dict):
            raise BackendError("unexpected reply")
    except Exception:
        st.toast("That question could not be opened.")
        return
    clear_feedback_keys()
    ss.messages = [
        {"role": "user", "content": d.get("query", ""), "ts": d.get("ts")},
        {"role": "assistant", "content": d.get("answer", ""), "sources": d.get("sources") or [],
         "confidence": d.get("confidence"), "latency": d.get("latency_ms"), "request_id": d.get("id"),
         "error": None, "query": d.get("query", ""), "ts": d.get("ts"), "feedback": d.get("feedback")},
    ]
    if d.get("feedback") in ("up", "down"):
        ss["fb_1"] = 1 if d["feedback"] == "up" else 0
    ss.active_id = item_id


def delete_saved(item_id: str):
    try:
        ok = requests.delete(f"{BACKEND_URL}/v1/history/{item_id}", timeout=(3, 8)).status_code == 200
    except Exception:
        ok = False
    load_history.clear()
    if ok and ss.active_id == item_id:
        new_conversation()
    st.toast("Removed from history." if ok else "That question could not be removed.")


def render_history(items, needle: str) -> None:
    if items is None:
        st.markdown("<p class='a-none'>History is not available right now.</p>",
                    unsafe_allow_html=True)
        return
    needle = needle.strip().lower()
    if needle:
        items = [i for i in items if needle in str(i.get("query", "")).lower()]
    if not items:
        st.markdown(f"<p class='a-none'>{'No saved question matches.' if needle else 'Questions you ask are saved here.'}</p>",
                    unsafe_allow_html=True)
        return
    css, group = [], None
    for it in items[:HISTORY_ROWS]:
        item_id = str(it.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", item_id):
            continue
        g, label = when(float(it.get("ts") or 0))
        if g != group:
            st.markdown(f"<div class='a-group'>{g}</div>", unsafe_allow_html=True)
            group = g
        route = str(it.get("route") or "").upper()
        kind = ROUTES.get(route, ("", "none", ""))[1]
        if route in ROUTES:
            label += " · " + route.split()[0].capitalize()  # the dot color is not the only cue
        # The time is drawn under the question with a small style rule per row.
        css.append(f'.st-key-open_{item_id} button::after{{content:"{label}"}}')
        if item_id == ss.active_id:
            css.append(f".st-key-hrow-{kind}-{item_id}{{background:var(--a-hover)}}"
                       f".st-key-open_{item_id} button p{{font-weight:600}}")
        with st.container(horizontal=True, vertical_alignment="center", key=f"hrow-{kind}-{item_id}"):
            st.button(plain_label(it.get("query", "")) or "(empty)", key=f"open_{item_id}", on_click=open_saved, args=(item_id,),
                      width="stretch")
            st.button("Remove", icon=":material/close:", key=f"del_{item_id}", on_click=delete_saved,
                      args=(item_id,), help="Remove from history")
    if len(items) > HISTORY_ROWS:
        st.markdown(f"<p class='a-none'>{len(items) - HISTORY_ROWS} older questions. Search to find them.</p>",
                    unsafe_allow_html=True)
    st.html("<style>" + "".join(css) + "</style>")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"<div class='a-brand'>{MARK}<span>AuroraStays <i>QnA</i></span></div>", unsafe_allow_html=True)
    st.button("New question", icon=":material/add:", on_click=new_conversation, width="stretch",
              type="primary")
    needle = st.text_input("Search history", placeholder="Search questions", label_visibility="collapsed",
                           key="needle", icon=":material/search:")
    # Filled at the end of the run so this run's question is already in the list.
    history_slot = st.container(key="history")
    with st.expander("Settings", icon=":material/tune:"):
        st.toggle("Show sources", value=True, key="show_sources")
        st.selectbox("Model", MODELS, key="model_name")
        st.slider("Passages to retrieve (k)", 1, 10, 6, key="k")
        st.slider("Temperature", 0.0, 1.0, 0.0, 0.05, key="temperature")
        st.slider("Top-p", 0.0, 1.0, 1.0, 0.05, key="top_p")
        st.slider("Max tokens", 64, 2048, 512, 64, key="max_tokens")
    download_slot = st.empty()
    h = get_health(BACKEND_URL)
    if h.get("ok") and h.get("status") == "ok":
        color, label = "var(--a-ok-fg)", f"Online · {h.get('vectors', '?')} passages"
        if not h.get("llm_configured", True):
            color, label = "var(--a-warn-fg)", "Online · language model not set up"
    else:
        color, label = "var(--a-bad-fg)", f"Backend offline · {h.get('error', '')}"
    st.markdown(f"<div class='a-health'><span class='a-dot' style='background:{color}'></span>"
                f"{html.escape(label)}</div>", unsafe_allow_html=True)

params = {"k": ss.get("k", 6), "model_name": ss.get("model_name", MODELS[0]),
          "temperature": ss.get("temperature", 0.0), "top_p": ss.get("top_p", 1.0),
          "max_tokens": ss.get("max_tokens", 512)}
show_sources = ss.get("show_sources", True)

# Read the question before the top bar, so the bar already shows "New" during the first answer.
query = st.chat_input("Ask about a hotel policy", max_chars=1000)
if ss.pending and not query:
    query, ss.pending = ss.pending, None
in_conversation = bool(ss.messages) or query is not None

# ---------------------------------------------------------------- top bar (always reachable on a phone)
with st.container(horizontal=True, vertical_alignment="center", gap="small", key="topbar", width="content"):
    if in_conversation:
        st.button("New", icon=":material/add:", on_click=new_conversation, key="new_top",
                  help="Start a new question")
    dark = ss.theme == "dark"
    st.button("Day mode" if dark else "Night mode", icon=":material/light_mode:" if dark else ":material/dark_mode:",
              on_click=toggle_theme, key="theme_btn",
              help="Switch to day mode" if dark else "Switch to night mode")

# ---------------------------------------------------------------- main
hero = st.empty()  # same slot every run, so the welcome screen clears as soon as a question is asked
if not ss.messages and not query:
    with hero.container():
        st.markdown(f"""<div class='a-hero'>{logo_svg(44, P['accent'], P['on_accent'])}
<h1>What would you like to know?</h1>
<p>Ask a guest-policy question in plain words. Answers come only from the AuroraStay policy documents,
with the sources shown.</p></div><div class='a-try'>Try asking</div>""", unsafe_allow_html=True)
        with st.container(horizontal=True, horizontal_alignment="center", gap="small", key="examples"):
            for i, q in enumerate(EXAMPLES):
                st.button(q, key=f"ex_{i}", on_click=ask, args=(q,), wrap=True)
        st.markdown("<p class='a-foot'>Check the sources before sharing anything marked Assisted review "
                    "or Manual lookup.</p>", unsafe_allow_html=True)

if ss.inflight:
    # The last run was stopped while an answer was being written (a click elsewhere).
    q0, ts0 = ss.inflight
    ss.inflight = None
    ss.messages += [{"role": "user", "content": q0, "ts": ts0},
                    {"role": "assistant", "content": "", "sources": [], "confidence": None, "query": q0,
                     "request_id": None, "error": "The answer was interrupted before it finished."}]

for i, m in enumerate(ss.messages):
    if m["role"] == "user":
        render_question(m["content"], m.get("ts"))
    else:
        render_assistant(m, i, show_sources)

if query is not None:
    q = query.strip() if isinstance(query, str) else str(getattr(query, "text", "")).strip()
    if not q:
        st.markdown("<div class='a-notice warn'><b>Empty question.</b> Type a policy question first.</div>",
                    unsafe_allow_html=True)
    else:
        asked = time.time()
        ss.inflight = (q, asked)
        render_question(q, asked)
        live = st.empty()
        with live.container(key="card_live"):
            slot = st.empty()
            slot.markdown(loading("Searching the policies"), unsafe_allow_html=True)
            msg = run_query(q, params, slot)
        live.empty()
        ss.messages += [{"role": "user", "content": q, "ts": asked}, msg]
        ss.inflight = None
        load_history.clear()
        if msg.get("request_id") and not msg.get("error"):
            ss.active_id = msg["request_id"]
        render_assistant(msg, len(ss.messages) - 1, show_sources)

# ---------------------------------------------------------------- drawn last, so both include this run's answer
with history_slot:
    render_history(load_history(BACKEND_URL) if h.get("ok") else None, needle)
with download_slot:
    st.download_button("Download conversation", data=conversation_markdown() if ss.messages else "",
                       file_name="aurorastays-conversation.md", mime="text/markdown",
                       icon=":material/download:", width="stretch", disabled=not ss.messages,
                       on_click="ignore")
