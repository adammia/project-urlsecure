
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tldextract
from dotenv import load_dotenv
from pybloom_live import BloomFilter

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
WHITELIST_CSV_PATH = APP_ROOT / "white_list_tranco_clean_for_a_zone.csv"
MODEL_PATH = APP_ROOT / "stage1_model.joblib"
META_PATH = APP_ROOT / "stage1_meta.json"
API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY", "").strip()

WEBRISK_ENDPOINT = "https://webrisk.googleapis.com/v1/uris:search"
DEFAULT_THREAT_TYPES = ["SOCIAL_ENGINEERING", "MALWARE", "UNWANTED_SOFTWARE"]

BYPASS_WHITELIST_FOR_TEST = True

SHARED_HOSTING_SUFFIXES = {
    "appspot.com", "github.io", "pages.dev", "vercel.app", "netlify.app",
    "cloudfront.net", "firebaseapp.com", "web.app", "herokuapp.com", "azurewebsites.net"
}
DOMAIN_PATTERN = re.compile(r"^[a-z0-9._-]+$")

st.set_page_config(page_title="BankSecure – integrált Stage–1", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

def ensure_scheme(url: str) -> str:
    url = str(url).strip()
    if not url:
        raise ValueError("Az URL üres.")
    if "://" not in url:
        return "http://" + url
    return url

def extract_hostname(url: str) -> str:
    parsed = urlparse(ensure_scheme(url))
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise ValueError(f"Nem sikerült hostnevet kinyerni: {url}")
    return hostname

def registrable_domain(hostname: str) -> str:
    extracted = tldextract.extract(hostname)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return hostname.lower()

def normalize_domain(value: object) -> Optional[str]:
    if value is None:
        return None
    domain = str(value).strip().lower().rstrip(".")
    if not domain:
        return None
    if domain.startswith("*."):
        domain = domain[2:]
    if domain == "_wildcard_":
        return None
    if " " in domain or "." not in domain:
        return None
    if not DOMAIN_PATTERN.match(domain):
        return None
    return domain

def is_shared_hosting_domain(base_domain: str) -> bool:
    return base_domain.lower() in SHARED_HOSTING_SUFFIXES

@st.cache_resource(show_spinner=False)
def build_bloom_from_csv(path_str: str):
    csv_path = Path(path_str)
    df = pd.read_csv(csv_path)
    col = "domain_clean" if "domain_clean" in df.columns else "domain"
    domains = [normalize_domain(x) for x in df[col].tolist()]
    domains = [d for d in domains if d]
    bloom = BloomFilter(capacity=max(1000, int(len(domains) * 1.05)), error_rate=1e-6)
    for d in domains:
        bloom.add(d)
    return bloom

def zone0_check(url: str, bloom_filter) -> Tuple[str, str, str]:
    hostname = extract_hostname(url)
    base_domain = registrable_domain(hostname)
    if is_shared_hosting_domain(base_domain):
        hit = hostname in bloom_filter
    else:
        hit = (hostname in bloom_filter) or (base_domain in bloom_filter)
    return ("ALLOW_WHITE_LIST" if hit else "MISS"), hostname, base_domain

def check_webrisk_service(api_key: str) -> dict:
    if not api_key:
        return {"enabled": False, "ok": False, "status_code": None, "detail": "Nincs megadva API kulcs."}
    try:
        resp = requests.get(
            WEBRISK_ENDPOINT,
            params=[("uri", "http://example.org"), ("threatTypes", "SOCIAL_ENGINEERING")],
            headers={"x-goog-api-key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            return {"enabled": True, "ok": True, "status_code": 200, "detail": "A Web Risk elérhető."}
        return {"enabled": True, "ok": False, "status_code": resp.status_code, "detail": resp.text[:400]}
    except Exception as exc:
        return {"enabled": True, "ok": False, "status_code": None, "detail": str(exc)}

class WebRiskLookupClient:
    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()

    def lookup_url(self, url: str):
        if not self.api_key:
            return {"signal": "PASS_TO_STAGE1", "threats": [], "error": "NO_API_KEY", "status_code": None}

        params = [("uri", ensure_scheme(url))]
        for threat in DEFAULT_THREAT_TYPES:
            params.append(("threatTypes", threat))

        try:
            response = requests.get(
                WEBRISK_ENDPOINT,
                params=params,
                headers={"x-goog-api-key": self.api_key},
                timeout=15,
            )
            payload = response.json() if response.content else {}
            if response.status_code == 200:
                threats = payload.get("threat", {}).get("threatTypes", [])
                if threats:
                    return {"signal": "BLOCK_WEBRISK", "threats": threats, "error": None, "status_code": 200}
                return {"signal": "PASS_TO_STAGE1", "threats": [], "error": None, "status_code": 200}
            return {"signal": "WEBRISK_ERROR", "threats": [], "error": payload if payload else response.text, "status_code": response.status_code}
        except Exception as exc:
            return {"signal": "WEBRISK_ERROR", "threats": [], "error": str(exc), "status_code": None}

@st.cache_resource(show_spinner=False)
def load_stage1():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta

def count_chars(s: str, chars: str) -> int:
    return sum(s.count(ch) for ch in chars)

def is_ip_host(host: str) -> int:
    m = re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host)
    return int(bool(m))

def shortener_flag(host: str) -> int:
    shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "cutt.ly"]
    return int(any(host.endswith(s) or host == s for s in shorteners))

def compute_stage1_features(url: str, feature_cols: list[str]) -> pd.DataFrame:
    """
    A Stage–1 feature-képzését a tanítóbázis kódolásával összhangban állítja elő.
    Itt nem imputálás történik, hanem ugyanannak a szemantikának a megőrzése:
    ha nincs directory / file / query rész az URL-ben, akkor az ezekhez tartozó
    feature-ök -1 értéket kapnak, mert a modell ezt a jelölést tanulta.
    """
    parsed = urlparse(ensure_scheme(url))
    full = ensure_scheme(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # A tanítóhalmaz logikája alapján a hiányzó path-részeket -1-gyel kell jelölni.
    if path and path != "/":
        if "/" in path:
            directory = path.rsplit("/", 1)[0]
            if directory.startswith("/"):
                directory = directory[1:]
            file_part = path.rsplit("/", 1)[-1]
        else:
            directory = ""
            file_part = path
    else:
        directory = None
        file_part = None

    if directory == "":
        directory = None
    if file_part == "":
        file_part = None

    params_count = len(parse_qs(query, keep_blank_values=True)) if query else -1

    punct = {
        "dot": ".", "hyphen": "-", "underline": "_", "slash": "/", "questionmark": "?",
        "equal": "=", "at": "@", "and": "&", "exclamation": "!", "space": " ",
        "tilde": "~", "comma": ",", "plus": "+", "asterisk": "*", "hashtag": "#",
        "dollar": "$", "percent": "%"
    }

    vals = {}

    # URL- és domain-feature-ök mindig értelmezhetők
    for name, ch in punct.items():
        vals[f"qty_{name}_url"] = full.count(ch)
        vals[f"qty_{name}_domain"] = host.count(ch)

    vals["length_url"] = len(full)
    vals["qty_vowels_domain"] = sum(host.count(v) for v in "aeiou")
    vals["domain_length"] = len(host)
    vals["domain_in_ip"] = is_ip_host(host)
    vals["email_in_url"] = int("@" in full)
    vals["url_shortened"] = shortener_flag(host)

    # Directory-feature-ök
    if directory is None:
        for name in punct:
            vals[f"qty_{name}_directory"] = -1
        vals["directory_length"] = -1
    else:
        for name, ch in punct.items():
            vals[f"qty_{name}_directory"] = directory.count(ch)
        vals["directory_length"] = len(directory)

    # File-feature-ök
    if file_part is None:
        for name in punct:
            vals[f"qty_{name}_file"] = -1
        vals["file_length"] = -1
    else:
        for name, ch in punct.items():
            vals[f"qty_{name}_file"] = file_part.count(ch)
        vals["file_length"] = len(file_part)

    # Paraméter-feature-ök
    if not query:
        for name in punct:
            vals[f"qty_{name}_params"] = -1
        vals["params_length"] = -1
        vals["qty_params"] = -1
    else:
        for name, ch in punct.items():
            vals[f"qty_{name}_params"] = query.count(ch)
        vals["params_length"] = len(query)
        vals["qty_params"] = params_count

    row = {col: vals.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row])

def stage1_signal_from_score(score: float, low: float, high: float) -> str:
    if score < low:
        return "ALLOW_STAGE1"
    if score >= high:
        return "REVIEW_STAGE1_HIGH_RISK"
    return "PASS_TO_STAGE2"

def feature_explanation(model, feature_df, feature_cols, top_n=5):
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        vals = np.abs(np.array(estimator.feature_importances_).reshape(-1))
    elif hasattr(estimator, "coef_"):
        vals = np.abs(np.array(estimator.coef_).reshape(-1))
    else:
        vals = np.zeros(len(feature_cols))
    pairs = sorted(zip(feature_cols, vals), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"feature": k, "impact": float(v)} for k, v in pairs]

def evaluate_url(url: str, bloom_filter):
    model, meta = load_stage1()
    low = meta["low_risk_threshold"]
    high = meta["high_risk_threshold"]
    feature_cols = meta["feature_columns"]

    url = ensure_scheme(url)
    zone0_signal, hostname, base_domain = zone0_check(url, bloom_filter)

    if zone0_signal == "ALLOW_WHITE_LIST":
        return {
            "url": url,
            "hostname": hostname,
            "registrable_domain": base_domain,
            "zone0_signal": zone0_signal,
            "a_zone_signal": "SKIPPED",
            "stage1_score": 0.0,
            "stage1_signal": "SKIPPED",
            "top_feature_explanation": [],
            "final_signal": "ALLOW",
            "final_severity": "green",
            "user_title": "Az URL a 0. zóna alapján legitimként átengedhető.",
            "user_message": "A domain szerepel a whitelistben.",
            "state_icon": "✅",
            "webrisk_status": None,
        }

    web = WebRiskLookupClient(API_KEY).lookup_url(url)
    if web["signal"] == "BLOCK_WEBRISK":
        return {
            "url": url,
            "hostname": hostname,
            "registrable_domain": base_domain,
            "zone0_signal": zone0_signal,
            "a_zone_signal": "BLOCK_WEBRISK",
            "stage1_score": None,
            "stage1_signal": "SKIPPED_BLOCKED_EARLY",
            "top_feature_explanation": [],
            "final_signal": "BLOCK",
            "final_severity": "red",
            "user_title": "Az URL az A zóna alapján kockázatosnak minősült.",
            "user_message": "A Web Risk ismert fenyegetést jelzett.",
            "state_icon": "⛔",
            "webrisk_status": web,
        }

    feature_df = compute_stage1_features(url, feature_cols)
    score = float(model.predict_proba(feature_df)[0, 1])
    signal = stage1_signal_from_score(score, low, high)
    explanation = feature_explanation(model, feature_df, feature_cols, top_n=5)

    if signal == "ALLOW_STAGE1":
        final_signal = "ALLOW"
        severity = "green"
        title = "Az URL Stage–1 alapján alacsony kockázatú."
        message = "A gyors URL-alapú modell szerint az oldal átengedhető."
        icon = "✅"
    elif signal == "REVIEW_STAGE1_HIGH_RISK":
        final_signal = "REVIEW"
        severity = "red"
        title = "Az URL Stage–1 alapján magas kockázatú."
        message = "A gyors URL-alapú modell alapján további blokkolás vagy manuális review indokolt."
        icon = "⛔"
    else:
        final_signal = "PASS_TO_STAGE2"
        severity = "yellow"
        title = "Az URL további vizsgálatot igényel."
        message = "A köztes kockázati pontszám miatt az URL az Enriched / Stage–2 réteg felé továbbítható."
        icon = "⚠️"

    return {
        "url": url,
        "hostname": hostname,
        "registrable_domain": base_domain,
        "zone0_signal": zone0_signal,
        "a_zone_signal": web["signal"],
        "stage1_score": score,
        "stage1_signal": signal,
        "top_feature_explanation": explanation,
        "final_signal": final_signal,
        "final_severity": severity,
        "user_title": title,
        "user_message": message,
        "state_icon": icon,
        "webrisk_status": web,
    }


st.markdown("""
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1240px;
}
header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}
div[data-testid="stToolbar"] {
    visibility: hidden;
    height: 0;
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 1.8rem 1.8rem;
    border-radius: 28px;
    background: linear-gradient(135deg, #08224a 0%, #143b86 52%, #2f5dff 100%);
    color: white;
    margin-top: 0.35rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 24px 48px rgba(7,27,58,.16);
}
.hero::before {
    content: "";
    position: absolute;
    right: -90px;
    top: -70px;
    width: 380px;
    height: 230px;
    background: radial-gradient(circle, rgba(255,255,255,.10) 0%, rgba(139,92,246,.12) 28%, rgba(255,255,255,0) 70%);
    filter: blur(12px);
}
.shield-box {
    width: 92px;
    height: 92px;
    border-radius: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(180deg, rgba(255,255,255,.20), rgba(255,255,255,.06));
    border: 1px solid rgba(255,255,255,.16);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.25);
    font-size: 44px;
    flex: 0 0 auto;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.04;
    margin: 0;
}
.hero-sub {
    margin-top: 12px;
    font-size: 1.1rem;
    opacity: .96;
}
.browser-bar {
    display:flex;
    align-items:center;
    gap:12px;
    background: rgba(255,255,255,.92);
    border:1px solid #dbe4ee;
    border-radius: 20px;
    padding: 12px 16px;
    box-shadow: 0 12px 30px rgba(15,23,42,.05);
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.browser-dot {width: 12px; height: 12px; border-radius: 50%;}
.browser-icon {font-size: 18px; color: #6b7a90; opacity: .95;}
.frame {
    background: white;
    border: 1px solid #e6edf5;
    border-radius: 24px;
    box-shadow: 0 18px 42px rgba(15,23,42,.06);
    overflow: hidden;
}
.safe-page, .warning-page {
    padding: 1.6rem 1.7rem 1.8rem 1.7rem;
    min-height: 280px;
}
.safe-page {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
}
.safe-badge {
    display:inline-block;
    padding: .42rem .78rem;
    border-radius: 999px;
    background: rgba(22,163,74,.10);
    color: #15803d;
    font-weight: 700;
    margin-bottom: .95rem;
}
.user-title {
    margin: 0 0 10px 0;
    color:#0f172a;
    font-size: 1.1rem;
    font-weight: 700;
}
.user-sub {
    margin:0 0 16px 0;
    color:#334155;
    font-size: .95rem;
}
.content-grid {
    display:grid;
    grid-template-columns: 1.1fr .9fr;
    gap: 18px;
    align-items: stretch;
}
.info-card, .action-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #dbeafe;
    border-radius: 20px;
    padding: 1.1rem 1.15rem;
    box-shadow: 0 10px 28px rgba(37,99,235,.06);
}
.warning-review {
    background: linear-gradient(180deg, rgba(245,158,11,.10), rgba(217,119,6,.08));
}
.warning-block {
    background: radial-gradient(circle at top right, rgba(225,29,141,.20), transparent 30%), linear-gradient(180deg, rgba(127,29,29,.07), rgba(225,29,141,.08));
}
.warning-card {
    background: rgba(255,255,255,.74);
    border: 1px solid rgba(255,255,255,.55);
    border-radius: 22px;
    padding: 1.15rem 1.2rem;
    box-shadow: 0 18px 36px rgba(15,23,42,.08);
}
.meta-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 24px rgba(15,23,42,.05);
}
.pill {
    display:inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    border:1px solid #cbd5e1;
    background:#f8fafc;
    margin-right:.35rem;
    margin-bottom:.35rem;
    font-size:.85rem;
    font-weight:600;
}
div.stButton > button {
    background: #e11d8d;
    color: white;
    border: none;
    border-radius: 999px;
    padding-top: 0.92rem;
    padding-bottom: 0.92rem;
    font-weight: 800;
    font-size: 1.02rem;
    box-shadow: 0 12px 28px rgba(225,29,141,.24);
}
div.stButton > button:hover {
    background: #c2187a;
    color: white;
}
section[data-testid="stSidebar"] {
    display:none;
}
@media (max-width: 900px) {
    .content-grid {grid-template-columns: 1fr;}
    .hero-title {font-size: 2rem;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div style="display:flex; align-items:center; gap:18px;">
    <div class="shield-box">🛡️</div>
    <div>
      <div class="hero-title">BankSecure – integrált Stage–1</div>
      <div class="hero-sub">A zóna és Stage–1 URL-kockázatértékelés egységes felületen.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not WHITELIST_CSV_PATH.exists():
    st.error("A whitelist fájl nem található.")
    st.stop()

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("A Stage–1 modell vagy a meta fájl nem található.")
    st.stop()

bloom = build_bloom_from_csv(str(WHITELIST_CSV_PATH))
webrisk_check = check_webrisk_service(API_KEY)
current_icon = st.session_state.get("last_state_icon", "")

if BYPASS_WHITELIST_FOR_TEST:
    st.warning("Tesztmód aktív: a whitelist ideiglenesen ki van kerülve.")

st.markdown(f"""
<div class="browser-bar">
  <div class="browser-dot" style="background:#ef4444;"></div>
  <div class="browser-dot" style="background:#f59e0b;"></div>
  <div class="browser-dot" style="background:#22c55e;"></div>
  <div style="color:#64748b; font-size:0.95rem; flex:1;">Kérjük, adja meg az URL-t</div>
  <div class="browser-icon">{current_icon}</div>
  <div class="browser-icon">⟳</div>
  <div class="browser-icon">✕</div>
</div>
""", unsafe_allow_html=True)

url = st.text_input("URL", placeholder="https://example.org", label_visibility="collapsed")
run = st.button("Vizsgálat indítása", type="primary", use_container_width=True)

if run:
    if not url.strip():
        st.error("Adj meg egy URL-t.")
    else:
        result = evaluate_url(url, bloom)
        st.session_state["last_state_icon"] = result["state_icon"]
        st.session_state["last_result"] = result
        st.experimental_rerun()

result = st.session_state.get("last_result")
if result:
    stage1_score_text = (
        f'{result["stage1_score"]:.4f}'
        if isinstance(result.get("stage1_score"), (int, float))
        else "n/a"
    )

    if result["final_severity"] == "green":
        st.markdown(
            f"""
            <div class="frame">
              <div class="safe-page">
                <div class="safe-badge">Biztonságos oldal</div>
                <div class="content-grid">
                  <div class="info-card">
                    <h2 class="user-title">{result["user_title"]}</h2>
                    <p class="user-sub">{result["user_message"]}</p>
                    <div class="pill">Stage–1 score: {stage1_score_text}</div>
                    <div class="pill">Végső jelzés: {result["final_signal"]}</div>
                  </div>
                  <div class="action-card">
                    <div style="font-size:1.02rem; font-weight:700; color:#0f172a;">A böngészés folytatható</div>
                    <div style="margin-top:10px; color:#475569;">A megadott cím: <strong>{result["url"]}</strong></div>
                    <div style="margin-top:16px;">
                      <a href="{result["url"]}" target="_blank" style="
                          display:inline-block;
                          padding:.78rem 1.1rem;
                          background:#e11d8d;
                          color:white;
                          text-decoration:none;
                          border-radius:999px;
                          font-weight:700;
                      ">Oldal megnyitása új lapon</a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        warning_class = "warning-review" if result["final_severity"] == "yellow" else "warning-block"
        accent = "#a16207" if result["final_severity"] == "yellow" else "#be185d"
        stage2_hint = ""
        if result["final_signal"] == "PASS_TO_STAGE2":
            stage2_hint = '<div style="margin-top:14px; color:#475569; font-size:.94rem;">A Stage–2 szövegbányászati mélyelemzés indítása javasolt.</div>'
        st.markdown(
            f"""
            <div class="frame">
              <div class="warning-page {warning_class}">
                <div class="warning-card">
                  <div style="font-size:1.8rem; line-height:1;">{result["state_icon"]}</div>
                  <h2 style="margin:10px 0 6px 0; color:{accent}; font-size:1.12rem;">{result["user_title"]}</h2>
                  <div style="font-size:.95rem; color:#334155;">{result["user_message"]}</div>
                  <div style="margin-top:16px; padding:12px 14px; border-radius:16px; background:rgba(255,255,255,.72); color:#475569;">
                    A megadott URL: <strong>{result["url"]}</strong>
                  </div>
                  <div style="margin-top:12px;">
                    <span class="pill">Stage–1 score: {stage1_score_text}</span>
                    <span class="pill">Stage–1 jelzés: {result["stage1_signal"]}</span>
                  </div>
                  {stage2_hint}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("Projektinformáció"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="meta-card"><div style="color:#64748b;font-size:13px;">Host</div><div><strong>{result["hostname"]}</strong></div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="meta-card"><div style="color:#64748b;font-size:13px;">Domain</div><div><strong>{result["registrable_domain"]}</strong></div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="meta-card"><div style="color:#64748b;font-size:13px;">Jelzés</div><div><strong>{result["final_signal"]}</strong></div></div>', unsafe_allow_html=True)

        st.markdown("### Pipeline jelek")
        st.markdown(
            f'<span class="pill">0. zóna: {result["zone0_signal"]}</span>'
            f'<span class="pill">A zóna: {result["a_zone_signal"]}</span>'
            f'<span class="pill">Stage–1: {result["stage1_signal"]}</span>',
            unsafe_allow_html=True
        )

        webrisk_status = result.get("webrisk_status") or {}

        if webrisk_status.get("threat_types"):
            st.error("Web Risk threat típusok: " + ", ".join(webrisk_status["threat_types"]))
        if webrisk_status.get("error"):
            st.caption("Web Risk hiba: " + str(webrisk_status["error"]))

        st.markdown("### Stage–1 top jellemzők")
        explanation = result.get("top_feature_explanation", [])
        if explanation:
            for item in explanation:
                st.markdown(f'- **{item["feature"]}**: {item["impact"]:.6f}')
        else:
            st.caption("Nincs elérhető feature magyarázat ehhez a döntéshez.")

        st.markdown("### Technikai állapot")
        st.json({
            "webrisk_service_check": webrisk_check,
            "pipeline_result": result
        })
