import streamlit as st
import pandas as pd
import numpy as np
import re
import pypdf
import os

# Import official SDKs safely
try:
    from perplexity import Perplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & EXECUTIVE DARK THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="African PEP & Candidate Data Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #090d16 0%, #0f172a 100%);
            color: #f1f5f9;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 95%; }
        
        section[data-testid="stSidebar"] {
            background-color: #0b1120 !important;
            border-right: 1px solid #1e293b;
        }

        .header-card {
            background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title { font-size: 24px; font-weight: 700; color: #ffffff; margin: 0; }
        .header-badge {
            background: rgba(56, 189, 248, 0.1);
            color: #38bdf8;
            border: 1px solid #0284c7;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }

        .ui-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }

        .info-box {
            background-color: #0b1329;
            border-left: 4px solid #38bdf8;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
            line-height: 1.6;
        }

        div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700; font-size: 24px; }
        .stMetric { background-color: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 12px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PARSER & MUNICIPAL ELECTION LOGIC HELPERS
# -----------------------------------------------------------------------------
MUNI_PREFIXES = [
    "BAARD", "KHOI", "HOOGLAND", "NKONYENI", "ALFRED NZO", "OR TAMBO",
    "AMATHOLE", "CHRIS HANI", "JOE GQABI", "CACADU", "SARAH BAARTMAN"
]

METRO_PREFIXES = ["BUF", "NMB", "CPT", "ETH", "EKU", "TSH", "JHB", "MAN", "City of", "eThekwini"]

def build_master_party_map(party_df=None):
    master_party_map = {}
    max_party_idx = 1

    if party_df is not None:
        for _, row in party_df.iterrows():
            pid = str(row["partyID"]).strip() if pd.notna(row.get("partyID")) else ""
            pname = str(row["name"]).strip() if pd.notna(row.get("name")) else ""
            pabbrv = str(row["abbrv"]).strip() if pd.notna(row.get("abbrv")) else ""

            if pid.startswith("pty_"):
                try:
                    max_party_idx = max(max_party_idx, int(pid.replace("pty_", "")))
                except ValueError:
                    pass

            is_polluted = any(pname.upper().startswith(prefix) for prefix in MUNI_PREFIXES)
            if is_polluted and pabbrv:
                continue

            if pname:
                master_party_map[pname.upper()] = (pid, pname)
            if pabbrv:
                master_party_map[pabbrv.upper()] = (pid, pname if pname else pabbrv)

    master_party_map["INDEPENDENT"] = ("pty_01", "Independent")
    sorted_keys = sorted(master_party_map.keys(), key=len, reverse=True)
    return master_party_map, sorted_keys, max_party_idx

def clean_and_match_party(raw_party_str, master_party_map, sorted_master_keys, max_party_idx):
    if not raw_party_str or pd.isna(raw_party_str):
        return "pty_01", "Independent"

    cleaned = str(raw_party_str).strip().upper()

    if "INDEPENDENT" in cleaned:
        return "pty_01", "Independent"

    for prefix in MUNI_PREFIXES:
        pattern = r"^\b" + re.escape(prefix) + r"\b\s*"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    if cleaned in master_party_map:
        return master_party_map[cleaned]

    for key in sorted_master_keys:
        if key in ["INDEPENDENT", "INDEP"]:
            continue
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, cleaned):
            return master_party_map[key]

    return f"pty_{max_party_idx+1}", cleaned.title()

def parse_name(full_name_str):
    parts = str(full_name_str).strip().split()
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]

def split_muni_and_party(muni_and_party_str):
    muni_and_party_str = muni_and_party_str.strip()

    if "INDEPENDENT" in muni_and_party_str:
        idx = muni_and_party_str.find("INDEPENDENT")
        return muni_and_party_str[:idx].strip(), "INDEPENDENT"

    match = re.search(r"^([A-Z0-9]{2,6}\s*-\s*.*?[a-z].*?)\s+([A-Z0-9]{2,}.*)$", muni_and_party_str)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    parts = re.split(r"\s{2,}", muni_and_party_str)
    if len(parts) >= 2:
        return parts[0].strip(), " ".join(parts[1:]).strip()

    fallback_match = re.match(r"^([A-Z0-9]{2,6}\s*-\s*[\w\s\'-]+?)\s+([A-Z0-9].*)$", muni_and_party_str)
    if fallback_match:
        return fallback_match.group(1).strip(), fallback_match.group(2).strip()

    return muni_and_party_str, muni_and_party_str

def determine_popolo_office(muni_str, ward_order_str):
    muni_str = str(muni_str).strip()
    ward_order_str = str(ward_order_str).strip()
    is_ward = ward_order_str.isdigit() and len(ward_order_str) < 5

    is_metro = any(p in muni_str for p in METRO_PREFIXES)
    is_district = muni_str.startswith("DC") or "DC" in muni_str

    if is_metro:
        return "Metro Council Ward" if is_ward else "Metro PR"
    elif is_district:
        return "District Council PR"
    else:
        return "Local Council Ward" if is_ward else "Local Council PR"

# -----------------------------------------------------------------------------
# 3. API CLIENT RESOLUTION
# -----------------------------------------------------------------------------
def get_perplexity_client():
    api_key = st.secrets.get("PERPLEXITY_API_KEY", os.environ.get("PERPLEXITY_API_KEY", ""))
    if not api_key or not PERPLEXITY_AVAILABLE:
        return None
    try:
        return Perplexity(api_key=api_key)
    except Exception:
        return None

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key or not GROQ_AVAILABLE:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 4. SIDEBAR & NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🌍 PEP Intelligence")
st.sidebar.caption("South Africa LGE Candidate Engine")
st.sidebar.markdown("---")

view_selection = st.sidebar.radio(
    "Go to",
    [
        "📥 Data Ingestion & Parser",
        "🗂️ Popolo Standard Data (6 Tabs)",
        "🌐 Perplexity Search API",
        "⚡ Groq AI Summarizer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("##### ⚙️ API Diagnostics")
st.sidebar.success("⚡ Perplexity API Active") if get_perplexity_client() else st.sidebar.warning("🔑 Perplexity Key Missing")
st.sidebar.success("⚡ Groq API Active") if get_groq_client() else st.sidebar.warning("🔑 Groq Key Missing")

# -----------------------------------------------------------------------------
# 5. WORKSPACE MODULES
# -----------------------------------------------------------------------------
st.markdown(f"""
    <div class="header-card">
        <div>
            <h1 class="header-title">{view_selection}</h1>
            <span style="color: #94a3b8; font-size: 13px;">South Africa LGE Candidate Analysis Hub</span>
        </div>
        <div>
            <span class="header-badge">South Africa 🇿🇦</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if view_selection == "📥 Data Ingestion & Parser":
    party_file = st.file_uploader("Upload Party Master CSV (Optional)", type=["csv"])
    pdf_files = st.file_uploader("Upload Certified Candidate List PDFs", type=["pdf"], accept_multiple_files=True)

    if pdf_files:
        party_df = pd.read_csv(party_file) if party_file else None
        master_party_map, sorted_keys, max_party_idx = build_master_party_map(party_df)

        id_pattern = re.compile(r"([0-9]{6}\*\*\*\*[0-9]{2}\*|[0-9]{13})")
        raw_nominations = []

        with st.spinner("Extracting candidate nominations and parsing ballot types..."):
            for file in pdf_files:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    for line in text.split("\n"):
                        line = line.strip()
                        match = id_pattern.search(line)
                        if match:
                            before_id = line[: match.start()].strip()
                            after_id = line[match.end() :].strip()

                            ward_match = re.search(r"(\d+|PR)$", before_id)
                            if ward_match:
                                ward_order = ward_match.group(1)
                                muni_and_party = before_id[: ward_match.start()].strip()
                            else:
                                ward_order = ""
                                muni_and_party = before_id

                            raw_muni, raw_party = split_muni_and_party(muni_and_party)
                            party_id, clean_party = clean_and_match_party(raw_party, master_party_map, sorted_keys, max_party_idx)
                            full_name = after_id.strip()

                            ballot_category = "WARD" if (ward_order.isdigit() and len(ward_order) < 5) else "PR"

                            raw_nominations.append({
                                "municipality": raw_muni,
                                "party_id": party_id,
                                "party_name": clean_party,
                                "ward_pr_order": ward_order,
                                "ballot_category": ballot_category,
                                "full_name": full_name,
                            })

        # Master Raw Dataframe of all Candidacies
        df_raw = pd.DataFrame(raw_nominations).drop_duplicates(
            subset=["municipality", "party_id", "ward_pr_order", "full_name"]
        )

        df_raw["clean_office"] = df_raw.apply(
            lambda row: determine_popolo_office(row["municipality"], row["ward_pr_order"]),
            axis=1
        )

        # -----------------------------------------------------------------------------
        # CANDIDATE & BALLOT DEDUPLICATION STATS
        # -----------------------------------------------------------------------------
        total_candidacy_records = len(df_raw)
        
        # Categorize per candidate
        candidate_summary = df_raw.groupby("full_name").agg(
            total_contests=("ballot_category", "count"),
            contest_types=("clean_office", lambda x: ", ".join(sorted(set(x)))),
            has_ward=("ballot_category", lambda x: "WARD" in list(x)),
            has_pr=("ballot_category", lambda x: "PR" in list(x)),
            party_name=("party_name", "first"),
            party_id=("party_id", "first")
        ).reset_index()

        total_unique_candidates = len(candidate_summary)
        ward_only_candidates = len(candidate_summary[candidate_summary["has_ward"] & ~candidate_summary["has_pr"]])
        pr_only_candidates = len(candidate_summary[~candidate_summary["has_ward"] & candidate_summary["has_pr"]])
        dual_candidates = len(candidate_summary[candidate_summary["has_ward"] & candidate_summary["has_pr"]])

        # Municipality breakdown
        unique_munis = df_raw["municipality"].unique()
        metro_munis = [m for m in unique_munis if any(p in m for p in METRO_PREFIXES)]
        local_district_munis = [m for m in unique_munis if m not in metro_munis]

        # Construct Persons Table with explicit contest tags
        df_persons = candidate_summary.copy()
        df_persons["id"] = [f"pers_{i+1:05d}" for i in range(len(df_persons))]
        df_persons[["first_name", "middle_name", "last_name"]] = df_persons["full_name"].apply(lambda x: pd.Series(parse_name(x)))
        df_persons["is_dual_candidate"] = df_persons.apply(lambda r: "Yes (Ward + PR)" if (r["has_ward"] and r["has_pr"]) else "No", axis=1)

        df_parties = df_raw[["party_id", "party_name"]].drop_duplicates().reset_index(drop=True)

        st.session_state["popolo_tables"] = {
            "Persons": df_persons,
            "Parties": df_parties,
            "Memberships": df_raw
        }

        # -----------------------------------------------------------------------------
        # UI DISPLAY: BALLOT BREAKDOWN SUMMARY
        # -----------------------------------------------------------------------------
        st.markdown("""
            <div class="info-box">
                <b>📌 Understanding South Africa Local Government Election (LGE) Ballot Counts:</b><br/>
                • <b>Metropolitan Municipalities (2 Ballots):</b> Metro Council Ward + Metro Council PR.<br/>
                • <b>All Other Municipalities (3 Ballots):</b> Local Council Ward + Local Council PR + District Council PR.<br/>
                <i>Note: Candidates frequently run on both a Ward ballot and a PR list, creating multiple candidacy entries for one unique person.</i>
            </div>
        """, unsafe_allow_html=True)

        # Primary Metrics
        st.markdown("### 📊 Candidate & Ballot Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Candidacy Records", f"{total_candidacy_records:,}", help="All registered entries across all ballots")
        m2.metric("Unique Individual Candidates", f"{total_unique_candidates:,}", help="Unique human candidates")
        m3.metric("Dual Candidates (Ward + PR)", f"{dual_candidates:,}", help="Candidates appearing on BOTH Ward and PR lists")
        m4.metric("Total Municipalities", f"{len(unique_munis):,}", help="Unique Municipalities parsed")

        # Secondary Detailed Metrics
        st.markdown("##### Detailed Breakdown by Candidate List & Municipality Type")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ward-Only Candidates", f"{ward_only_candidates:,}")
        c2.metric("PR-Only Candidates", f"{pr_only_candidates:,}")
        c3.metric("Metropolitan Council Munis (2 Ballots)", f"{len(metro_munis):,}")
        c4.metric("Local / District Munis (3 Ballots)", f"{len(local_district_munis):,}")

        st.markdown("---")
        st.markdown("### 🧹 Persons Table Preview (With Dual-Candidacy Tags)")
        st.dataframe(
            df_persons[["id", "full_name", "party_name", "is_dual_candidate", "total_contests", "contest_types"]].head(15),
            use_container_width=True
        )

elif view_selection == "🗂️ Popolo Standard Data ":
    if "popolo_tables" in st.session_state:
        for tab_name, df_table in st.session_state["popolo_tables"].items():
            st.subheader(f"{tab_name} Table")
            st.dataframe(df_table, use_container_width=True)
            st.download_button(
                f"📥 Download {tab_name} CSV",
                df_table.to_csv(index=False).encode('utf-8'),
                f"SA_LGE_{tab_name}.csv",
                "text/csv"
            )
    else:
        st.info("💡 Please upload candidate lists in the Data Ingestion module.")
