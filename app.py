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
# 1. PAGE CONFIG & DARK THEME
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

# Session State Storage
if "popolo_tables" not in st.session_state:
    st.session_state["popolo_tables"] = None

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & POPOLO PIPELINE
# -----------------------------------------------------------------------------
MUNI_PREFIXES = [
    "BAARD", "KHOI", "HOOGLAND", "NKONYENI", "ALFRED NZO", "OR TAMBO",
    "AMATHOLE", "CHRIS HANI", "JOE GQABI", "CACADU", "SARAH BAARTMAN"
]

METRO_PREFIXES = ["BUF", "NMB", "CPT", "ETH", "EKU", "TSH", "JHB", "MAN", "City of", "eThekwini"]

def slugify(text: str) -> str:
    """Creates a clean, consistent ID slug from any text string."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

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

    return f"pty_{max_party_idx+1:02d}", cleaned.title()

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

def process_full_popolo_dataset(df_raw):
    """
    Processes extracted nominations into 6 fully linked Popolo standard tables.
    """
    # -------------------------------------------------------------------------
    # 1. CHAMBERS TABLE (Code for Africa Standard Schema)
    # -------------------------------------------------------------------------
    df_chambers = pd.DataFrame([
        {
            "id": "sa_national_assembly",
            "name": "National Assembly",
            "area_id": "sa_country",
            "office": "National Assembly"
        },
        {
            "id": "sa_provincial_legislature",
            "name": "Provincial Legislature",
            "area_id": "sa_country",
            "office": "Provincial Legislature"
        },
        {
            "id": "sa_municipal_council",
            "name": "Municipal Council",
            "area_id": "sa_country",
            "office": "Municipal Council"
        }
    ])

    # -------------------------------------------------------------------------
    # 2. ROLES TABLE (CfA Schema Compliant: id, title, area_id, role, chamber_id)
    # -------------------------------------------------------------------------
    unique_munis = df_raw["municipality"].drop_duplicates().tolist()
    
    roles_records = []
    muni_to_role_id = {}
    muni_to_area_id = {}
    
    for muni in unique_munis:
        muni_slug = slugify(muni)
        role_id = f"sa_cllr_{muni_slug}"
        area_id = f"sa_ed_{muni_slug}"
        
        muni_to_role_id[muni] = role_id
        muni_to_area_id[muni] = area_id
        
        roles_records.append({
            "id": role_id,
            "title": f"Councillor - {muni.title()}",
            "area_id": area_id,
            "role": "Municipal Councillor",
            "chamber_id": "sa_municipal_council"
        })
        
    df_roles = pd.DataFrame(roles_records)

    # -------------------------------------------------------------------------
    # 3. CONTESTS TABLE
    # -------------------------------------------------------------------------
    df_raw["contest_key"] = df_raw.apply(
        lambda r: f"{slugify(r['municipality'])}_{slugify(r['ward_pr_order'])}", axis=1
    )
    
    contests_records = []
    contest_key_to_id = {}
    
    for idx, row in df_raw[["municipality", "ward_pr_order", "contest_key"]].drop_duplicates().iterrows():
        c_key = row["contest_key"]
        muni = row["municipality"]
        list_type = "WARD" if str(row["ward_pr_order"]).isdigit() and len(str(row["ward_pr_order"])) < 5 else "PR"
        
        contest_id = f"cntst_{c_key}"
        contest_key_to_id[c_key] = contest_id
        
        contests_records.append({
            "id": contest_id,
            "name": f"{muni} - {list_type} Contest ({row['ward_pr_order']})",
            "area_id": muni_to_area_id.get(muni, f"sa_ed_{slugify(muni)}"),
            "election": "South Africa Local Government Elections 2026",
            "type": list_type
        })
        
    df_contests = pd.DataFrame(contests_records).drop_duplicates(subset=["id"])

    # -------------------------------------------------------------------------
    # 4. PARTIES TABLE
    # -------------------------------------------------------------------------
    df_parties = df_raw[["party_id", "party_name"]].drop_duplicates().reset_index(drop=True)
    df_parties = df_parties.rename(columns={"party_id": "id", "party_name": "name"})
    df_parties["country"] = "South Africa"

    # -------------------------------------------------------------------------
    # 5. PERSONS TABLE (Padded 2-digit ID format: pers_01, pers_02, ...)
    # -------------------------------------------------------------------------
    df_persons_unique = df_raw[["full_name"]].drop_duplicates().reset_index(drop=True)
    
    df_persons_unique["id"] = [f"pers_{i+1:02d}" for i in range(len(df_persons_unique))]
    
    parsed_names = df_persons_unique["full_name"].apply(parse_name)
    df_persons_unique["first_name"] = [p[0] for p in parsed_names]
    df_persons_unique["middle_name"] = [p[1] for p in parsed_names]
    df_persons_unique["last_name"] = [p[2] for p in parsed_names]
    df_persons_unique["gender"] = None
    
    df_persons = df_persons_unique[["id", "full_name", "first_name", "middle_name", "last_name", "gender"]]

    # -------------------------------------------------------------------------
    # 6. MEMBERSHIPS TABLE (Central Junction Linking Table)
    # -------------------------------------------------------------------------
    df_m = df_raw.merge(df_persons[["full_name", "id"]], on="full_name", how="left")
    df_m = df_m.rename(columns={"id": "person_id"})

    df_m["role_id"] = df_m["municipality"].map(muni_to_role_id)
    df_m["contest_id"] = df_m["contest_key"].map(contest_key_to_id)
    df_m["list_type"] = df_m["ward_pr_order"].apply(
        lambda x: "WARD" if str(x).isdigit() and len(str(x)) < 5 else "PR"
    )
    
    df_m["id"] = df_m.apply(
        lambda r: f"mshp_{r['person_id']}_{r['role_id']}_{slugify(r['ward_pr_order'])}", axis=1
    )
    
    df_memberships = df_m[[
        "id",
        "person_id",    # FK -> Persons.id
        "party_id",     # FK -> Parties.id
        "role_id",      # FK -> Roles.id
        "contest_id",   # FK -> Contests.id
        "municipality",
        "list_type",
        "ward_pr_order"
    ]].rename(columns={
        "municipality": "Municipality",
        "ward_pr_order": "Ward_PR_Order"
    })

    return {
        "Persons": df_persons,
        "Parties": df_parties,
        "Memberships": df_memberships,
        "Roles": df_roles,
        "Chambers": df_chambers,
        "Contests": df_contests,
        "Raw_Noms": df_raw
    }

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

nav_options = {
    "ingest": "📥 Data Ingestion & Parser",
    "popolo": "🗂️ Popolo Standard Data (6 Tabs)",
    "perplexity": "🌐 Perplexity Search API",
    "groq": "⚡ Groq AI Summarizer"
}

selected_label = st.sidebar.radio(
    "Go to",
    options=list(nav_options.values())
)

view_key = [k for k, v in nav_options.items() if v == selected_label][0]

st.sidebar.markdown("---")
st.sidebar.markdown("##### ⚙️ API Diagnostics")

if get_perplexity_client():
    st.sidebar.success("⚡ Perplexity API Active")
else:
    st.sidebar.warning("🔑 Perplexity Key Missing")

if get_groq_client():
    st.sidebar.success("⚡ Groq API Active")
else:
    st.sidebar.warning("🔑 Groq Key Missing")

# Header Rendering
st.markdown(f"""
    <div class="header-card">
        <div>
            <h1 class="header-title">{selected_label}</h1>
            <span style="color: #94a3b8; font-size: 13px;">South Africa LGE Candidate Analysis Hub</span>
        </div>
        <div>
            <span class="header-badge">South Africa 🇿🇦</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 1: DATA INGESTION & PARSER
# -----------------------------------------------------------------------------
if view_key == "ingest":
    party_file = st.file_uploader("Upload Party Master CSV (Optional)", type=["csv"])
    pdf_files = st.file_uploader("Upload Certified Candidate List PDFs", type=["pdf"], accept_multiple_files=True)

    if pdf_files:
        party_df = pd.read_csv(party_file) if party_file else None
        master_party_map, sorted_keys, max_party_idx = build_master_party_map(party_df)

        id_pattern = re.compile(r"([0-9]{6}\*\*\*\*[0-9]{2}\*|[0-9]{13})")
        raw_nominations = []

        with st.spinner("Extracting candidate nominations from PDFs..."):
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

                            raw_nominations.append({
                                "municipality": raw_muni,
                                "party_id": party_id,
                                "party_name": clean_party,
                                "ward_pr_order": ward_order,
                                "full_name": full_name,
                            })

        df_raw = pd.DataFrame(raw_nominations).drop_duplicates(
            subset=["municipality", "party_id", "ward_pr_order", "full_name"]
        )

        st.session_state["popolo_tables"] = process_full_popolo_dataset(df_raw)
        st.success("✅ Data ingested successfully! Saved across all 6 Popolo standard tables.")

    if st.session_state["popolo_tables"] is not None:
        pop = st.session_state["popolo_tables"]
        df_persons = pop["Persons"]
        df_memberships = pop["Memberships"]

        total_candidacy_records = len(df_memberships)
        total_unique_candidates = len(df_persons)

        candidate_lists = df_memberships.groupby("person_id")["list_type"].unique()
        dual_candidates = sum(candidate_lists.apply(lambda x: "WARD" in x and "PR" in x))
        ward_only_candidates = sum(candidate_lists.apply(lambda x: "WARD" in x and "PR" not in x))
        pr_only_candidates = sum(candidate_lists.apply(lambda x: "WARD" not in x and "PR" in x))

        total_ward_nominations = sum(df_memberships["list_type"] == "WARD")
        total_pr_nominations = sum(df_memberships["list_type"] == "PR")

        unique_munis = df_memberships["Municipality"].unique()
        metro_munis = [m for m in unique_munis if any(p in m for p in METRO_PREFIXES)]
        local_district_munis = [m for m in unique_munis if m not in metro_munis]

        st.markdown("""
            <div class="info-box">
                <b>📌 Types of Elections & Ballot System Breakdown (By Municipality Type):</b><br/>
                • <b>Metropolitan Municipalities – 2 Ballots per Ward:</b><br/>
                &nbsp;&nbsp;&nbsp;&nbsp;1. Metropolitan Council Ward Ballot<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;2. Metropolitan Proportional Representation (PR) Ballot<br/>
                • <b>All Other Municipalities – 3 Ballots per Ward:</b><br/>
                &nbsp;&nbsp;&nbsp;&nbsp;1. Local Council Ward Ballot<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;2. Local Council Proportional Representation (PR) Ballot<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;3. District Council Proportional Representation (PR) Ballot<br/>
                <br/>
                <i>Note: Candidates frequently stand for election in both a Ward contest and on a party's PR list, leading to multiple candidacy records for a single unique individual.</i>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Dataset & Candidacy Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Candidacy Records", f"{total_candidacy_records:,}")
        m2.metric("Unique Individual Candidates", f"{total_unique_candidates:,}")
        m3.metric("Ward Nominations Count", f"{total_ward_nominations:,}")
        m4.metric("PR Nominations Count", f"{total_pr_nominations:,}")

        st.markdown("##### Candidate Dual-Standing & Municipal Distribution")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dual Candidates (Ward + PR)", f"{dual_candidates:,}")
        c2.metric("Ward-Only Candidates", f"{ward_only_candidates:,}")
        c3.metric("PR-Only Candidates", f"{pr_only_candidates:,}")
        c4.metric("Total Unique Municipalities", f"{len(unique_munis):,}")

        st.markdown("##### Municipality Classification")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Metropolitan Municipalities (2 Ballots)", f"{len(metro_munis):,}")
        m_col2.metric("Local & District Municipalities (3 Ballots)", f"{len(local_district_munis):,}")

        st.markdown("---")
        st.markdown("### 🧹 Persons Table Preview")
        st.dataframe(df_persons.head(15), use_container_width=True)

        if st.button("🗑️ Clear & Reset Dataset"):
            st.session_state["popolo_tables"] = None
            st.rerun()

# -----------------------------------------------------------------------------
# VIEW 2: POPOLO STANDARD DATA (6 TABS)
# -----------------------------------------------------------------------------
elif view_key == "popolo":
    if st.session_state.get("popolo_tables") is not None:
        pop = st.session_state["popolo_tables"]
        
        tab_names = ["Persons", "Parties", "Memberships", "Roles", "Chambers", "Contests"]
        tabs = st.tabs([f"{i+1}. {name}" for i, name in enumerate(tab_names)])

        for idx, tab in enumerate(tabs):
            with tab:
                key = tab_names[idx]
                curr_df = pop[key]
                st.markdown(f"### {key} Table ({len(curr_df):,} records)")
                st.dataframe(curr_df, use_container_width=True)
                
                csv_bytes = curr_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download {key} CSV",
                    data=csv_bytes,
                    file_name=f"Master_{key}_Cleaned_Popolo.csv",
                    mime="text/csv",
                    key=f"dl_btn_{key}"
                )
    else:
        st.info("💡 No active dataset found in memory. Please upload candidate PDFs in the **Data Ingestion & Parser** tab first.")

# -----------------------------------------------------------------------------
# VIEW 3: PERPLEXITY SEARCH API
# -----------------------------------------------------------------------------
elif view_key == "perplexity":
    st.write("### Perplexity Search Hub")
    st.info("Query real-time intelligence for South African political entities and PEPs.")

# -----------------------------------------------------------------------------
# VIEW 4: GROQ AI SUMMARIZER
# -----------------------------------------------------------------------------
elif view_key == "groq":
    st.write("### Groq High-Speed AI Analysis")
    st.info("Generate high-speed summaries and reports for candidates and political parties.")
