import streamlit as st
import pandas as pd
import numpy as np
import re
import pypdf
import os
import json

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
    """Processes extracted nominations into 6 fully linked Popolo standard tables."""
    
    # -------------------------------------------------------------------------
    # 1. CHAMBERS TABLE
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
    # 2. CONTESTS TABLE
    # -------------------------------------------------------------------------
    contests_records = []
    contest_key_to_id = {}
    
    for idx, row in df_raw[["municipality", "ward_pr_order"]].drop_duplicates().iterrows():
        muni = row["municipality"]
        w_pr = str(row["ward_pr_order"]).strip()
        muni_slug = slugify(muni)
        
        if w_pr.isdigit() and len(w_pr) < 5:
            contest_type = "Ward"
            contest_id = f"sa_ward_{muni_slug}_w{w_pr}_2026"
            contest_name = f"Ward {w_pr} - {muni.title()}"
        else:
            contest_type = "PR"
            contest_id = f"sa_pr_{muni_slug}_2026"
            contest_name = f"PR List - {muni.title()}"
            
        contest_key_to_id[(muni, w_pr)] = contest_id
        
        contests_records.append({
            "id": contest_id,
            "name": contest_name,
            "area_id": f"sa_ed_{muni_slug}",
            "election_id": "sa_lge_2026",
            "type": contest_type,
            "muni_slug": muni_slug,
            "muni_raw": muni
        })
        
    df_contests = pd.DataFrame(contests_records).drop_duplicates(subset=["id"])

    # -------------------------------------------------------------------------
    # 3. ROLES TABLE
    # -------------------------------------------------------------------------
    roles_records = []
    for _, c_row in df_contests.iterrows():
        cid = c_row["id"]
        muni_raw = c_row["muni_raw"]
        
        if "sa_ward_" in cid:
            muni_slug = cid.replace("sa_ward_", "").split("_w")[0]
        elif "sa_pr_" in cid:
            muni_slug = cid.replace("sa_pr_", "").replace("_2026", "")
        else:
            muni_slug = c_row["muni_slug"]

        role_id = f"sa_cllr_{muni_slug}"
        area_id = f"sa_ed_{muni_slug}"
        
        roles_records.append({
            "id": role_id,
            "title": f"Councillor - {muni_slug.replace('_', ' ').title()}",
            "area_id": area_id,
            "role": "Municipal Councillor",
            "chamber_id": "sa_municipal_council"
        })
        
    df_roles = pd.DataFrame(roles_records).drop_duplicates(subset=["id"])

    # -------------------------------------------------------------------------
    # 4. PARTIES TABLE
    # -------------------------------------------------------------------------
    df_parties = df_raw[["party_id", "party_name"]].drop_duplicates().reset_index(drop=True)
    df_parties = df_parties.rename(columns={"party_id": "id", "party_name": "name"})
    df_parties["country"] = "South Africa"
    df_parties["fb_urls"] = "N/A"
    df_parties["x_urls"] = "N/A"
    df_parties["ig_urls"] = "N/A"
    df_parties["tiktok_urls"] = "N/A"
    df_parties["yt_urls"] = "N/A"
    df_parties["linkedin_urls"] = "N/A"
    df_parties["wa_numbers"] = "N/A"
    df_parties["websites"] = "N/A"

    # -------------------------------------------------------------------------
    # 5. PERSONS TABLE
    # -------------------------------------------------------------------------
    df_persons_unique = df_raw[["full_name"]].drop_duplicates().reset_index(drop=True)
    df_persons_unique["id"] = [f"pers_{i+1:02d}" for i in range(len(df_persons_unique))]
    
    parsed_names = df_persons_unique["full_name"].apply(parse_name)
    df_persons_unique["first_name"] = [p[0] for p in parsed_names]
    df_persons_unique["middle_name"] = [p[1] for p in parsed_names]
    df_persons_unique["last_name"] = [p[2] for p in parsed_names]
    df_persons_unique["gender"] = None
    df_persons_unique["fb_urls"] = "N/A"
    df_persons_unique["x_urls"] = "N/A"
    df_persons_unique["ig_urls"] = "N/A"
    df_persons_unique["tiktok_urls"] = "N/A"
    df_persons_unique["yt_urls"] = "N/A"
    df_persons_unique["linkedin_urls"] = "N/A"
    df_persons_unique["wa_numbers"] = "N/A"
    df_persons_unique["websites"] = "N/A"
    
    df_persons = df_persons_unique[[
        "id", "full_name", "first_name", "middle_name", "last_name", "gender",
        "fb_urls", "x_urls", "ig_urls", "tiktok_urls", "yt_urls", "linkedin_urls", "wa_numbers", "websites"
    ]]

    # -------------------------------------------------------------------------
    # 6. MEMBERSHIPS TABLE
    # -------------------------------------------------------------------------
    df_m = df_raw.merge(df_persons[["full_name", "id"]], on="full_name", how="left")
    df_m = df_m.rename(columns={"id": "person_id"})

    df_m["contest_id"] = df_m.apply(
        lambda r: contest_key_to_id.get((r["municipality"], str(r["ward_pr_order"]).strip())), axis=1
    )
    
    def resolve_role_id(contest_id, raw_muni):
        if pd.notna(contest_id):
            if "sa_ward_" in str(contest_id):
                muni_slug = str(contest_id).replace("sa_ward_", "").split("_w")[0]
                return f"sa_cllr_{muni_slug}"
            elif "sa_pr_" in str(contest_id):
                muni_slug = str(contest_id).replace("sa_pr_", "").replace("_2026", "")
                return f"sa_cllr_{muni_slug}"
        return f"sa_cllr_{slugify(raw_muni)}"

    df_m["role_id"] = df_m.apply(lambda r: resolve_role_id(r["contest_id"], r["municipality"]), axis=1)
    df_m["id"] = df_m["person_id"].apply(lambda pid: f"mshp_{pid}_26")
    df_m["membership_type"] = "Municipal Councillor"
    df_m["start_date"] = "2026-09-04"
    df_m["end_date"] = "2026-09-04"
    df_m["is_partisan"] = True
    df_m["has_end_date"] = True

    df_memberships = df_m[[
        "id", "role_id", "person_id", "party_id", "membership_type",
        "start_date", "end_date", "is_partisan", "has_end_date", "contest_id"
    ]]

    df_contests_clean = df_contests.drop(columns=["muni_slug", "muni_raw"])

    return {
        "Persons": df_persons,
        "Parties": df_parties,
        "Memberships": df_memberships,
        "Roles": df_roles,
        "Chambers": df_chambers,
        "Contests": df_contests_clean,
        "Raw_Noms": df_raw
    }

def export_popolo_json(popolo_dict) -> str:
    """Exports the in-memory Popolo DataFrames into standard structured Popolo JSON without social_network_accounts."""
    df_persons = popolo_dict["Persons"]
    df_parties = popolo_dict["Parties"]
    df_memberships = popolo_dict["Memberships"]
    df_roles = popolo_dict["Roles"]
    df_chambers = popolo_dict["Chambers"]
    df_contests = popolo_dict["Contests"]

    areas = [{
        "id": "sa_country",
        "ocd_id": "ocd-division/country:za",
        "country": "ZA",
        "state": "South Africa",
        "name": {"en_US": "South Africa"},
        "district_type": "NATIONAL",
        "parent_area_id": "",
        "city": ""
    }]

    chambers = [
        {"id": str(r["id"]), "name": {"en_US": str(r["name"])}, "area_id": str(r["area_id"])}
        for _, r in df_chambers.iterrows()
    ]

    roles = [
        {
            "id": str(r["id"]),
            "title": {"en_US": str(r["title"])},
            "area_id": str(r["area_id"]),
            "role": str(r["role"]),
            "chamber_id": str(r["chamber_id"]),
            "description": {}
        }
        for _, r in df_roles.iterrows()
    ]

    contests = [
        {
            "id": str(r["id"]),
            "title": {"en_US": str(r["name"])},
            "start_date": "2026-09-04",
            "end_date": "2026-09-04",
            "is_partisan": True,
            "role_ids": [],
            "election_identifier": str(r["election_id"])
        }
        for _, r in df_contests.iterrows()
    ]

    parties = []
    for _, r in df_parties.iterrows():
        parties.append({
            "id": str(r["id"]),
            "name": {"en_US": str(r["name"])},
            "abbreviation": [{"en_US": {"en_US": str(r["name"])}}],
            "fb_urls": r.get("fb_urls", "N/A"),
            "x_urls": r.get("x_urls", "N/A"),
            "ig_urls": r.get("ig_urls", "N/A"),
            "tiktok_urls": r.get("tiktok_urls", "N/A"),
            "yt_urls": r.get("yt_urls", "N/A"),
            "linkedin_urls": r.get("linkedin_urls", "N/A"),
            "wa_numbers": r.get("wa_numbers", "N/A"),
            "websites": r.get("websites", "N/A"),
            "colors": [],
            "logo_urls": []
        })

    persons = []
    for _, r in df_persons.iterrows():
        persons.append({
            "id": str(r["id"]),
            "full_name": {"en_US": str(r["full_name"])},
            "gender": str(r["gender"]) if pd.notna(r["gender"]) else "",
            "fb_urls": r.get("fb_urls", "N/A"),
            "x_urls": r.get("x_urls", "N/A"),
            "ig_urls": r.get("ig_urls", "N/A"),
            "tiktok_urls": r.get("tiktok_urls", "N/A"),
            "yt_urls": r.get("yt_urls", "N/A"),
            "linkedin_urls": r.get("linkedin_urls", "N/A"),
            "wa_numbers": r.get("wa_numbers", "N/A"),
            "websites": r.get("websites", "N/A"),
            "identifiers": [],
            "first_name": {"en_US": str(r["first_name"])},
            "middle_name": {"en_US": str(r["middle_name"])},
            "last_name": {"en_US": str(r["last_name"])},
            "other_names": [],
            "date_of_birth": "",
            "emails": [],
            "photo_urls": []
        })

    memberships = [
        {
            "id": str(r["id"]),
            "role_id": str(r["role_id"]),
            "person_id": str(r["person_id"]),
            "membership_type": str(r["membership_type"]),
            "start_date": str(r["start_date"]),
            "end_date": str(r["end_date"]),
            "is_partisan": bool(r["is_partisan"]),
            "has_end_date": bool(r["has_end_date"]),
            "contest_id": str(r["contest_id"]),
            "party_ids": [str(r["party_id"])],
            "source_urls": []
        }
        for _, r in df_memberships.iterrows()
    ]

    popolo_json_structure = {
        "areas": areas,
        "chambers": chambers,
        "memberships": memberships,
        "parties": parties,
        "persons": persons,
        "roles": roles,
        "contests": contests
    }

    return json.dumps(popolo_json_structure, indent=4, ensure_ascii=False)

# -----------------------------------------------------------------------------
# 3. API CLIENT RESOLUTION & SOCIAL SEARCH ENGINE
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

def clean_social_value(val):
    """Ensures missing, empty, or None values default strictly to N/A."""
    if not val or str(val).strip() in ["", "none", "null", "None", "Null"]:
        return "N/A"
    return str(val).strip()

def search_person_socials(p_client, person_name, party_name=""):
    """Uses Perplexity to discover official social profiles for candidate persons."""
    prompt = f"""
    Find official public social media URLs, WhatsApp contact, and website for South African candidate '{person_name}' (Party: {party_name}).
    Respond ONLY with a raw valid JSON object with no markdown codeblocks:
    {{
      "fb_urls": "Facebook URL or N/A",
      "x_urls": "X/Twitter URL or N/A",
      "ig_urls": "Instagram URL or N/A",
      "tiktok_urls": "TikTok URL or N/A",
      "yt_urls": "YouTube URL or N/A",
      "linkedin_urls": "LinkedIn URL or N/A",
      "wa_numbers": "WhatsApp Number or N/A",
      "websites": "Website URL or N/A"
    }}
    If any link or number is not found, use "N/A".
    """
    default_na = {
        "fb_urls": "N/A", "x_urls": "N/A", "ig_urls": "N/A",
        "tiktok_urls": "N/A", "yt_urls": "N/A", "linkedin_urls": "N/A",
        "wa_numbers": "N/A", "websites": "N/A"
    }
    try:
        res = p_client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "system", "content": "You are a precise political OSINT researcher. Output valid JSON only with 'N/A' for missing fields."},
                {"role": "user", "content": prompt}
            ]
        )
        content = res.choices[0].message.content.strip()
        content = re.sub(r"^```json|```$", "", content).strip()
        parsed = json.loads(content)
        return {k: clean_social_value(parsed.get(k)) for k in default_na.keys()}
    except Exception:
        return default_na

def search_party_socials(p_client, party_name):
    """Uses Perplexity to discover official social media profiles and websites for political parties."""
    prompt = f"""
    Find official social media URLs, WhatsApp contact, and website for South African political party '{party_name}'.
    Respond ONLY with a raw valid JSON object with no markdown codeblocks:
    {{
      "fb_urls": "Facebook URL or N/A",
      "x_urls": "X/Twitter URL or N/A",
      "ig_urls": "Instagram URL or N/A",
      "tiktok_urls": "TikTok URL or N/A",
      "yt_urls": "YouTube URL or N/A",
      "linkedin_urls": "LinkedIn URL or N/A",
      "wa_numbers": "WhatsApp Number or N/A",
      "websites": "Website URL or N/A"
    }}
    If any link or number is not found, use "N/A".
    """
    default_na = {
        "fb_urls": "N/A", "x_urls": "N/A", "ig_urls": "N/A",
        "tiktok_urls": "N/A", "yt_urls": "N/A", "linkedin_urls": "N/A",
        "wa_numbers": "N/A", "websites": "N/A"
    }
    try:
        res = p_client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "system", "content": "You are a precise political OSINT researcher. Output valid JSON only with 'N/A' for missing fields."},
                {"role": "user", "content": prompt}
            ]
        )
        content = res.choices[0].message.content.strip()
        content = re.sub(r"^```json|```$", "", content).strip()
        parsed = json.loads(content)
        return {k: clean_social_value(parsed.get(k)) for k in default_na.keys()}
    except Exception:
        return default_na

# -----------------------------------------------------------------------------
# 4. SIDEBAR & NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🌍 PEP Intelligence")
st.sidebar.caption("South Africa LGE Candidate Engine")
st.sidebar.markdown("---")

nav_options = {
    "ingest": "📥 Data Ingestion & Parser",
    "popolo": "🗂️ Popolo Standard Data ",
    "perplexity": "🌐 Social Media & PEP Search (Perplexity)",
    "groq": "⚡ Groq AI Summarizer"
}

selected_label = st.sidebar.radio("Go to", options=list(nav_options.values()))
view_key = [k for k, v in nav_options.items() if v == selected_label][0]

st.sidebar.markdown("---")
st.sidebar.markdown("##### ⚙️ API Diagnostics")

p_client = get_perplexity_client()
g_client = get_groq_client()

if p_client:
    st.sidebar.success("⚡ Perplexity API Active")
else:
    st.sidebar.warning("🔑 Perplexity Key Missing")

if g_client:
    st.sidebar.success("⚡ Groq API Active")
else:
    st.sidebar.warning("🔑 Groq Key Missing")

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

        st.markdown("### 📊 Dataset Overview")
        m1, m2 = st.columns(2)
        m1.metric("Total Candidacy Records", f"{len(df_memberships):,}")
        m2.metric("Unique Individual Candidates", f"{len(df_persons):,}")

        st.markdown("---")
        st.markdown("### 🧹 Persons Table Preview")
        st.dataframe(df_persons.head(15), use_container_width=True)

        if st.button("🗑️ Clear & Reset Dataset"):
            st.session_state["popolo_tables"] = None
            st.rerun()

# -----------------------------------------------------------------------------
# VIEW 2: POPOLO STANDARD DATA (6 TABS + JSON EXPORT)
# -----------------------------------------------------------------------------
elif view_key == "popolo":
    if st.session_state.get("popolo_tables") is not None:
        pop = st.session_state["popolo_tables"]
        popolo_json_data = export_popolo_json(pop)
        
        st.markdown("### 📦 Master Export Options")
        st.download_button(
            label="📥 Download Complete Dataset (Full Popolo JSON Format)",
            data=popolo_json_data,
            file_name="Master_Popolo_Dataset_2026.json",
            mime="application/json",
            key="dl_btn_master_json"
        )
        
        st.markdown("---")

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
# VIEW 3: PERPLEXITY SEARCH & SOCIAL ENRICHMENT API
# -----------------------------------------------------------------------------
elif view_key == "perplexity":
    st.markdown("### 🌐 Real-Time Political & Social Media Search")
    st.write("Search for social media channels (`fb_urls`, `x_urls`, `ig_urls`, `tiktok_urls`, `yt_urls`, `linkedin_urls`, `wa_numbers`, `websites`). Missing values automatically default to `N/A`.")

    if not p_client:
        st.error("⚠️ Perplexity API Key is not configured in `st.secrets` or environment variables.")
    else:
        st.markdown("---")
        st.markdown("#### 🔗 Automated Social Media Link Finder")

        if st.session_state.get("popolo_tables") is not None:
            pop = st.session_state["popolo_tables"]
            enrich_type = st.radio("Select Table to Enrich Social Links:", ["Persons (Candidates)", "Parties (Political Parties)"], horizontal=True)

            if enrich_type == "Persons (Candidates)":
                df_pers = pop["Persons"]
                max_cand = st.slider("Select Max Candidates to Search:", 1, len(df_pers), min(10, len(df_pers)))

                if st.button("🔍 Search & Enrich Candidates Social Links"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx in range(max_cand):
                        person_name = df_pers.at[idx, "full_name"]
                        status_text.text(f"Searching social profiles for: {person_name} ({idx+1}/{max_cand})")

                        socials = search_person_socials(p_client, person_name)

                        for col_key in ["fb_urls", "x_urls", "ig_urls", "tiktok_urls", "yt_urls", "linkedin_urls", "wa_numbers", "websites"]:
                            df_pers.at[idx, col_key] = socials.get(col_key, "N/A")

                        progress_bar.progress((idx + 1) / max_cand)

                    st.session_state["popolo_tables"]["Persons"] = df_pers
                    st.success("✅ Persons table updated with discovered social media links!")
                    st.dataframe(df_pers.head(max_cand), use_container_width=True)

            else:
                df_part = pop["Parties"]
                max_party = st.slider("Select Max Parties to Search:", 1, len(df_part), min(10, len(df_part)))

                if st.button("🔍 Search & Enrich Parties Social Links"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx in range(max_party):
                        party_name = df_part.at[idx, "name"]
                        status_text.text(f"Searching social profiles for party: {party_name} ({idx+1}/{max_party})")

                        socials = search_party_socials(p_client, party_name)

                        for col_key in ["fb_urls", "x_urls", "ig_urls", "tiktok_urls", "yt_urls", "linkedin_urls", "wa_numbers", "websites"]:
                            df_part.at[idx, col_key] = socials.get(col_key, "N/A")

                        progress_bar.progress((idx + 1) / max_party)

                    st.session_state["popolo_tables"]["Parties"] = df_part
                    st.success("✅ Parties table updated with discovered social media links!")
                    st.dataframe(df_part.head(max_party), use_container_width=True)

        else:
            st.info("💡 Upload PDF data first in the Data Ingestion tab to run social media enrichment.")

# -----------------------------------------------------------------------------
# VIEW 4: GROQ AI SUMMARIZER
# -----------------------------------------------------------------------------
elif view_key == "groq":
    st.markdown("### ⚡ High-Speed AI Dataset Summarizer (Groq)")
    st.write("Generate automated summaries and analytical reports using Qwen models on Groq.")

    if not g_client:
        st.error("⚠️ Groq API Key is not configured in `st.secrets` or environment variables.")
    else:
        model_choice = st.selectbox(
            "Select Qwen Model:",
            ["qwen/qwen3.6-27b", "qwen/qwen3.8-27b", "qwen-qwq-32b", "llama-3.3-70b-versatile"]
        )
        
        has_data = st.session_state.get("popolo_tables") is not None
        prompt_option = st.selectbox("Analysis Goal:", [
            "Summarize Dataset Candidate & Party Distribution",
            "Identify Potential High-Risk PEPs or Key Parties",
            "Custom Query"
        ])

        custom_prompt = st.text_area("Enter Custom Prompt:") if prompt_option == "Custom Query" else ""

        if st.button("⚡ Generate AI Summary"):
            with st.spinner(f"Analyzing dataset with {model_choice}..."):
                try:
                    if has_data:
                        pop = st.session_state["popolo_tables"]
                        context = f"Parties Sample:\n{pop['Parties'].head(10).to_string()}\n\nPersons Sample:\n{pop['Persons'].head(10).to_string()}"
                    else:
                        context = "No specific dataset uploaded in memory."

                    full_user_prompt = f"{custom_prompt if prompt_option == 'Custom Query' else prompt_option}\n\nData Context:\n{context}"

                    chat_completion = g_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are an executive political analyst expert in South African candidate data."},
                            {"role": "user", "content": full_user_prompt}
                        ],
                        model=model_choice,
                    )
                    st.markdown(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Execution Error: {e}")
