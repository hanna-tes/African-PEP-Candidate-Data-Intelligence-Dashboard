import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import pypdf
import os
from datetime import datetime

# Import official Perplexity SDK safely
try:
    from perplexity import Perplexity, APIStatusError, APIConnectionError
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & EXECUTIVE DARK THEME (CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PEP & Candidate Data Intelligence Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Executive UI
st.markdown("""
    <style>
        /* Main background and base font */
        .stApp {
            background: linear-gradient(135deg, #090d16 0%, #0f172a 100%);
            color: #f1f5f9;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 95%;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0b1120 !important;
            border-right: 1px solid #1e293b;
        }

        /* Executive Header Container */
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

        .header-title {
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
            letter-spacing: -0.5px;
        }

        .header-badge {
            background: rgba(56, 189, 248, 0.1);
            color: #38bdf8;
            border: 1px solid #0284c7;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }

        /* Cards & Content Containers */
        .ui-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        
        .ui-card:hover {
            border-color: #38bdf8;
        }

        /* Metric Tiles */
        div[data-testid="stMetricValue"] {
            color: #38bdf8 !important;
            font-weight: 700;
            font-size: 26px;
        }
        
        .stMetric {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 14px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }

        /* Buttons Styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        /* Custom Badges */
        .badge-verified { background-color: #064e3b; color: #34d399; border: 1px solid #059669; padding: 2px 8px; border-radius: 6px; font-size: 11px; }
        .badge-pending { background-color: #451a03; color: #fbbf24; border: 1px solid #d97706; padding: 2px 8px; border-radius: 6px; font-size: 11px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. ALL 54 AFRICAN NATIONS & JURISDICTION TAXONOMY
# -----------------------------------------------------------------------------
EXPLICIT_TIERS = {
    "Ethiopia 🇪🇹": {
        "code": "ET",
        "tiers": [
            "House of Peoples' Representatives (HOPR)",
            "Regional State Council",
            "Executive Members / Administration"
        ]
    },
    "South Africa 🇿🇦": {
        "code": "ZA",
        "tiers": [
            "National Assembly (National List)",
            "National Assembly (Regional List)",
            "Provincial Legislature",
            "Local Government / Municipal Council"
        ]
    },
    "Kenya 🇰🇪": {
        "code": "KE",
        "tiers": [
            "Presidential Candidate",
            "Senatorial Candidate",
            "Member of National Assembly (MP)",
            "County Woman Representative",
            "Member of County Assembly (MCA)"
        ]
    }
}

COMMON_NATIONAL_TIERS = [
    "National / Parliamentary Candidate",
    "Provincial / Regional Council",
    "Local Government / Municipal",
    "Executive Cabinet / State Official"
]

ALL_AFRICAN_COUNTRIES = [
    "Algeria 🇩🇿", "Angola 🇦🇴", "Benin 🇧🇯", "Botswana 🇧🇼", "Burkina Faso 🇧🇫",
    "Burundi 🇧🇮", "Cabo Verde 🇨🇻", "Cameroon 🇨🇲", "Central African Republic 🇨🇫",
    "Chad 🇹🇩", "Comoros 🇰🇲", "Congo 🇨🇬", "DR Congo 🇨🇩", "Djibouti 🇩🇯",
    "Egypt 🇪🇬", "Equatorial Guinea 🇬🇶", "Eritrea 🇪🇷", "Eswatini 🇸🇿",
    "Ethiopia 🇪🇹", "Gabon 🇬🇦", "Gambia 🇬🇲", "Ghana 🇬🇭", "Guinea 🇬🇳",
    "Guinea-Bissau 🇬🇼", "Ivory Coast 🇨🇮", "Kenya 🇰🇪", "Lesotho 🇱🇸",
    "Liberia 🇱🇷", "Libya 🇱🇾", "Madagascar 🇲🇬", "Malawi 🇲🇼", "Mali 🇲🇱",
    "Mauritania 🇲🇷", "Mauritius 🇲🇺", "Morocco 🇲🇦", "Mozambique 🇲🇿",
    "Namibia 🇳🇦", "Niger 🇳🇪", "Nigeria 🇳🇬", "Rwanda 🇷🇼", "Sao Tome & Principe 🇸🇹",
    "Senegal 🇸🇳", "Seychelles 🇸🇨", "Sierra Leone 🇸🇱", "Somalia 🇸🇴",
    "South Africa 🇿🇦", "South Sudan 🇸🇸", "Sudan 🇸🇩", "Tanzania 🇹🇿",
    "Togo 🇹🇬", "Tunisia 🇹🇳", "Uganda 🇺🇬", "Zambia 🇿🇲", "Zimbabwe 🇿🇼"
]

SOCIAL_PLATFORMS = [
    {"name": "Facebook", "url_col": "fb_url", "note_col": "note_fb"},
    {"name": "Instagram", "url_col": "instagram_url", "note_col": "note_ig"},
    {"name": "Twitter / X", "url_col": "twitter_url", "note_col": "note_tw"},
    {"name": "TikTok", "url_col": "tiktok_url", "note_col": "note_tk"},
    {"name": "YouTube", "url_col": "YouTube_url", "note_col": "note_yt"},
    {"name": "LinkedIn", "url_col": "LinkedIn_url", "note_col": "note_li"},
    {"name": "Website", "url_col": "website", "note_col": "note_web"},
    {"name": "Wikipedia", "url_col": "wikipedia_url", "note_col": "note_wiki"},
    {"name": "WhatsApp", "url_col": "wa_numbers", "note_col": "note_wa"},
    {"name": "Telegram", "url_col": "telegram", "note_col": "note_te"}
]

# -----------------------------------------------------------------------------
# 3. PERPLEXITY SEARCH API INTEGRATION ENGINE
# -----------------------------------------------------------------------------
def get_perplexity_client():
    """Resolves API key from st.secrets or os.environ safely."""
    api_key = st.secrets.get("PERPLEXITY_API_KEY", os.environ.get("PERPLEXITY_API_KEY", ""))
    if not api_key:
        return None
    try:
        return Perplexity(api_key=api_key)
    except Exception:
        return None

def run_perplexity_entity_search(query_list: list, country_name: str, max_results: int = 5):
    """Executes POST /search using Perplexity SDK with multi-query support & URL deduplication."""
    client = get_perplexity_client()
    if not client:
        return None, "PERPLEXITY_API_KEY is missing. Please set it in Streamlit Secrets or terminal environment variables."

    try:
        iso_code = EXPLICIT_TIERS.get(country_name, {}).get("code", None)
        search_response = client.search.create(
            query=query_list[:5],
            max_results=max_results,
            country=iso_code.lower() if iso_code else None
        )
        
        raw_results = getattr(search_response, "results", [])
        seen_urls = set()
        deduped_results = []
        
        for item in raw_results:
            url = getattr(item, "url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped_results.append({
                    "title": getattr(item, "title", "Untitled"),
                    "url": url,
                    "snippet": getattr(item, "snippet", ""),
                    "date": getattr(item, "date", None)
                })
        return deduped_results, None

    except APIStatusError as e:
        if e.status_code == 401:
            return None, "401 Authentication Error: Invalid PERPLEXITY_API_KEY."
        elif e.status_code == 429:
            return None, "429 Rate Limit Exceeded: Honoring retry-after period."
        else:
            return None, f"Perplexity API Error [{e.status_code}]: {str(e)}"
    except APIConnectionError:
        return None, "Connection error: Failed to reach api.perplexity.ai."
    except Exception as e:
        return None, f"Search failed: {str(e)}"

# -----------------------------------------------------------------------------
# 4. PARSER & POPOLO SCHEMA BUILDER
# -----------------------------------------------------------------------------
def sanitize_name(raw_name: str) -> str:
    if not raw_name or pd.isna(raw_name):
        return ""
    text = re.sub(r"(?i)\s*Page\s+\d+\s+of\s+\d+.*", "", str(raw_name))
    text = re.sub(r"\s+\d+$", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def parse_name(full_name: str):
    clean_str = sanitize_name(full_name)
    parts = clean_str.split()
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], "", parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return parts[0], " ".join(parts[1:-1]), parts[-1]

def build_popolo_tables(raw_df: pd.DataFrame, country: str, tier: str):
    raw_df["sanitized_name"] = raw_df["full_name"].apply(sanitize_name)
    df_persons = raw_df[["sanitized_name", "party_name", "constituency"]].drop_duplicates().reset_index(drop=True)
    
    df_persons["id"] = [f"pers_{i+1:05d}" for i in range(len(df_persons))]
    df_persons["name_id"] = [f"name_{i+1:05d}" for i in range(len(df_persons))]
    df_persons["full_name"] = df_persons["sanitized_name"]
    
    splits = df_persons["full_name"].apply(parse_name)
    df_persons["first_name"] = [s[0] for s in splits]
    df_persons["middle_name"] = [s[1] for s in splits]
    df_persons["last_name"] = [s[2] for s in splits]
    df_persons["gender"] = "Unspecified"
    
    df_persons["party_id"] = [f"pty_{abs(hash(p)) % 1000:03d}" for p in df_persons["party_name"]]
    df_persons["abbrv"] = df_persons["party_name"].apply(lambda x: "".join([w[0] for w in str(x).split()]).upper()[:4])
    df_persons["office"] = tier
    df_persons["municipality"] = df_persons["constituency"]
    df_persons["ward_pr_order"] = "PR"
    df_persons["researcher"] = "Auto-Parser Pipeline"

    for plat in SOCIAL_PLATFORMS:
        df_persons[plat["note_col"]] = "Pending Mapping"
        df_persons[plat["url_col"]] = "N/A"

    df_parties_tab = df_persons[["party_id", "party_name", "abbrv"]].drop_duplicates().reset_index(drop=True)
    df_parties_tab["country"] = country
    df_parties_tab["official_website"] = "N/A"
    df_parties_tab["twitter_url"] = "N/A"
    df_parties_tab["facebook_url"] = "N/A"
    df_parties_tab["mapping_status"] = "Pending Review"

    return {
        "Persons": df_persons,
        "Parties": df_parties_tab,
        "Memberships": df_persons[["id", "party_id", "party_name", "office", "municipality"]].copy(),
        "Roles": pd.DataFrame([{"role_id": "role_cand", "role_title": f"Candidate - {tier}", "jurisdiction": country}]),
        "Chambers": pd.DataFrame([{"chamber_id": f"ch_{tier[:3].lower()}", "chamber_name": tier, "country": country}]),
        "Contests": df_persons[["municipality"]].drop_duplicates().rename(columns={"municipality": "electoral_district"})
    }

# -----------------------------------------------------------------------------
# 5. STREAMLINED SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🌍 PEP Intelligence")
st.sidebar.caption("African Candidate & Political Data Hub")
st.sidebar.markdown("---")

# Section 1: Country & Scope Controls
st.sidebar.markdown("##### 📍 Target Jurisdiction")
selected_country = st.sidebar.selectbox("Country", ALL_AFRICAN_COUNTRIES, index=45) # Default to South Africa

if selected_country in EXPLICIT_TIERS:
    available_tiers = EXPLICIT_TIERS[selected_country]["tiers"]
else:
    available_tiers = COMMON_NATIONAL_TIERS

selected_tier = st.sidebar.selectbox("Scope Tier / Office", available_tiers)

st.sidebar.markdown("---")

# Section 2: Main Module Navigation
st.sidebar.markdown("##### 🧭 Module Navigation")
view_selection = st.sidebar.radio(
    "Go to",
    [
        "📥 Data Ingestion & Parser",
        "🗂️ Popolo Standard Data (6 Tabs)",
        "🌐 Perplexity Entity Search Engine"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Section 3: Live API Status Tracker
st.sidebar.markdown("##### ⚙️ System Diagnostics")
perplexity_client_check = get_perplexity_client()
if perplexity_client_check:
    st.sidebar.success("⚡ Perplexity API: Active")
else:
    st.sidebar.warning("🔑 Perplexity API: Key Missing")

# -----------------------------------------------------------------------------
# 6. MAIN WORKSPACE CONTENT
# -----------------------------------------------------------------------------

# Dynamic Header Bar
st.markdown(f"""
    <div class="header-card">
        <div>
            <h1 class="header-title">{view_selection}</h1>
            <span style="color: #94a3b8; font-size: 13px;">African Candidate Data Intelligence Dashboard</span>
        </div>
        <div>
            <span class="header-badge">{selected_country} • {selected_tier}</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# MODULE 1: INGESTION
if view_selection == "📥 Data Ingestion & Parser":
    st.markdown("""
        <div class="ui-card">
            <h4 style="margin-top:0;">📄 Upload Candidate Lists</h4>
            <p style="color: #94a3b8; font-size: 14px;">Select official election PDFs or CSV files to extract, sanitize, and structure candidates automatically.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("", type=["pdf", "csv"], accept_multiple_files=True)

    if uploaded_files:
        raw_records = []
        id_pattern = re.compile(r"([0-9]{6}\*\*\*\*[0-9]{2}\*)")
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if not text: continue
                    for line in text.split("\n"):
                        match = id_pattern.search(line)
                        if match:
                            before = line[:match.start()].strip()
                            after = line[match.end():].strip()
                            raw_records.append({"constituency": before if before else "Central Region", "party_name": "Extracted Political Party", "full_name": after})
            elif file.name.endswith(".csv"):
                csv_df = pd.read_csv(file)
                for _, r in csv_df.iterrows():
                    raw_records.append({"constituency": r.get("municipality", r.get("constituency", "District 1")), "party_name": r.get("party_name", "Independent"), "full_name": r.get("full_name", "")})

        if raw_records:
            df_raw = pd.DataFrame(raw_records)
            st.session_state["popolo_tables"] = build_popolo_tables(df_raw, selected_country, selected_tier)
            
            # Quick Stats Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Parsed Candidates", f"{len(df_raw):,}")
            c2.metric("Unique Parties", len(st.session_state["popolo_tables"]["Parties"]))
            c3.metric("Electoral Districts", len(st.session_state["popolo_tables"]["Contests"]))
            
            st.markdown("### 🧹 Persons Schema Preview")
            st.dataframe(st.session_state["popolo_tables"]["Persons"].head(10), use_container_width=True)

# MODULE 2: POPOLO 6 TABS
elif view_selection == "🗂️ Popolo Standard Data (6 Tabs)":
    if "popolo_tables" in st.session_state:
        pop = st.session_state["popolo_tables"]
        
        tabs = st.tabs([
            "1. Persons", "2. Parties", "3. Memberships", 
            "4. Roles", "5. Chambers", "6. Contests"
        ])
        tab_names = ["Persons", "Parties", "Memberships", "Roles", "Chambers", "Contests"]

        for idx, tab in enumerate(tabs):
            with tab:
                key = tab_names[idx]
                curr_df = pop[key]
                
                st.dataframe(curr_df, use_container_width=True)
                
                csv_bytes = curr_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download {key} CSV Data",
                    data=csv_bytes,
                    file_name=f"{selected_country}_{selected_tier}_{key}.csv",
                    mime="text/csv",
                    type="secondary"
                )
    else:
        st.info("💡 Please upload candidate lists in the **Data Ingestion & Parser** module to generate Popolo standard tables.")

# MODULE 3: PERPLEXITY SEARCH
elif view_selection == "🌐 Perplexity Entity Search Engine":
    if "popolo_tables" in st.session_state:
        df_persons = st.session_state["popolo_tables"]["Persons"]
        df_parties = st.session_state["popolo_tables"]["Parties"]

        target_tab = st.radio("Search Focus", ["👤 Candidate Search", "🏛️ Party Search"], horizontal=True)

        if target_tab == "👤 Candidate Search":
            st.markdown("##### Candidate Social Media Handle Discovery")
            
            selected_cand_idx = st.selectbox(
                "Select Candidate", 
                df_persons.index, 
                format_func=lambda i: f"{df_persons.at[i, 'full_name']} — {df_persons.at[i, 'party_name']}"
            )
            
            cand_row = df_persons.loc[selected_cand_idx]
            
            queries = [
                f'"{cand_row["full_name"]}" "{cand_row["party_name"]}" twitter OR x.com OR linkedin',
                f'"{cand_row["full_name"]}" candidate {selected_country} facebook'
            ]

            if st.button("🔎 Run Web Search via Perplexity", type="primary"):
                with st.spinner("Searching live web indexes..."):
                    results, err = run_perplexity_entity_search(queries, country_name=selected_country)
                    if err:
                        st.error(err)
                    else:
                        st.markdown(f"**Found {len(results)} Ranked Results:**")
                        for res in results:
                            with st.expander(f"🌐 {res['title']}"):
                                st.write(f"**URL:** [{res['url']}]({res['url']})")
                                st.write(f"**Snippet:** {res['snippet']}")
                                if st.button(f"Attach Link to Candidate", key=f"att_{res['url']}"):
                                    st.session_state["popolo_tables"]["Persons"].at[selected_cand_idx, "website"] = res["url"]
                                    st.session_state["popolo_tables"]["Persons"].at[selected_cand_idx, "note_web"] = "Verified via Perplexity Search API"
                                    st.success(f"Attached link to {cand_row['full_name']}!")

        elif target_tab == "🏛️ Party Search":
            st.markdown("##### Political Party Domain Discovery")
            
            selected_party_idx = st.selectbox(
                "Select Political Party", 
                df_parties.index, 
                format_func=lambda i: f"{df_parties.at[i, 'party_name']} ({df_parties.at[i, 'abbrv']})"
            )
            
            party_row = df_parties.loc[selected_party_idx]
            p_queries = [f'"{party_row["party_name"]}" official website {selected_country}']

            if st.button("🔎 Search Party Web Profiles", type="primary"):
                with st.spinner("Executing Perplexity Search API..."):
                    p_results, p_err = run_perplexity_entity_search(p_queries, country_name=selected_country)
                    if p_err:
                        st.error(p_err)
                    else:
                        for p_res in p_results:
                            st.write(f"👉 **[{p_res['title']}]({p_res['url']})**")
                            st.caption(p_res['snippet'])
    else:
        st.info("💡 Upload data in the **Data Ingestion & Parser** module to enable web search verification.")
