import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import pypdf
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & DARK MODERN UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="African PEP & Candidate Data Intelligence Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #0b0e14; color: #e2e8f0; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700; font-size: 24px; }
        .stMetric { background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 12px; }
        
        .entity-card {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .status-badge {
            font-size: 11px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 12px;
        }
        .badge-approved { background-color: #052e16; color: #4ade80; border: 1px solid #166534; }
        .badge-pending { background-color: #451a03; color: #fbbf24; border: 1px solid #854d0e; }
        .badge-na { background-color: #1f2937; color: #9ca3af; border: 1px solid #374151; }
        .badge-excluded { background-color: #450a0a; color: #f87171; border: 1px solid #991b1b; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. AFRICAN TAXONOMY & ALL 10 SOCIAL PLATFORMS LIST
# -----------------------------------------------------------------------------
AFRICAN_JURISDICTION_TAXONOMY = {
    "Ethiopia 🇪🇹": [
        "House of Peoples' Representatives (HOPR)",
        "Regional State Council",
        "Executive Members / Administration"
    ],
    "South Africa 🇿🇦": [
        "National Assembly (National List)",
        "National Assembly (Regional List)",
        "Provincial Legislature",
        "Local Government / Municipal Council"
    ],
    "Kenya 🇰🇪": [
        "Presidential Candidate",
        "Senatorial Candidate",
        "Member of National Assembly (MP)",
        "County Woman Representative",
        "Member of County Assembly (MCA)"
    ],
    "Other African Country 🌍": [
        "National / Parliamentary",
        "Provincial / Regional",
        "Local Government / Municipal",
        "Executive Council"
    ]
}

# Standardized 10 Platforms mapping list aligned with reference schema
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
# 3. AUTOMATED SANITATION & POPOLO SCHEMA BUILDER
# -----------------------------------------------------------------------------
def sanitize_name(raw_name: str) -> str:
    """Strips PDF artifacts (e.g. 'Page 129 of 228'), trailing numbers, and extra spaces."""
    if not raw_name or pd.isna(raw_name):
        return ""
    text = re.sub(r"(?i)\s*Page\s+\d+\s+of\s+\d+.*", "", str(raw_name))
    text = re.sub(r"\s+\d+$", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def parse_name(full_name: str):
    """Parses First, Middle, and Last name components without order corruption."""
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
    """
    Transforms raw uploaded candidate data into full 6-Tab Popolo CSV tables,
    strictly matching the target dataset columns for Persons and Parties.
    """
    raw_df["sanitized_name"] = raw_df["full_name"].apply(sanitize_name)
    
    # 1. PERSONS TAB (Matches uploaded standard schema)
    df_persons = raw_df[["sanitized_name", "party_name", "constituency"]].drop_duplicates().reset_index(drop=True)
    
    df_persons["id"] = [f"pers_{i+1}" for i in range(len(df_persons))]
    df_persons["name_id"] = [f"name_{i+1}" for i in range(len(df_persons))]
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

    # Add all 10 Social Media URLs & Note columns
    for plat in SOCIAL_PLATFORMS:
        df_persons[plat["note_col"]] = "Pending Mapping"
        df_persons[plat["url_col"]] = "N/A"

    persons_col_order = [
        "id", "name_id", "full_name", "first_name", "middle_name", "last_name", "gender",
        "party_id", "party_name", "abbrv", "office", "municipality", "ward_pr_order", "researcher",
        "note_fb", "fb_url", "note_ig", "instagram_url", "note_tw", "twitter_url", 
        "note_tk", "tiktok_url", "note_yt", "YouTube_url", "note_li", "LinkedIn_url", 
        "note_web", "website", "note_wiki", "wikipedia_url", "note_wa", "wa_numbers", 
        "note_te", "telegram"
    ]
    df_persons_tab = df_persons[persons_col_order]

    # 2. PARTIES TAB (Unique Political Parties + Social Profile Tracking)
    df_parties_tab = df_persons[["party_id", "party_name", "abbrv"]].drop_duplicates().reset_index(drop=True)
    df_parties_tab["country"] = country
    df_parties_tab["official_website"] = "N/A"
    df_parties_tab["twitter_url"] = "N/A"
    df_parties_tab["facebook_url"] = "N/A"
    df_parties_tab["mapping_status"] = "Pending Review"

    # 3. MEMBERSHIPS TAB
    df_mshp_tab = df_persons[["id", "party_id", "party_name", "office", "municipality"]].copy()
    df_mshp_tab.rename(columns={"id": "person_id"}, inplace=True)
    df_mshp_tab["membership_id"] = [f"mshp_{pid}" for pid in df_mshp_tab["person_id"]]
    df_mshp_tab["membership_type"] = "certified_candidate"

    # 4. ROLES TAB
    df_roles_tab = pd.DataFrame([{"role_id": "role_cand", "role_title": f"Candidate - {tier}", "jurisdiction": country}])

    # 5. CHAMBERS TAB
    df_chambers_tab = pd.DataFrame([{"chamber_id": f"ch_{tier[:3].lower()}", "chamber_name": tier, "country": country}])

    # 6. CONTESTS TAB
    df_contests_tab = df_persons[["municipality"]].drop_duplicates().rename(columns={"municipality": "electoral_district"})
    df_contests_tab["country"] = country
    df_contests_tab["election_tier"] = tier

    return {
        "Persons": df_persons_tab,
        "Parties": df_parties_tab,
        "Memberships": df_mshp_tab,
        "Roles": df_roles_tab,
        "Chambers": df_chambers_tab,
        "Contests": df_contests_tab
    }

def mock_persons_social_mappings(df_persons):
    """Simulates multi-platform social link discovery across Persons dataset."""
    df_out = df_persons.copy()
    for idx, row in df_out.iterrows():
        rand_val = np.random.rand()
        if rand_val < 0.65:
            plat = np.random.choice(SOCIAL_PLATFORMS)
            conf = int(np.random.uniform(40, 98))
            fname = str(row['first_name']).lower()
            lname = str(row['last_name']).lower()
            
            df_out.at[idx, plat["url_col"]] = f"https://{plat['name'].split()[0].lower()}.com/{fname}_{lname}"
            df_out.at[idx, plat["note_col"]] = f"Auto-Matched ({conf}% Confidence)"
    return df_out

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🌍 African Candidate Intelligence")
st.sidebar.markdown("---")

selected_country = st.sidebar.selectbox("📍 Jurisdiction", list(AFRICAN_JURISDICTION_TAXONOMY.keys()))
selected_tier = st.sidebar.selectbox("🏛️ Election Tier / Scope", AFRICAN_JURISDICTION_TAXONOMY[selected_country])

st.sidebar.markdown("---")
view_selection = st.sidebar.radio("Platform Section", [
    "📥 1. Candidate List Ingestion & Parsing",
    "🗂️ 2. Popolo Standard Data (6 Tabs)",
    "🌐 3. Social Media Mapping & Review (Persons & Parties)"
])

# -----------------------------------------------------------------------------
# 5. APPLICATION VIEWS
# -----------------------------------------------------------------------------

# --- VIEW 1: INGESTION ---
if view_selection == "📥 1. Candidate List Ingestion & Parsing":
    st.title("📥 Candidate List Ingestion Engine")
    st.markdown(f"Selected Scope: **{selected_country}** ➔ **{selected_tier}**")

    uploaded_files = st.file_uploader("Upload Official PDF/CSV Lists", type=["pdf", "csv"], accept_multiple_files=True)

    if uploaded_files:
        raw_records = []
        id_pattern = re.compile(r"([0-9]{6}\*\*\*\*[0-9]{2}\*)")

        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    for line in text.split("\n"):
                        match = id_pattern.search(line)
                        if match:
                            before = line[:match.start()].strip()
                            after = line[match.end():].strip()
                            raw_records.append({
                                "constituency": before if before else "Central Region",
                                "party_name": "Extracted Political Party",
                                "full_name": after
                            })
            elif file.name.endswith(".csv"):
                csv_df = pd.read_csv(file)
                for _, r in csv_df.iterrows():
                    raw_records.append({
                        "constituency": r.get("municipality", r.get("constituency", "District 1")),
                        "party_name": r.get("party_name", "Independent"),
                        "full_name": r.get("full_name", "")
                    })

        if raw_records:
            df_raw = pd.DataFrame(raw_records)
            st.success(f"Successfully processed files! Extracted **{len(df_raw):,}** candidates.")
            
            popolo_tables = build_popolo_tables(df_raw, selected_country, selected_tier)
            popolo_tables["Persons"] = mock_persons_social_mappings(popolo_tables["Persons"])
            
            st.session_state["popolo_tables"] = popolo_tables

            st.markdown("### 🧹 Cleaned Persons Output Preview")
            st.dataframe(popolo_tables["Persons"].head(10), use_container_width=True)

# --- VIEW 2: POPOLO 6 TABS ---
elif view_selection == "🗂️ 2. Popolo Standard Data (6 Tabs)":
    st.title("🗂️ Popolo Standard Data Model")
    st.markdown(f"Active Context: **{selected_country}** | Scope: **{selected_tier}**")

    if "popolo_tables" in st.session_state:
        pop = st.session_state["popolo_tables"]
        tabs = st.tabs(["1. Persons", "2. Parties", "3. Memberships", "4. Roles", "5. Chambers", "6. Contests"])
        tab_names = ["Persons", "Parties", "Memberships", "Roles", "Chambers", "Contests"]

        for idx, tab in enumerate(tabs):
            with tab:
                key = tab_names[idx]
                df_curr = pop[key]
                st.dataframe(df_curr, use_container_width=True)

                csv_bytes = df_curr.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download {key} CSV",
                    data=csv_bytes,
                    file_name=f"{selected_country}_{selected_tier}_{key}.csv",
                    mime="text/csv"
                )
    else:
        st.warning("Please upload candidate data in the Ingestion tab first.")

# --- VIEW 3: SOCIAL MEDIA MAPPING & HUMAN REVIEW ---
elif view_selection == "🌐 3. Social Media Mapping & Review (Persons & Parties)":
    st.title("🌐 Social Media Mapping & Human Review Portal")

    if "popolo_tables" in st.session_state:
        df_persons = st.session_state["popolo_tables"]["Persons"]
        df_parties = st.session_state["popolo_tables"]["Parties"]

        target_tab = st.radio("Select Review Domain", ["👤 Candidate Persons Mapping", "🏛️ Political Parties Mapping"], horizontal=True)

        if target_tab == "👤 Candidate Persons Mapping":
            st.subheader("Candidate Social Media Coverage Analytics")

            # Metrics
            total_cand = len(df_persons)
            has_tw = len(df_persons[df_persons["twitter_url"] != "N/A"])
            has_fb = len(df_persons[df_persons["fb_url"] != "N/A"])
            has_li = len(df_persons[df_persons["LinkedIn_url"] != "N/A"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Candidates", total_cand)
            c2.metric("Twitter / X Mapped", has_tw)
            c3.metric("Facebook Mapped", has_fb)
            c4.metric("LinkedIn Mapped", has_li)

            st.markdown("---")
            st.subheader("🖐️ Candidate Account Verification Workflow")

            # Filter controls
            search = st.text_input("Search Candidate Name or ID", "")
            filtered_p = df_persons.copy()
            if search:
                filtered_p = filtered_p[filtered_p["full_name"].str.contains(search, case=False) | filtered_p["id"].str.contains(search, case=False)]

            for idx, row in filtered_p.head(10).iterrows():
                st.markdown(f"""
                    <div class="entity-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:16px; font-weight:600;">{row['full_name']} <span style="font-size:12px; color:#9ca3af;">({row['id']} | {row['party_name']})</span></span>
                            <span class="status-badge badge-pending">Review Queue</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                col_plat, col_url, col_act = st.columns([2, 4, 2])
                with col_plat:
                    selected_platform = st.selectbox("Platform", [p["name"] for p in SOCIAL_PLATFORMS], key=f"plat_sel_{idx}")
                
                # Fetch current URL field dynamically
                plat_info = next(p for p in SOCIAL_PLATFORMS if p["name"] == selected_platform)
                current_url = row[plat_info["url_col"]]

                with col_url:
                    new_url = st.text_input("URL Link", value=current_url, key=f"url_val_{idx}")
                with col_act:
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("✅ Approve", key=f"btn_app_{idx}"):
                            st.session_state["popolo_tables"]["Persons"].at[idx, plat_info["url_col"]] = new_url
                            st.session_state["popolo_tables"]["Persons"].at[idx, plat_info["note_col"]] = "Verified & Approved"
                            st.rerun()
                    with b2:
                        if st.button("🚫 Set N/A", key=f"btn_na_{idx}"):
                            st.session_state["popolo_tables"]["Persons"].at[idx, plat_info["url_col"]] = "N/A"
                            st.session_state["popolo_tables"]["Persons"].at[idx, plat_info["note_col"]] = "Confirmed N/A"
                            st.rerun()

        elif target_tab == "🏛️ Political Parties Mapping":
            st.subheader("Party Domain Social Media Coverage")

            total_parties = len(df_parties)
            c1, c2 = st.columns(2)
            c1.metric("Unique Extracted Parties", total_parties)
            c2.metric("Target Country", selected_country)

            st.markdown("---")
            st.subheader("🖐️ Political Party Verification Workflow")

            for idx, row in df_parties.iterrows():
                st.markdown(f"""
                    <div class="entity-card">
                        <span style="font-size:16px; font-weight:600;">{row['party_name']} <span style="font-size:12px; color:#9ca3af;">({row['abbrv']} | ID: {row['party_id']})</span></span>
                    </div>
                """, unsafe_allow_html=True)

                cp1, cp2, cp3, cp4 = st.columns([3, 3, 3, 1.5])
                with cp1:
                    p_web = st.text_input("Official Website", value=row["official_website"], key=f"p_web_{idx}")
                with cp2:
                    p_tw = st.text_input("Twitter / X", value=row["twitter_url"], key=f"p_tw_{idx}")
                with cp3:
                    p_fb = st.text_input("Facebook Page", value=row["facebook_url"], key=f"p_fb_{idx}")
                with cp4:
                    if st.button("✅ Save", key=f"p_save_{idx}"):
                        st.session_state["popolo_tables"]["Parties"].at[idx, "official_website"] = p_web
                        st.session_state["popolo_tables"]["Parties"].at[idx, "twitter_url"] = p_tw
                        st.session_state["popolo_tables"]["Parties"].at[idx, "facebook_url"] = p_fb
                        st.session_state["popolo_tables"]["Parties"].at[idx, "mapping_status"] = "Approved"
                        st.rerun()
    else:
        st.warning("Please upload candidate data in the Ingestion tab to activate social media mapping.")
