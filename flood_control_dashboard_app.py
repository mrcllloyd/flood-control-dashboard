
import os, re
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import plotly.express as px

st.set_page_config(page_title="Flood Control Projects — Dashboard",
                   layout="wide",
                   page_icon="🌊")

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 1.1rem; max-width: 1500px;}
h1, h2, h3 { margin-top: 0.5rem; margin-bottom: 0.35rem; }
section[data-testid="stSidebar"] .block-container {padding-top: 0.6rem;}
.dataframe tbody tr, .dataframe thead tr { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(path_local: str, path_alt: str = None) -> pd.DataFrame:
    p = path_local if os.path.exists(path_local) else (path_alt if (path_alt and os.path.exists(path_alt)) else path_local)
    df = pd.read_excel(p, sheet_name="Data")
    df.columns = [re.sub(r"\s+", " ", str(c).strip().lower()) for c in df.columns]
    col_map = {
        "projectid": "ProjectID",
        "projectdescription": "ProjectDescription",
        "presterm": "Year",
        "approvedbudgetforthecontract": "ABC",
        "contractcost": "ContractCost",
        "contractor": "Contractor",
        "region": "Region",
        "province": "Province",
        "municipality": "Municipality",
        "districtengineeringoffice": "DEO",
        "latitude": "Latitude",
        "longitude": "Longitude",
    }
    for k,v in col_map.items():
        if k in df.columns:
            df.rename(columns={k:v}, inplace=True)
    for c in ["ABC","ContractCost","Latitude","Longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    def parse_year(x):
        if pd.isna(x): return None
        s = str(x).strip()
        m = re.search(r"(20\d{2}|19\d{2})", s)
        return int(m.group(1)) if m else None
    df["YearParsed"] = df.get("Year", pd.Series([None]*len(df))).apply(parse_year)
    df["HasGeo"] = (~df.get("Latitude", pd.Series([np.nan]*len(df))).isna()) & (~df.get("Longitude", pd.Series([np.nan]*len(df))).isna())
    return df

# Resolve data path (env var takes precedence)
DATA_ENV = os.environ.get("FC_DASHBOARD_XLSX", "").strip()
DEFAULT_MAIN = "/mnt/data/flood_control_v6 (1).xlsx"
DEFAULT_ALT  = "data/flood_control_v6.xlsx"
data_path = DATA_ENV if DATA_ENV else DEFAULT_MAIN

df = load_data(data_path, DEFAULT_ALT)

st.title("Flood Control Projects — Research & Application Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

year_min = int(df["YearParsed"].min()) if df["YearParsed"].notna().any() else None
year_max = int(df["YearParsed"].max()) if df["YearParsed"].notna().any() else None
if year_min and year_max:
    sel_years = st.sidebar.slider("Year Range", min_value=year_min, max_value=year_max, value=(year_min, year_max), step=1)
else:
    sel_years = (None, None)

regions = ["All"] + sorted(df.get("Region", pd.Series(dtype=str)).dropna().unique().tolist()) if "Region" in df.columns else ["All"]
provinces = ["All"] + sorted(df.get("Province", pd.Series(dtype=str)).dropna().unique().tolist()) if "Province" in df.columns else ["All"]
deos = ["All"] + sorted(df.get("DEO", pd.Series(dtype=str)).dropna().unique().tolist()) if "DEO" in df.columns else ["All"]
contractors = ["All"] + sorted(df.get("Contractor", pd.Series(dtype=str)).dropna().unique().tolist()) if "Contractor" in df.columns else ["All"]

sel_region = st.sidebar.selectbox("Region", regions, index=0)
sel_province = st.sidebar.selectbox("Province", provinces, index=0)
sel_deo = st.sidebar.selectbox("District Engineering Office (DEO)", deos, index=0)
sel_contractor = st.sidebar.selectbox("Contractor", contractors, index=0)

min_cost = float(df.get("ContractCost", pd.Series([0])).fillna(0).min()) if "ContractCost" in df.columns else 0.0
max_cost = float(df.get("ContractCost", pd.Series([0])).fillna(0).max()) if "ContractCost" in df.columns else 1000.0
sel_cost = st.sidebar.slider("Contract Cost (₱)", min_value=0.0, max_value=max(1000.0, max_cost), value=(0.0, max_cost))

kw = st.sidebar.text_input("Search in Project Description", "")

# Map layer toggle
layer_choice = st.sidebar.radio("Map Layer", ["Scatter (points)", "Heatmap", "Hexagon"], index=0)

# Apply filters
f = df.copy()

if sel_years[0] is not None and sel_years[1] is not None and "YearParsed" in f.columns:
    f = f[(f["YearParsed"].fillna(0) >= sel_years[0]) & (f["YearParsed"].fillna(0) <= sel_years[1])]

def apply_eq(frame, col, val):
    if col in frame.columns and val != "All":
        return frame[frame[col] == val]
    return frame

f = apply_eq(f, "Region", sel_region)
f = apply_eq(f, "Province", sel_province)
f = apply_eq(f, "DEO", sel_deo)
f = apply_eq(f, "Contractor", sel_contractor)

if "ContractCost" in f.columns:
    f = f[(f["ContractCost"].fillna(0) >= sel_cost[0]) & (f["ContractCost"].fillna(0) <= sel_cost[1])]

if kw.strip():
    pat = re.compile(re.escape(kw.strip()), re.IGNORECASE)
    text_col = "ProjectDescription" if "ProjectDescription" in f.columns else f.columns[0]
    f = f[f[text_col].astype(str).str.contains(pat)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
total_projects = len(f)
total_cost = float(f.get("ContractCost", pd.Series([0])).fillna(0).sum())
avg_cost = (total_cost / total_projects) if total_projects else 0.0
geo_cov = int(f.get("HasGeo", pd.Series([False]*len(f))).sum())
with col1: st.metric("Total Projects", f"{total_projects:,}")
with col2: st.metric("Total Contract Cost (₱)", f"{total_cost:,.0f}")
with col3: st.metric("Average Cost / Project (₱)", f"{avg_cost:,.0f}")
with col4: st.metric("Projects With Coordinates", f"{geo_cov:,}")

st.divider()

# Map with layer choice (uses ALL filtered rows with coordinates)
st.subheader("Geographic Distribution")
if ("Latitude" in f.columns) and ("Longitude" in f.columns) and (f["HasGeo"].any() if "HasGeo" in f.columns else False):
    g = f[f["HasGeo"]].copy()

    if len(g) == 0:
        st.info("No rows with coordinates after applying filters.")
    else:
        # default center
        center_lat = float(g["Latitude"].mean())
        center_lon = float(g["Longitude"].mean())
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=5.2, pitch=0)

        layers = []
        if layer_choice == "Scatter (points)":
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=g,
                get_position='[Longitude, Latitude]',
                get_radius=65,
                get_fill_color=[22, 96, 167, 160],
                pickable=True,
                radius_scale=10,
                radius_min_pixels=3,
                radius_max_pixels=80,
            ))
        elif layer_choice == "Heatmap":
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=g,
                get_position='[Longitude, Latitude]',
                aggregation='MEAN',
                threshold=0.3,
                intensity=1.0,
                radius_pixels=40,
            ))
        else:  # Hexagon
            layers.append(pdk.Layer(
                "HexagonLayer",
                data=g,
                get_position='[Longitude, Latitude]',
                radius=2500,
                elevation_scale=30,
                elevation_range=[0, 4000],
                extruded=True,
                coverage=1
            ))

        tooltip = {
            "html": "<b>{ProjectDescription}</b><br/>"
                    "Region: {Region}<br/>Province: {Province}<br/>Municipality: {Municipality}<br/>"
                    "Year: {YearParsed}<br/>Contract Cost: ₱{ContractCost}",
            "style": {"backgroundColor": "rgba(30,30,30,0.8)", "color": "white", "font-size": "12px"}
        }

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=layers,
            tooltip=tooltip
        ))
else:
    st.info("No usable Latitude/Longitude columns found, or no rows with coordinates after filtering.")

st.divider()

# Charts
st.subheader("Funding Overview")
colA, colB = st.columns(2)

if "Region" in f.columns and "ContractCost" in f.columns:
    by_region = f.groupby("Region", as_index=False)["ContractCost"].sum().sort_values("ContractCost", ascending=False).head(20)
    figA = px.bar(by_region, x="ContractCost", y="Region", orientation="h",
                  title="Top Regions by Contract Cost", labels={"ContractCost":"₱"})
    colA.plotly_chart(figA, use_container_width=True)
else:
    colA.info("Region/ContractCost columns missing.")

if "Province" in f.columns and "ContractCost" in f.columns:
    by_prov = f.groupby("Province", as_index=False)["ContractCost"].sum().sort_values("ContractCost", ascending=False).head(20)
    figB = px.bar(by_prov, x="ContractCost", y="Province", orientation="h",
                  title="Top Provinces by Contract Cost", labels={"ContractCost":"₱"})
    colB.plotly_chart(figB, use_container_width=True)
else:
    colB.info("Province/ContractCost columns missing.")

st.divider()

# Details + Downloads
st.subheader("Project Details (Filtered)")
show_cols = [c for c in ["ProjectID","ProjectDescription","YearParsed","Region","Province","Municipality","DEO","Contractor","ABC","ContractCost","Latitude","Longitude"] if c in f.columns]
if not show_cols:
    show_cols = list(f.columns)
st.dataframe(f[show_cols].sort_values(["Region","Province","YearParsed"], na_position="last"), use_container_width=True, height=450)

def to_csv_bytes(df_):
    return df_.to_csv(index=False).encode("utf-8")

st.download_button("Download filtered data (CSV)", data=to_csv_bytes(f[show_cols]), file_name="flood_control_filtered.csv")
if "Region" in f.columns and "ContractCost" in f.columns:
    st.download_button("Download Region summary (CSV)", data=to_csv_bytes(by_region), file_name="summary_by_region.csv")
if "Province" in f.columns and "ContractCost" in f.columns:
    st.download_button("Download Province summary (CSV)", data=to_csv_bytes(by_prov), file_name="summary_by_province.csv")

st.caption("Adjust filters in the left sidebar. The map, charts, and table update instantly. All rows in the data file are used—no sampling.")
