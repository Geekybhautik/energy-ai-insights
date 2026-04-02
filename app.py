import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Global Energy AI Insights",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1a1f2e; border-radius: 10px; padding: 10px; border: 1px solid #2d3748; }
    .stMetric label { color: #a0aec0 !important; }
    div[data-testid="stSidebarContent"] { background-color: #111827; }
    .insight-box { background: #1a1f2e; border-left: 4px solid #38a169;
                   border-radius: 8px; padding: 15px; margin: 10px 0; color: #e2e8f0; }
    .warning-box { background: #1a1f2e; border-left: 4px solid #e53e3e;
                   border-radius: 8px; padding: 15px; margin: 10px 0; color: #e2e8f0; }
    h1, h2, h3 { color: #e2e8f0; }
    .stButton > button { background: linear-gradient(135deg, #2d7d46, #38a169);
                         color: white; border: none; border-radius: 8px;
                         padding: 10px 24px; font-weight: 600; }
    .stButton > button:hover { background: linear-gradient(135deg, #38a169, #48bb78); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. LOAD ORIGINAL DATA — EDA & Visualizations
# ─────────────────────────────────────────────
@st.cache_data
def load_data():

    # Try multiple filename formats (important fix)
    possible_files = [
        "World_Energy_Consumption.csv",
        "World Energy Consumption.csv"
    ]

    file_found = None
    for f in possible_files:
        if os.path.exists(f):
            file_found = f
            break

    if file_found is None:
        st.error("❌ Dataset file not found. Please check filename.")
        st.stop()

    df = pd.read_csv(file_found)

    # ✅ FIX: Normalize column names (THIS SOLVES YOUR ERROR)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Required columns
    desired_cols = [
        'country', 'year',
        'energy_per_capita',
        'renewables_share_energy',
        'fossil_share_energy',
        'co2_per_capita',
        'gdp', 'population',
        'solar_share_energy',
        'wind_share_energy',
        'nuclear_share_energy',
        'coal_share_energy',
        'oil_share_energy',
        'gas_share_energy',
        'electricity_demand',
        'primary_energy_consumption'
    ]

    # Keep only available columns
    available = [c for c in desired_cols if c in df.columns]
    df = df[available]

    # ✅ Safe check before drop
    if 'energy_per_capita' not in df.columns:
        st.error("❌ 'energy_per_capita' column not found in dataset.")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    df = df.dropna(subset=['country', 'year', 'energy_per_capita'])

    # Remove aggregated regions
    regions_to_exclude = [
        'World', 'Europe', 'Asia', 'Africa', 'North America',
        'South America', 'Oceania', 'Middle East', 'European Union (27)',
        'High-income countries', 'Low-income countries', 'OECD',
        'Non-OECD', 'Upper-middle-income countries', 'Lower-middle-income countries'
    ]

    df = df[~df['country'].isin(regions_to_exclude)]
    df['year'] = df['year'].astype(int)

    return df

# ─────────────────────────────────────────────
# 4. LOAD CLEAN DATA — ML Training only
# ─────────────────────────────────────────────
@st.cache_data
def load_clean_data():

    possible_files = [
        "energy_ml_clean.csv",
        "energy_ml_clean (1).csv",
        "energy_ml_clean(1).csv"
    ]

    for fname in possible_files:
        if os.path.exists(fname):
            df = pd.read_csv(fname)

            # ✅ Normalize columns here too
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            df['year'] = df['year'].astype(int)
            return df

    st.warning("⚠️ Clean dataset not found — using original dataset.")
    return load_data()

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING — on EDA data
# ─────────────────────────────────────────────
@st.cache_data
def engineer_features(df):
    df = df.copy()

    if 'renewables_share_energy' in df.columns and 'fossil_share_energy' in df.columns:
        df['renewable_fossil_gap'] = df['renewables_share_energy'] - df['fossil_share_energy']

    if 'gdp' in df.columns and 'population' in df.columns:
        df['energy_per_gdp'] = df['energy_per_capita'] / (
            df['gdp'] / df['population'].replace(0, np.nan) + 1
        )

    if 'renewables_share_energy' in df.columns and 'co2_per_capita' in df.columns:
        r_norm = df['renewables_share_energy'] / (df['renewables_share_energy'].max() + 1e-9)
        c_norm = 1 - df['co2_per_capita'] / (df['co2_per_capita'].max() + 1e-9)
        df['sustainability_score'] = (r_norm + c_norm) / 2 * 100

    return df

# ─────────────────────────────────────────────
# 6. TRAIN ML MODEL — using CLEAN dataset
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = load_clean_data()
    df = df.copy()

    # ✅ Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # ✅ Check required column
    if 'country' not in df.columns:
        st.error("❌ 'country' column not found in clean dataset.")
        st.stop()

    df['country_encoded'] = le.fit_transform(df['country'])

    # ✅ Use correct target column
    target_col = "primary_energy_consumption"

    if target_col not in df.columns:
        st.error(f"❌ '{target_col}' column not found in clean dataset.")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    # ✅ Feature selection
    feature_cols = ['country_encoded', 'year']
    optional_feats = [
        'gdp', 'population',
        'energy_lag_1', 'energy_lag_2', 'energy_lag_5',
        'energy_intensity'
    ]

    for col in optional_feats:
        if col in df.columns:
            feature_cols.append(col)

    model_df = df[feature_cols + [target_col]].dropna()

    if len(model_df) < 50:
        st.error("❌ Not enough rows in clean dataset to train model.")
        st.stop()

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import numpy as np

    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "r2": round(r2_score(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    }

    return model, le, feature_cols, metrics

# ─────────────────────────────────────────────
# 7. GROQ AI CLIENT
# ─────────────────────────────────────────────
def get_groq_client():
    try:
        from groq import Groq
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception:
        return None

def ask_groq(client, prompt: str) -> str:
    if client is None:
        return "⚠️ AI unavailable — add GROQ_API_KEY to `.streamlit/secrets.toml`."
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI error: {str(e)}"

# ─────────────────────────────────────────────
# 8. INITIALISE EVERYTHING
# ─────────────────────────────────────────────
df_raw      = load_data()
df          = engineer_features(df_raw)
model, le, feature_cols, model_metrics = train_model()
groq_client = get_groq_client()

# Countries the ML model knows about (from clean dataset)
_clean_df       = load_clean_data()
clean_countries = sorted(_clean_df['country'].unique()) if 'country' in _clean_df.columns else sorted(df['country'].unique())

# ─────────────────────────────────────────────
# 9. SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
st.sidebar.title("⚡ Energy AI Insights")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔍 EDA & Patterns", "🤖 AI Chatbot", "🔮 Energy Predictor", "📈 ML Model Info"]
)

st.sidebar.markdown("---")
st.sidebar.metric("Countries (EDA)", df['country'].nunique())
st.sidebar.metric("Year Range",      f"{df['year'].min()} – {df['year'].max()}")
st.sidebar.metric("Total Records",   f"{len(df):,}")
st.sidebar.markdown("---")
st.sidebar.caption("📂 EDA  → World_Energy_Consumption.csv")
st.sidebar.caption("🤖 ML   → energy_ml_clean (1).csv")

# ─────────────────────────────────────────────
# PAGE 1: DASHBOARD
# ─────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("📊 Global Energy Dashboard")
    st.caption("Explore worldwide energy consumption, renewable adoption, and CO₂ emissions.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Energy/Capita",    f"{df['energy_per_capita'].mean():.0f} kWh")
    col2.metric("Max Renewables Share",
                f"{df['renewables_share_energy'].max():.1f}%"
                if 'renewables_share_energy' in df.columns else "N/A")
    col3.metric("Avg CO₂/Capita",
                f"{df['co2_per_capita'].mean():.2f} t"
                if 'co2_per_capita' in df.columns else "N/A")
    col4.metric("Total Records", f"{len(df):,}")

    st.markdown("---")

    # World map
    st.subheader("🌍 Energy Consumption per Capita (Latest Year)")
    latest_year = int(df['year'].max())
    map_df = df[df['year'] == latest_year].dropna(subset=['energy_per_capita'])
    fig_map = px.choropleth(
        map_df, locations="country", locationmode="country names",
        color="energy_per_capita", hover_name="country",
        color_continuous_scale="Viridis",
        title=f"Global Energy Use per Capita — {latest_year}"
    )
    fig_map.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="#e2e8f0", geo=dict(bgcolor="#0e1117")
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Renewable vs Fossil trend
    if 'renewables_share_energy' in df.columns and 'fossil_share_energy' in df.columns:
        st.subheader("🔄 Global Energy Transition: Renewables vs Fossil Fuels")
        trend_cols   = ['renewables_share_energy', 'fossil_share_energy']
        global_trend = df.groupby('year')[trend_cols].mean().reset_index()
        fig_trend = px.line(
            global_trend, x="year", y=trend_cols,
            labels={"value": "Share (%)", "variable": "Energy Source"},
            title="Global Average Energy Share Over Time",
            color_discrete_map={
                "renewables_share_energy": "#68d391",
                "fossil_share_energy":     "#fc8181"
            }
        )
        fig_trend.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                                 font_color="#e2e8f0")
        st.plotly_chart(fig_trend, use_container_width=True)

    # Top 10 consumers
    st.subheader(f"🏆 Top 10 Energy Consuming Countries ({latest_year})")
    top10 = (df[df['year'] == latest_year]
             .nlargest(10, 'energy_per_capita')[['country', 'energy_per_capita']])
    fig_bar = px.bar(
        top10, x="energy_per_capita", y="country", orientation="h",
        color="energy_per_capita", color_continuous_scale="Viridis",
        title="Top 10 by Energy per Capita"
    )
    fig_bar.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
        font_color="#e2e8f0", yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2: EDA & PATTERNS
# ─────────────────────────────────────────────
elif page == "🔍 EDA & Patterns":
    st.title("🔍 EDA & Energy Patterns")
    st.caption("Source: World_Energy_Consumption.csv (original dataset)")

    countries = sorted(df['country'].unique())
    default_c = [c for c in ["India", "United States", "Germany", "China"] if c in countries]
    selected  = st.multiselect("Select countries to compare", countries, default=default_c)

    if selected:
        cdf = df[df['country'].isin(selected)]

        # Renewable share over time
        if 'renewables_share_energy' in df.columns:
            st.subheader("📈 Renewable Energy Share Over Time")
            fig = px.line(
                cdf, x="year", y="renewables_share_energy", color="country",
                title="Renewable Share (%) by Country",
                labels={"renewables_share_energy": "Renewables (%)"}
            )
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                               font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)

        # CO2 vs Renewable scatter
        if 'co2_per_capita' in df.columns and 'renewables_share_energy' in df.columns:
            st.subheader("🔗 CO₂ Emissions vs Renewable Share")
            scatter_df = df[df['year'] == int(df['year'].max())].dropna(
                subset=['co2_per_capita', 'renewables_share_energy', 'energy_per_capita']
            )
            fig_scatter = px.scatter(
                scatter_df, x="renewables_share_energy", y="co2_per_capita",
                hover_name="country", size="energy_per_capita",
                color="co2_per_capita", color_continuous_scale="RdYlGn_r",
                title=f"CO₂ vs Renewable Share — {int(df['year'].max())}",
                labels={
                    "renewables_share_energy": "Renewable Share (%)",
                    "co2_per_capita":          "CO₂ per Capita (t)"
                }
            )
            fig_scatter.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                                       font_color="#e2e8f0")
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("""<div class="insight-box">
            📌 <b>Insight:</b> Countries with higher renewable share consistently show lower
            CO₂ emissions per capita — confirming the energy transition's climate impact.
            </div>""", unsafe_allow_html=True)

        # Stacked energy source breakdown
        source_cols = [c for c in [
            'solar_share_energy', 'wind_share_energy', 'nuclear_share_energy',
            'coal_share_energy', 'oil_share_energy', 'gas_share_energy'
        ] if c in df.columns]

        if source_cols:
            st.subheader("⚡ Energy Source Breakdown (Latest Year)")
            latest_df  = df[df['year'] == int(df['year'].max())]
            sel_latest = latest_df[latest_df['country'].isin(selected)]
            melt = sel_latest.melt(
                id_vars='country', value_vars=source_cols,
                var_name='Source', value_name='Share'
            )
            melt['Source'] = melt['Source'].str.replace('_share_energy', '').str.title()
            fig_stack = px.bar(
                melt, x='country', y='Share', color='Source', barmode='stack',
                title="Energy Mix by Country (% share)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_stack.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                                     font_color="#e2e8f0")
            st.plotly_chart(fig_stack, use_container_width=True)

        # Energy per capita over time
        st.subheader("🔋 Energy per Capita Comparison Over Time")
        fig_epc = px.line(
            cdf, x="year", y="energy_per_capita", color="country",
            title="Energy per Capita (kWh) Over Time",
            labels={"energy_per_capita": "Energy/Capita (kWh)"}
        )
        fig_epc.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                               font_color="#e2e8f0")
        st.plotly_chart(fig_epc, use_container_width=True)

    # Sustainability Score ranking
    if 'sustainability_score' in df.columns:
        st.subheader("🌱 Sustainability Score — Top & Bottom Countries")
        latest_sus = (
            df[df['year'] == int(df['year'].max())]
            .dropna(subset=['sustainability_score'])
            .sort_values('sustainability_score', ascending=False)
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 10 Most Sustainable**")
            st.dataframe(
                latest_sus[['country', 'sustainability_score']].head(10)
                .rename(columns={"sustainability_score": "Score (0–100)"})
                .reset_index(drop=True),
                use_container_width=True
            )
        with c2:
            st.markdown("**Bottom 10 Least Sustainable**")
            st.dataframe(
                latest_sus[['country', 'sustainability_score']].tail(10)
                .rename(columns={"sustainability_score": "Score (0–100)"})
                .reset_index(drop=True),
                use_container_width=True
            )

# ─────────────────────────────────────────────
# PAGE 3: AI CHATBOT
# ─────────────────────────────────────────────
elif page == "🤖 AI Chatbot":
    st.title("🤖 AI Energy Policy Advisor")
    st.caption("Ask anything about global energy trends, sustainability, and policy.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    latest_year    = int(df['year'].max())
    top_emitters   = (df[df['year'] == latest_year]
                      .nlargest(5, 'co2_per_capita')[['country', 'co2_per_capita']]
                      if 'co2_per_capita' in df.columns else pd.DataFrame())
    top_renewables = (df[df['year'] == latest_year]
                      .nlargest(5, 'renewables_share_energy')[['country', 'renewables_share_energy']]
                      if 'renewables_share_energy' in df.columns else pd.DataFrame())

    system_context = f"""
You are an expert energy policy advisor and data scientist.
Dataset summary:
- Spans {df['year'].min()} to {df['year'].max()} across {df['country'].nunique()} countries.
- Average global energy per capita: {df['energy_per_capita'].mean():.0f} kWh.
- Top CO₂ emitters per capita ({latest_year}): {top_emitters.to_string(index=False) if not top_emitters.empty else 'N/A'}.
- Top renewable energy countries ({latest_year}): {top_renewables.to_string(index=False) if not top_renewables.empty else 'N/A'}.
Answer questions accurately and concisely. Focus on sustainability insights.
"""

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about energy trends, countries, policies...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = ask_groq(groq_client, system_context + f"\n\nQuestion: {user_input}")
                st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    st.markdown("---")
    st.markdown("**Quick questions:**")
    quick_qs = [
        "Which countries lead in renewable energy adoption?",
        "What is the relationship between GDP and energy consumption?",
        "Which regions are most dependent on fossil fuels?",
        "What policies can accelerate the clean energy transition?"
    ]
    btn_cols = st.columns(2)
    for i, q in enumerate(quick_qs):
        if btn_cols[i % 2].button(q, key=f"q_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": q})
            reply = ask_groq(groq_client, system_context + f"\n\nQuestion: {q}")
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
# PAGE 4: ENERGY PREDICTOR
# ─────────────────────────────────────────────
elif page == "🔮 Energy Predictor":
    st.title("🔮 Energy Demand Predictor")
    st.caption("ML model trained on clean dataset → energy_ml_clean (1).csv")

    col1, col2 = st.columns(2)
    with col1:
        target_country = st.selectbox("Select Country", clean_countries)
    with col2:
        target_year = st.slider("Year to Predict", min_value=2023, max_value=2050, value=2030)

    st.markdown("#### Optional: Additional inputs for better accuracy")
    adv1, adv2 = st.columns(2)
    with adv1:
        gdp_input = st.number_input("GDP (USD — 0 = use historical avg)",
                                     min_value=0.0, value=0.0, step=1e9, format="%.0f")
        pop_input = st.number_input("Population (0 = use historical avg)",
                                     min_value=0.0, value=0.0, step=1e6, format="%.0f")
    with adv2:
        fossil_input = st.slider("Fossil Fuel Share (%)", 0.0, 100.0, 50.0)
        co2_input    = st.number_input("CO₂ per Capita (tonnes)", min_value=0.0, value=5.0)

    if st.button("🔮 Predict Energy Demand"):
        if target_country not in le.classes_:
            st.error(f"❌ '{target_country}' was not in the training data. Please select another.")
        else:
            encoded_country = le.transform([target_country])[0]
            country_hist    = df[df['country'] == target_country]

            def hist_mean(col):
                return country_hist[col].mean() if col in country_hist.columns and not country_hist[col].isna().all() else 0.0

            input_data = {'country_encoded': encoded_country, 'year': target_year}
            if 'gdp'                         in feature_cols:
                input_data['gdp']                         = gdp_input if gdp_input > 0 else hist_mean('gdp')
            if 'population'                  in feature_cols:
                input_data['population']                  = pop_input if pop_input > 0 else hist_mean('population')
            if 'fossil_share_energy'         in feature_cols:
                input_data['fossil_share_energy']         = fossil_input
            if 'co2_per_capita'              in feature_cols:
                input_data['co2_per_capita']              = co2_input
            if 'renewables_share_energy'     in feature_cols:
                input_data['renewables_share_energy']     = hist_mean('renewables_share_energy')
            if 'primary_energy_consumption'  in feature_cols:
                input_data['primary_energy_consumption']  = hist_mean('primary_energy_consumption')

            X_pred     = pd.DataFrame([input_data])[feature_cols]
            prediction = model.predict(X_pred)[0]

            hist_epc   = country_hist['energy_per_capita'].dropna()
            latest_val = hist_epc.iloc[-1] if not hist_epc.empty else None

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.metric(f"Predicted ({target_year})", f"{prediction:,.0f} kWh/capita")
            if latest_val:
                change = ((prediction - latest_val) / latest_val) * 100
                r2.metric("Latest Actual", f"{latest_val:,.0f} kWh/capita")
                r3.metric("Projected Change", f"{change:+.1f}%",
                           delta_color="inverse" if change > 20 else "normal")

            # Historical + prediction chart
            if not hist_epc.empty:
                hist_df      = country_hist[['year', 'energy_per_capita']].dropna().copy()
                hist_df['type'] = 'Historical'
                pred_df      = pd.DataFrame({
                    'year': [target_year], 'energy_per_capita': [prediction], 'type': ['Predicted']
                })
                combined = pd.concat([hist_df, pred_df], ignore_index=True)
                fig = px.line(
                    combined[combined['type'] == 'Historical'],
                    x='year', y='energy_per_capita',
                    title=f"{target_country} — Energy Consumption History + Prediction",
                    labels={'energy_per_capita': 'Energy per Capita (kWh)'}
                )
                fig.add_scatter(
                    x=[target_year], y=[prediction], mode='markers',
                    marker=dict(size=14, color='#68d391', symbol='star'),
                    name=f'Predicted {target_year}'
                )
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                                   font_color="#e2e8f0")
                st.plotly_chart(fig, use_container_width=True)

            # AI interpretation
            st.markdown("---")
            st.subheader("💡 AI Interpretation")
            with st.spinner("Generating AI insight..."):
                explain_prompt = f"""
You are an energy policy expert. Analyze this ML model prediction:
- Country: {target_country}
- Predicted energy consumption in {target_year}: {prediction:,.0f} kWh per capita
- Fossil fuel share: {fossil_input}%
- CO₂ per capita: {co2_input} tonnes

In 3-4 sentences, explain what this prediction means for {target_country}'s 
sustainability trajectory and what specific policy actions could help.
"""
                ai_reply = ask_groq(groq_client, explain_prompt)
                st.info(ai_reply)

# ─────────────────────────────────────────────
# PAGE 5: ML MODEL INFO
# ─────────────────────────────────────────────
elif page == "📈 ML Model Info":
    st.title("📈 ML Model — Details & Performance")
    st.caption("Model trained on: energy_ml_clean (1).csv")

    st.subheader("Model Architecture")
    st.markdown(f"""
| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Task | Regression — predict energy per capita (kWh) |
| Target variable | `energy_per_capita` |
| Training dataset | energy_ml_clean (1).csv |
| Train / Test split | 80% / 20% |
| Number of trees | 150 estimators |
| Features used | {len(feature_cols)} |
""")

    st.subheader("📊 Model Performance on Test Set")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{model_metrics['r2']:.4f}",
              help="1.0 = perfect. >0.85 is excellent.")
    m2.metric("MAE",      f"{model_metrics['mae']:,} kWh",
              help="Mean Absolute Error.")
    m3.metric("RMSE",     f"{model_metrics['rmse']:,} kWh",
              help="Root Mean Squared Error.")

    if model_metrics['r2'] >= 0.85:
        st.markdown("""<div class="insight-box">
        ✅ <b>R² ≥ 0.85</b> — the model explains over 85% of variance in energy consumption.
        Strong performance for country-level tabular data.
        </div>""", unsafe_allow_html=True)
    elif model_metrics['r2'] >= 0.70:
        st.markdown("""<div class="warning-box">
        ⚠️ <b>R² between 0.70–0.85</b> — decent. Consider adding more features.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="warning-box">
        ❌ <b>R² below 0.70</b> — check clean dataset for sufficient rows and relevant columns.
        </div>""", unsafe_allow_html=True)

    # Feature importance
    st.subheader("🔍 Feature Importance")
    feat_df = pd.DataFrame({
        'Feature':    feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig_imp = px.bar(
        feat_df, x='Importance', y='Feature', orientation='h',
        title="Feature Importance — Random Forest",
        color='Importance', color_continuous_scale='Greens'
    )
    fig_imp.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                           font_color="#e2e8f0")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("📋 Features Used for Training")
    st.code(", ".join(feature_cols), language="python")

    st.subheader("🎯 What This Model Predicts")
    st.markdown("""
Given a **country** and **year** (plus optional economic features), the model predicts
how much energy (kWh per capita) that country will consume.

This directly serves the project objective:
> *"Predict sustainability indicators using ML models"*

**Predictions are used to:**
- Identify countries likely to increase fossil fuel dependency
- Flag nations needing urgent energy efficiency policies
- Power the AI advisor with data-grounded forecasts
- Compare a country's trajectory against global sustainability targets
""")