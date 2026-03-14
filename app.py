# PARCOURS A : DETECTION DES BIAIS
# DATASET : MEDICAL INSURANCE COST

#importation des bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(
    page_title="Détection de biais – Insurance Cost",
    page_icon="🧮",
    layout="wide",
)


#helpers & métriques de fairness

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    desc = pd.DataFrame({
        "colonne": df.columns,
        "type": [str(t) for t in df.dtypes],
        "exemples": [", ".join(map(str, df[c].dropna().unique()[:5])) for c in df.columns]
    })
    return desc

def make_age_band(s: pd.Series) -> pd.Series:
    bins = [0, 30, 50, 100]
    labels = ['<30', '30-50', '>50']
    return pd.cut(s, bins=bins, labels=labels, right=True, include_lowest=True)

def demographic_parity_difference(y: pd.Series, sensitive) -> tuple[float, dict]:
    """Écart de parité démographique = max(taux+) - min(taux+) entre groupes."""
    y_series = pd.Series(y).reset_index(drop=True)
    s_series = pd.Series(sensitive).reset_index(drop=True)
    rates = {}
    for g in s_series.unique():
        mask = (s_series == g)
        rates[g] = float(y_series[mask].mean()) if mask.sum() > 0 else np.nan
    rates = {k: v for k, v in rates.items() if pd.notna(v)}
    if not rates:
        return np.nan, rates
    return float(max(rates.values()) - min(rates.values())), rates

def disparate_impact_ratio(y: pd.Series, sensitive) -> tuple[float, dict]:
    """Disparate Impact = min(taux+) / max(taux+) entre groupes."""
    y_series = pd.Series(y).reset_index(drop=True)
    s_series = pd.Series(sensitive).reset_index(drop=True)
    rates = {}
    for g in s_series.unique():
        mask = (s_series == g)
        rates[g] = float(y_series[mask].mean()) if mask.sum() > 0 else np.nan
    rates = {k: v for k, v in rates.items() if pd.notna(v)}
    if not rates:
        return np.nan, rates
    m = max(rates.values())
    return (float(min(rates.values()) / m) if m > 0 else np.nan), rates

@st.cache_resource
def build_rf_clf(numeric_features, categorical_features):
    from sklearn.ensemble import RandomForestClassifier
    preprocess = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline(steps=[('preprocess', preprocess),
                           ('model', rf)])
    return pipe


#sidebar : Navigation & paramètres

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Aller à",
    ["🏠 Accueil", "📊 Exploration des données", "⚠️ Détection de biais", "🤖 Modélisation"],
    key="nav_radio"
)

# Paramètres globaux
st.sidebar.markdown("---")
st.sidebar.subheader("Paramètres généraux")
data_path = st.sidebar.text_input("Chemin du dataset", "insurance.csv", key="data_path")
df = load_data(data_path)

#outcome binaire basé sur charges
default_threshold = float(df["charges"].median())
threshold = st.sidebar.slider(
    "Seuil 'high_cost' (charges ≥ seuil)",
    min_value=float(df["charges"].quantile(0.1)),
    max_value=float(df["charges"].quantile(0.9)),
    value=default_threshold,
    step=100.0,
    help="Par défaut: médiane des charges",
    key="threshold_slider"
)
df["high_cost"] = (df["charges"] >= threshold).astype(int)
df["age_band"] = make_age_band(df["age"])

#filtres globaux (utilisés aussi en Exploration)
st.sidebar.markdown("---")
st.sidebar.subheader("Filtres")
age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider("Âge", min_value=age_min, max_value=age_max, value=(age_min, age_max), key="age_range")
regions = st.sidebar.multiselect("Région", sorted(df["region"].unique()), default=sorted(df["region"].unique()), key="regions_ms")
sexes = st.sidebar.multiselect("Sexe", sorted(df["sex"].unique()), default=sorted(df["sex"].unique()), key="sexes_ms")
smokers = st.sidebar.multiselect("Fumeur", sorted(df["smoker"].unique()), default=sorted(df["smoker"].unique()), key="smokers_ms")

mask = (
    (df["age"].between(age_range[0], age_range[1])) &
    (df["region"].isin(regions)) &
    (df["sex"].isin(sexes)) &
    (df["smoker"].isin(smokers))
)
df_filt = df.loc[mask].copy()


# PAGE 1 : Accueil

if page == "🏠 Accueil":
    st.title("Détection de biais – Medical Insurance Cost")
    st.caption("Parcours A – Application Streamlit (Pages 1 à 4)")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 Contexte & Problématique
        Jeu de données d'assurance santé (États‑Unis) : `age`, `sex`, `bmi`, `children`, `smoker`, `region`, **`charges`** (coût annuel).
        
        **Objectif** : Application interactive pour **explorer** les données et **détecter d’éventuels biais**
        sur un résultat binaire dérivé des charges (`high_cost`).
        """)

        st.markdown("""
        ### 🧪 Méthode
        - `high_cost = 1` si `charges ≥ seuil` (par défaut: **médiane**), sinon `0`.
        - Comparaison des **taux positifs** par groupes et mesures :
          **Parité démographique (écart)**, **Disparate Impact (ratio)**.
        - Modèle de classification (**Random Forest**) pour **prédire** `high_cost` et évaluer l’équité des **prédictions**.
        """)

    with col2:
        st.info("📄 Exigences : 4 pages, KPIs, 3+ charts, 2 métriques fairness, 1 modèle (Random Forest).", icon="ℹ️")

    #KPIs de base
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Nombre de lignes", f"{df.shape[0]:,}".replace(",", " "))
    k2.metric("Nombre de colonnes", f"{df.shape[1]}")
    missing_rate = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
    k3.metric("Valeurs manquantes (%)", f"{100*missing_rate:.2f}%")
    k4.metric("Seuil high_cost (€)", f"{threshold:,.0f}".replace(",", " "))

    st.markdown("### 👀 Aperçu des données")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 🧾 Description des colonnes")
    st.dataframe(describe_columns(df), use_container_width=True)

    st.caption("Structure et attendus issus des instructions (pages 1–4).")
    st.caption("Source des données : `insurance.csv`.")


# PAGE 2 : Exploration

elif page == "📊 Exploration des données":
    st.title("Exploration des données")

    #4 KPIs en colonnes
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes (filtré)", f"{df_filt.shape[0]:,}".replace(",", " "))
    c2.metric("Âge médian (filtré)", f"{df_filt['age'].median():.0f}")
    c3.metric("BMI moyen (filtré)", f"{df_filt['bmi'].mean():.2f}")
    c4.metric("Charges moyennes (filtré)", f"{df_filt['charges'].mean():,.0f}".replace(",", " "))

    #visualisation 1 : Distribution charges
    st.subheader("Distribution des charges")
    fig1 = px.histogram(df_filt, x="charges", nbins=40, color="smoker",
                        marginal="box", opacity=0.8,
                        title="Distribution de `charges` par statut fumeur")
    st.plotly_chart(fig1, use_container_width=True, key="hist_charges")

    #visualisation 2 : Comparaison par attribut sensible (taux high_cost par groupe)
    st.subheader("Taux de high_cost par groupe")
    attr = st.selectbox("Choisir l’attribut de comparaison", ["sex", "age_band", "smoker"], index=0, key="attr_explo")
    grp = df_filt.groupby(attr)["high_cost"].mean().reset_index().rename(columns={"high_cost": "taux_high_cost"})
    fig2 = px.bar(grp, x=attr, y="taux_high_cost", color=attr, text=(grp["taux_high_cost"]*100).round(1),
                  title=f"Taux de high_cost par {attr}", labels={"taux_high_cost": "Taux (0-1)"})
    fig2.update_traces(texttemplate="%{text}%")
    st.plotly_chart(fig2, use_container_width=True, key=f"bar_highcost_{attr}")

    #visualisation 3 : Box plot charges par groupe (ex: sexe & fumeur)
    st.subheader("Charges par groupe")
    fig3 = px.box(df_filt, x="sex", y="charges", color="smoker", points=False,
                  title="Distribution des charges par sexe et statut fumeur")
    st.plotly_chart(fig3, use_container_width=True, key="box_sex_smoker")

    #scatter charges vs BMI
    with st.expander("🔎 Relation charges vs BMI", expanded=False):
        fig_sc = px.scatter(df_filt, x="bmi", y="charges", color="smoker", facet_col="sex",
                            title="Charges en fonction du BMI (facetté par sexe)")
        st.plotly_chart(fig_sc, use_container_width=True, key="scatter_bmi_charges")


# PAGE 3 : Détection de biais

elif page == "⚠️ Détection de biais":
    st.title("Détection de biais")

    st.markdown("""
    #### Quel biais cherche-t-on ?
    Équité sur `high_cost` (charges ≥ seuil) pour des **attributs sensibles** :
    - **Sexe** (`sex`)
    - **Tranche d’âge** (`age_band`)
    - **Statut fumeur** (`smoker`)
    """)

    colA, colB = st.columns(2)

    with colA:
        target_attr = st.selectbox("Attribut sensible", ["sex", "age_band", "smoker"], index=0, key="attr_bias")
        rates = df.groupby(target_attr)["high_cost"].mean().reset_index().rename(columns={"high_cost": "taux+"})
        fig_rates = px.bar(rates, x=target_attr, y="taux+", color=target_attr,
                           title=f"Taux positifs (observés) par {target_attr}",
                           labels={"taux+": "Taux (0-1)"})
        st.plotly_chart(fig_rates, use_container_width=True, key=f"bar_obs_{target_attr}")

    with colB:
        #calcul des métriques de fairness (observées)
        if target_attr == "sex":
            diff, r = demographic_parity_difference(df["high_cost"], df["sex"])
            di, _ = disparate_impact_ratio(df["high_cost"], df["sex"])
        elif target_attr == "age_band":
            diff, r = demographic_parity_difference(df["high_cost"], df["age_band"])
            di, _ = disparate_impact_ratio(df["high_cost"], df["age_band"])
        else:
            diff, r = demographic_parity_difference(df["high_cost"], df["smoker"])
            di, _ = disparate_impact_ratio(df["high_cost"], df["smoker"])

        st.metric("Écart de parité démographique (observé)", f"{diff:.3f}")
        st.metric("Disparate Impact (observé)", f"{di:.3f}")
        st.json({"taux_par_groupe": {str(k): round(float(v), 3) for k, v in r.items()}}, expanded=False)

    st.markdown("""
    #### Interprétation (observé)
    - **Écart de parité** élevé → groupes n’ayant **pas la même probabilité** d’avoir `high_cost=1`.
    - **Disparate Impact** **proche de 1** → meilleure parité; **< 0.8** → **alerte potentielle**.
    - *Statut fumeur* n’est pas protégé légalement, mais structure fortement les coûts → à surveiller.
    """)


# PAGE 4 : Modélisation (Random Forest + UI stable)

else:
    st.title("Modélisation (classification de high_cost)")
    st.markdown("Modèle : **Random Forest** avec variables `age, bmi, children, sex, smoker, region`.")

    #containers stables pour éviter les remplacements de nœuds DOM
    c_intro = st.container()
    c_metrics = st.container()
    c_fairness = st.container()
    c_conf = st.container()

    with c_intro:
        #données + features/target (copie locale)
        df_mod = df.copy()
        X = df_mod.drop(columns=["charges", "high_cost"])
        y = df_mod["high_cost"]

        numeric_features = ["age", "bmi", "children"]
        categorical_features = ["sex", "smoker", "region"]

        #pipeline Random Forest
        clf = build_rf_clf(numeric_features, categorical_features)

        #split + entraînement
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        #conserver séries alignées pour la suite
        y_test_s = pd.Series(y_test).reset_index(drop=True)
        y_pred_s = pd.Series(y_pred).reset_index(drop=True)

    with c_metrics:
        #perf globales : colonnes fixes
        c1, c2, c3 = st.columns(3)
        c1.metric("Précision (Accuracy)", f"{accuracy_score(y_test_s, y_pred_s):.3f}")
        c2.metric("Précision (Precision)", f"{precision_score(y_test_s, y_pred_s):.3f}")
        c3.metric("Rappel (Recall)", f"{recall_score(y_test_s, y_pred_s):.3f}")

    with c_fairness:
        #fairness sur les prédictions
        st.subheader("Fairness sur les prédictions (taux+ par groupe)")
        attr_pred = st.selectbox(
            "Attribut sensible (prédictions)",
            ["sex", "age_band", "smoker"],
            index=0,
            key="attr_pred_select"
        )

        #récupérer l'attribut sensible du split test et aligner les index
        g_test = df.loc[y_test.index, attr_pred].reset_index(drop=True)

        rates_pred = (
            pd.DataFrame({"grp": g_test, "y_pred": y_pred_s})
            .groupby("grp", dropna=False)["y_pred"]
            .mean()
            .reset_index()
            .rename(columns={"grp": attr_pred, "y_pred": "taux_pred+"})
        )

        #bar chart avec une clé stable
        figp = px.bar(
            rates_pred, x=attr_pred, y="taux_pred+",
            color=attr_pred, title=f"Taux positifs prédits par {attr_pred}",
            labels={"taux_pred+": "Taux (0-1)"}
        )
        st.plotly_chart(figp, use_container_width=True, key=f"pred_rate_{attr_pred}")

        #métriques de fairness (prédictions)
        diff_p, r_p = demographic_parity_difference(y_pred_s, g_test)
        di_p, _ = disparate_impact_ratio(y_pred_s, g_test)

        cp1, cp2 = st.columns(2)
        cp1.metric("Écart de parité (préd.)", f"{diff_p:.3f}")
        cp2.metric("Disparate Impact (préd.)", f"{di_p:.3f}")
        st.json({"taux_pred_par_groupe": {str(k): round(float(v), 3) for k, v in r_p.items()}}, expanded=False)

    with c_conf:
        #confusion matrices par sexe : structure UI fixe (2 expanders) ---
        st.subheader("Confusion matrices par sexe (test)")
        g_sex = df.loc[y_test.index, "sex"].reset_index(drop=True)

        for g in ["female", "male"]:
            with st.expander(f"Sexe : {g}", expanded=(g == "female")):
                mask = (g_sex == g)
                if mask.any():
                    cm = confusion_matrix(y_test_s[mask], y_pred_s[mask], labels=[0, 1])
                    cm_df = pd.DataFrame(cm, index=["Vrai 0", "Vrai 1"], columns=["Prédit 0", "Prédit 1"])
                    st.dataframe(cm_df, use_container_width=True)
                else:
                    st.info("Aucune observation en test pour ce groupe.")