# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import calendar
from datetime import datetime

# Add authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if "password" not in st.session_state:
            st.session_state["password_correct"] = False
            return
        
        # Vérifier si les secrets sont configurés
        if "passwords" not in st.secrets:
            st.error("❌ Les secrets ne sont pas configurés. Veuillez configurer les secrets dans les paramètres de l'application.")
            st.session_state["password_correct"] = False
            return
            
        if st.session_state["password"] == st.secrets["passwords"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Fonction pour extraire l'année manuellement avec gestion des erreurs
def extract_year_manual(date_str):
    try:
        if isinstance(date_str, str) and date_str.strip():
            components = date_str.split()
            if len(components) >= 3:
                year_str = components[2]
                year = int(year_str)
                return year
        return None
    except Exception as e:
        return None

# Fonction pour extraire le mois manuellement avec gestion des erreurs
def extract_month_manual(date_str):
    try:
        if isinstance(date_str, str) and date_str.strip():
            components = date_str.split()
            if len(components) >= 2:
                month_str = components[1]
                return month_str.lower()
        return None
    except Exception as e:
        return None

# Charger et préparer les données
def load_data():
    # Vérifier si les secrets sont configurés
    if "tricount_data" not in st.secrets:
        st.error("❌ Les données ne sont pas configurées. Veuillez configurer les secrets dans les paramètres de l'application.")
        return pd.DataFrame()
    
    # Récupérer les données depuis les secrets
    import base64
    import io
    
    try:
        # Décoder les données base64
        csv_data = base64.b64decode(st.secrets["tricount_data"]["csv_data"])
        
        # Convertir en DataFrame
        df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
        
        # Traitement des données comme avant
        df['Date & heure'] = df['Date & heure'].astype(str)
        df['Année'] = df['Date & heure'].apply(extract_year_manual)
        df['Mois'] = df['Date & heure'].apply(extract_month_manual)

        mois_map = {
            'janvier': 1, 'février': 2, 'mars': 3,
            'avril': 4, 'mai': 5, 'juin': 6,
            'juillet': 7, 'août': 8, 'septembre': 9,
            'octobre': 10, 'novembre': 11, 'décembre': 12
        }

        df['Année'] = df['Année'].fillna(0).astype(int)
        df = df.dropna(subset=['Année', 'Mois'])
        df['Numéro Mois'] = df['Mois'].map(mois_map)
        df['Date'] = pd.to_datetime(
            df['Année'].astype(str) + '-' + 
            df['Numéro Mois'].astype(str) + '-01', 
            errors='coerce'
        )
        df = df.dropna(subset=['Date'])
        montant_cols = ['Montant', 'Montant dans la devise du tricount (CHF)', 'Payé par Caps', 'Payé par Emilian', 'Impacté à Caps', 'Impacté à Emilian']
        for col in montant_cols:
            if col in df.columns:
                df[col] = df[col].abs()
        columns = ["Date", "Année", "Mois", "Type de transaction", "Catégorie", "Impacté à Caps", "Impacté à Emilian"]
        df = df[columns]
        df = df.dropna(subset=['Date'])
        df = df.dropna(how='all')
        return df
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Configuration de la page Streamlit
st.set_page_config(page_title="Dashboard de Suivi des Finances", layout="wide")

# Sidebar pour les filtres
st.sidebar.header("Filtres")

# Création d'une colonne YearMonth pour faciliter le filtrage
df['YearMonth'] = df['Date'].dt.to_period('M')

# Récupération des périodes disponibles
available_periods = sorted(df['YearMonth'].unique())

# Sélection de la période
start_period = st.sidebar.selectbox(
    "Période de début",
    options=available_periods,
    index=0,
    format_func=lambda x: x.strftime('%B %Y')
)

end_period = st.sidebar.selectbox(
    "Période de fin",
    options=available_periods,
    index=len(available_periods)-1,
    format_func=lambda x: x.strftime('%B %Y')
)

# Filtrage des données selon la période sélectionnée
df_filtered = df[(df['YearMonth'] >= start_period) & (df['YearMonth'] <= end_period)]

# Sélection de la personne
personne = st.sidebar.selectbox("Filtrer par", options=["Caps", "Emilian", "Tous"])

# Calcul du nombre de mois dans la période sélectionnée
nb_mois = len(df_filtered['YearMonth'].unique())

# Affichage des indicateurs clés
st.title("Dashboard de Suivi des Finances")

# Calcul des dépenses moyennes mensuelles et des revenus mensuels
if personne == "Caps":
    depenses_mensuelles = df_filtered[df_filtered['Type de transaction'] == 'Dépense'].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Caps'].sum()
    revenus_mensuels = df_filtered[df_filtered['Type de transaction'] == "Entrée d'argent"].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Caps'].sum()
elif personne == "Emilian":
    depenses_mensuelles = df_filtered[df_filtered['Type de transaction'] == 'Dépense'].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Emilian'].sum()
    revenus_mensuels = df_filtered[df_filtered['Type de transaction'] == "Entrée d'argent"].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Emilian'].sum()
else:
    depenses_mensuelles = df_filtered[df_filtered['Type de transaction'] == 'Dépense'].groupby(df_filtered['Date'].dt.to_period('M'))[['Impacté à Caps', 'Impacté à Emilian']].sum().sum(axis=1)
    revenus_mensuels = df_filtered[df_filtered['Type de transaction'] == "Entrée d'argent"].groupby(df_filtered['Date'].dt.to_period('M'))[['Impacté à Caps', 'Impacté à Emilian']].sum().sum(axis=1)

# Calcul du taux d'épargne moyen mensuel
epargne_mensuelle = revenus_mensuels - depenses_mensuelles
epargne_moyenne_mensuelle = epargne_mensuelle.sum() / nb_mois

# Calcul du revenu mensuel moyen
revenu_mensuel_moyen = revenus_mensuels.sum() / nb_mois

# Calcul des dépenses moyennes mensuelles
depenses_mensuelles_moyen = depenses_mensuelles.sum() / nb_mois

# Calcul des pourcentages de dépenses fixes et variables par rapport au revenu
categories_fixes = ["Loyer & Charges", "Transport", "Courses", "Assurance"]

if personne == "Caps":
    depenses_fixes = df_filtered[(df_filtered['Catégorie'].isin(categories_fixes)) & 
                                (df_filtered['Type de transaction'] == 'Dépense')]['Impacté à Caps'].sum()
    depenses_variables = df_filtered[(~df_filtered['Catégorie'].isin(categories_fixes)) & 
                                   (df_filtered['Type de transaction'] == 'Dépense')]['Impacté à Caps'].sum()
elif personne == "Emilian":
    depenses_fixes = df_filtered[(df_filtered['Catégorie'].isin(categories_fixes)) & 
                                (df_filtered['Type de transaction'] == 'Dépense')]['Impacté à Emilian'].sum()
    depenses_variables = df_filtered[(~df_filtered['Catégorie'].isin(categories_fixes)) & 
                                   (df_filtered['Type de transaction'] == 'Dépense')]['Impacté à Emilian'].sum()
else:
    depenses_fixes = df_filtered[(df_filtered['Catégorie'].isin(categories_fixes)) & 
                                (df_filtered['Type de transaction'] == 'Dépense')][['Impacté à Caps', 'Impacté à Emilian']].sum().sum()
    depenses_variables = df_filtered[(~df_filtered['Catégorie'].isin(categories_fixes)) & 
                                   (df_filtered['Type de transaction'] == 'Dépense')][['Impacté à Caps', 'Impacté à Emilian']].sum().sum()

# Calcul des moyennes mensuelles
depenses_fixes_moyen = depenses_fixes / nb_mois
depenses_variables_moyen = depenses_variables / nb_mois

# Calcul des pourcentages par rapport au revenu mensuel moyen
pct_depenses_fixes = (depenses_fixes_moyen / revenu_mensuel_moyen) * 100
pct_depenses_variables = (depenses_variables_moyen / revenu_mensuel_moyen) * 100
taux_epargne_moyen = (epargne_moyenne_mensuelle / revenu_mensuel_moyen) * 100

# Affichage des métriques
col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenu Mensuel Moyen", f"{revenu_mensuel_moyen:.0f} CHF")
col2.metric("Dépenses Fixes (%)", f"{pct_depenses_fixes:.0f}%")
col3.metric("Dépenses Variables (%)", f"{pct_depenses_variables:.0f}%")
col4.metric("Taux d'Épargne Moyen", f"{taux_epargne_moyen:.0f}%")

# Affichage des montants moyens
col1, col2, col3, col4 = st.columns(4)
col1.metric("Épargne Moyenne", f"{epargne_moyenne_mensuelle:.0f} CHF")
col2.metric("Dépenses Fixes", f"{depenses_fixes_moyen:.0f} CHF")
col3.metric("Dépenses Variables", f"{depenses_variables_moyen:.0f} CHF")
col4.metric("Dépenses Totales", f"{depenses_mensuelles_moyen:.0f} CHF")

# Calcul de l'épargne cumulée et création du graphique de tendance d'épargne
epargne_cumulee = epargne_mensuelle.sum()

# Calcul du taux d'épargne mensuel
taux_epargne_mensuel = (epargne_mensuelle / revenus_mensuels) * 100

# Définition des objectifs d'épargne
objectif_emilian = 50  # 50% pour Emilian
objectif_caps = 35     # 35% pour Capucine

# Formatage des dates pour l'affichage
dates_index = taux_epargne_mensuel.index.to_timestamp()
dates_formatees = [d.strftime('%b %Y') for d in dates_index]  # Format court : 'Jan 2024'

# Création du graphique de tendance
fig_trend = go.Figure()

# Ajout de la ligne de tendance d'épargne (taux)
fig_trend.add_trace(go.Scatter(
    x=list(range(len(dates_formatees))),
    y=taux_epargne_mensuel.values,
    mode='lines+markers',
    name='Taux d\'épargne (%)',
    line=dict(color='blue', width=2),
    marker=dict(size=8),
    yaxis='y1',
    hovertemplate='%{text}<br>Taux: %{y:.1f}%<extra>Taux d\'épargne</extra>',
    text=dates_formatees
))

# Ajout de la ligne de tendance d'épargne (montants)
fig_trend.add_trace(go.Scatter(
    x=list(range(len(dates_formatees))),
    y=epargne_mensuelle.values,
    mode='lines',
    name='Montant épargné (CHF)',
    line=dict(color='red', width=2, dash='dot'),
    yaxis='y2',
    hovertemplate='%{text}<br>Montant: %{y:.0f} CHF<extra>Montant épargné</extra>',
    text=dates_formatees
))

# Ajout des lignes d'objectif selon la personne sélectionnée
if personne == "Emilian":
    fig_trend.add_shape(
        type="line",
        x0=-0.5,
        y0=objectif_emilian,
        x1=len(dates_formatees) - 0.5,
        y1=objectif_emilian,
        line=dict(
            color="green",
            width=2,
            dash="dash",
        ),
        name=f"Objectif Emilian ({objectif_emilian}%)",
        yref='y1'
    )
elif personne == "Caps":
    fig_trend.add_shape(
        type="line",
        x0=-0.5,
        y0=objectif_caps,
        x1=len(dates_formatees) - 0.5,
        y1=objectif_caps,
        line=dict(
            color="green",
            width=2,
            dash="dash",
        ),
        name=f"Objectif Capucine ({objectif_caps}%)",
        yref='y1'
    )
else:
    # Si "Tous" est sélectionné, afficher les deux objectifs
    fig_trend.add_shape(
        type="line",
        x0=-0.5,
        y0=objectif_emilian,
        x1=len(dates_formatees) - 0.5,
        y1=objectif_emilian,
        line=dict(
            color="green",
            width=2,
            dash="dash",
        ),
        name=f"Objectif Emilian ({objectif_emilian}%)",
        yref='y1'
    )
    fig_trend.add_shape(
        type="line",
        x0=-0.5,
        y0=objectif_caps,
        x1=len(dates_formatees) - 0.5,
        y1=objectif_caps,
        line=dict(
            color="purple",
            width=2,
            dash="dash",
        ),
        name=f"Objectif Capucine ({objectif_caps}%)",
        yref='y1'
    )

# Mise à jour du layout pour le graphique de tendance
fig_trend.update_layout(
    title=dict(
        text="Évolution du Taux d'Épargne (%) et des Montants (CHF)",
        y=0.95,
        x=0.5,
        xanchor='center',
        yanchor='top'
    ),
    xaxis_title="Mois",
    yaxis_title="Taux d'épargne (%)",
    showlegend=False,  # Suppression de la légende
    height=400,
    yaxis=dict(
        title="Taux d'épargne (%)",
        range=[min(0, taux_epargne_mensuel.min() * 1.1), max(taux_epargne_mensuel.max(), objectif_emilian, objectif_caps) * 1.1],
        side='left',
        gridcolor='lightgray',
        tickformat='.0f'  # Format sans décimales pour les pourcentages
    ),
    yaxis2=dict(
        title="Montant épargné (CHF)",
        range=[min(0, epargne_mensuelle.min() * 1.1), epargne_mensuelle.max() * 1.1],
        overlaying='y',
        side='right',
        gridcolor='lightgray',
        tickformat='.0f'  # Format sans décimales pour les montants
    ),
    xaxis=dict(
        tickangle=45,
        tickmode='array',
        tickvals=list(range(len(dates_formatees))),
        ticktext=dates_formatees,
        range=[-0.2, len(dates_formatees) - 0.8],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    margin=dict(
        l=50,
        r=50,
        t=80,  # Réduction de la marge supérieure
        b=100
    ),
    plot_bgcolor='white',
    hoverlabel=dict(bgcolor="white")
)



# Répartition des Dépenses par Catégorie (Moyenne Mensuelle)
depenses = df_filtered[df_filtered['Type de transaction'] == 'Dépense']

# Calculer la moyenne mensuelle des dépenses par catégorie
if personne == "Caps":
    depenses_par_categorie_moyenne = depenses.groupby(['Catégorie', depenses['Date'].dt.to_period('M')])['Impacté à Caps'].sum().reset_index()
    # Convertir en nombres pour éviter les erreurs de type
    depenses_par_categorie_moyenne['Impacté à Caps'] = pd.to_numeric(depenses_par_categorie_moyenne['Impacté à Caps'], errors='coerce')
    
    # Calculer la somme par catégorie et convertir en moyenne
    depenses_par_categorie_moyenne = depenses_par_categorie_moyenne.groupby('Catégorie')['Impacté à Caps'].sum().reset_index()
    depenses_par_categorie_moyenne['Total_Impact'] = depenses_par_categorie_moyenne['Impacté à Caps'] / nb_mois
    depenses_par_categorie_moyenne.drop(columns='Impacté à Caps', inplace=True)

elif personne == "Emilian":
    depenses_par_categorie_moyenne = depenses.groupby(['Catégorie', depenses['Date'].dt.to_period('M')])['Impacté à Emilian'].sum().reset_index()
    # Convertir en nombres pour éviter les erreurs de type
    depenses_par_categorie_moyenne['Impacté à Emilian'] = pd.to_numeric(depenses_par_categorie_moyenne['Impacté à Emilian'], errors='coerce')
    
    # Calculer la somme par catégorie et convertir en moyenne
    depenses_par_categorie_moyenne = depenses_par_categorie_moyenne.groupby('Catégorie')['Impacté à Emilian'].sum().reset_index()
    depenses_par_categorie_moyenne['Total_Impact'] = depenses_par_categorie_moyenne['Impacté à Emilian'] / nb_mois
    depenses_par_categorie_moyenne.drop(columns='Impacté à Emilian', inplace=True)

else:
    depenses_caps = depenses.groupby(['Catégorie', depenses['Date'].dt.to_period('M')])['Impacté à Caps'].sum().reset_index()
    depenses_emilian = depenses.groupby(['Catégorie', depenses['Date'].dt.to_period('M')])['Impacté à Emilian'].sum().reset_index()

    # Convertir en nombres pour éviter les erreurs de type
    depenses_caps['Impacté à Caps'] = pd.to_numeric(depenses_caps['Impacté à Caps'], errors='coerce')
    depenses_emilian['Impacté à Emilian'] = pd.to_numeric(depenses_emilian['Impacté à Emilian'], errors='coerce')

    # Fusionner les deux DataFrames pour obtenir une somme des deux impacts
    depenses_combined = pd.merge(depenses_caps, depenses_emilian, on=['Catégorie', 'Date'], how='outer', suffixes=('_Caps', '_Emilian'))
    depenses_combined.fillna(0, inplace=True)  # Remplacer les valeurs NaN par 0
    depenses_combined['Total_Impact'] = depenses_combined['Impacté à Caps'] + depenses_combined['Impacté à Emilian']
    
    # Calculer la somme totale par catégorie et convertir en moyenne
    depenses_par_categorie_moyenne = depenses_combined.groupby('Catégorie')['Total_Impact'].sum().reset_index()
    depenses_par_categorie_moyenne['Total_Impact'] = depenses_par_categorie_moyenne['Total_Impact'] / nb_mois

# Ajouter une colonne de pourcentage
total_dépenses_moyenne = depenses_par_categorie_moyenne['Total_Impact'].sum()
depenses_par_categorie_moyenne['Pourcentage'] = (depenses_par_categorie_moyenne['Total_Impact'] / total_dépenses_moyenne) * 100

# Créer le camembert des dépenses moyennes mensuelles par catégorie
fig_pie = px.pie(
    depenses_par_categorie_moyenne,
    names='Catégorie',
    values='Total_Impact',
    title="Répartition des Dépenses Mensuelles par Catégorie"
)


# Mettre à jour les étiquettes du graphique pour n'afficher que les valeurs >= 1%
fig_pie.update_traces(
    textinfo='label+percent',
    texttemplate='%{label}: %{percent:.0f}%',
    insidetextorientation='radial'
)

# Filtrer les segments avec moins de 1% pour les masquer
fig_pie.update_traces(
    texttemplate=depenses_par_categorie_moyenne.apply(
        lambda x: f"{x['Pourcentage']:.0f}%" if x['Pourcentage'] >= 1 else "",
        axis=1
    )
)

# Mettre à jour le graphique pour masquer les petites catégories
fig_pie.update_traces(
    textinfo='label+percent'
)


# Graphique : Revenus et Dépenses par Mois
df_combined = pd.DataFrame({
    'Mois': depenses_mensuelles.index.to_timestamp(),
    'Dépenses': depenses_mensuelles.values,
    'Revenus': revenus_mensuels.reindex(depenses_mensuelles.index).values
}).reset_index(drop=True)

# Formater les dates en "Mois - Année"
df_combined['Mois'] = df_combined['Mois'].dt.strftime('%B - %Y')

fig_bar = px.bar(
    df_combined,
    x='Mois',
    y=['Dépenses', 'Revenus'],
    title="Revenus et Dépenses par Mois",
    labels={'value': 'Montant', 'Mois': 'Mois'},
    barmode='group'
)


# Correction de la partie où vous calculez les moyennes pour les charges variables
charges_variables = depenses[~depenses['Catégorie'].isin(['Loyer & Charges', 'Transport', 'Courses', 'Assurance'])]

# Convertir 'Date' en format périodique pour le groupement
charges_variables['Date'] = charges_variables['Date'].dt.to_period('M')

# Préparation des données pour le graphique en barres empilées
if personne == "Caps":
    charges_variables_par_mois = charges_variables.pivot_table(
        index='Date',
        columns='Catégorie',
        values='Impacté à Caps',
        aggfunc='sum',
        fill_value=0
    )
elif personne == "Emilian":
    charges_variables_par_mois = charges_variables.pivot_table(
        index='Date',
        columns='Catégorie',
        values='Impacté à Emilian',
        aggfunc='sum',
        fill_value=0
    )
else:
    # Pour "Tous", on somme les deux colonnes
    charges_variables_par_mois = charges_variables.pivot_table(
        index='Date',
        columns='Catégorie',
        values=['Impacté à Caps', 'Impacté à Emilian'],
        aggfunc='sum',
        fill_value=0
    )
    # Somme des deux colonnes pour chaque catégorie
    charges_variables_par_mois = charges_variables_par_mois.groupby(level=1, axis=1).sum()

# Trier les catégories par montant total décroissant
totaux_par_categorie = charges_variables_par_mois.sum()
categories_triees = totaux_par_categorie.sort_values(ascending=False).index[:5]

# Filtrer pour ne garder que les 5 principales catégories
charges_variables_par_mois = charges_variables_par_mois[categories_triees]

# Convertir les dates en format lisible
dates_str = [d.strftime('%b %Y') for d in charges_variables_par_mois.index.to_timestamp()]

# Création du graphique en barres empilées
fig_stacked = go.Figure()

# Ajouter chaque catégorie comme une couche de barres
for categorie in categories_triees:
    fig_stacked.add_trace(go.Bar(
        name=categorie,
        x=dates_str,
        y=charges_variables_par_mois[categorie],
        hovertemplate="%{x}<br>" + categorie + ": %{y:.0f} CHF<extra></extra>"
    ))

# Mise à jour du layout
fig_stacked.update_layout(
    title="Évolution des Principales Dépenses Variables par Catégorie",
    xaxis_title="Mois",
    yaxis_title="Montant (CHF)",
    barmode='stack',
    showlegend=True,
    height=400,
    xaxis=dict(
        tickangle=45,
        type='category'
    ),
    yaxis=dict(
        gridcolor='lightgray'
    ),
    plot_bgcolor='white',
    bargap=0.2
)


# Préparation des données pour le graphique en barres 100%
if personne == "Caps":
    repartition_mensuelle = pd.DataFrame({
        'Date': depenses_mensuelles.index,
        'Dépenses Fixes': df_filtered[
            (df_filtered['Catégorie'].isin(categories_fixes)) & 
            (df_filtered['Type de transaction'] == 'Dépense')
        ].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Caps'].sum(),
        'Dépenses Variables': df_filtered[
            (~df_filtered['Catégorie'].isin(categories_fixes)) & 
            (df_filtered['Type de transaction'] == 'Dépense')
        ].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Caps'].sum(),
        'Épargne': epargne_mensuelle
    })
elif personne == "Emilian":
    repartition_mensuelle = pd.DataFrame({
        'Date': depenses_mensuelles.index,
        'Dépenses Fixes': df_filtered[
            (df_filtered['Catégorie'].isin(categories_fixes)) & 
            (df_filtered['Type de transaction'] == 'Dépense')
        ].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Emilian'].sum(),
        'Dépenses Variables': df_filtered[
            (~df_filtered['Catégorie'].isin(categories_fixes)) & 
            (df_filtered['Type de transaction'] == 'Dépense')
        ].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Emilian'].sum(),
        'Épargne': epargne_mensuelle
    })
else:
    repartition_mensuelle = pd.DataFrame({
        'Date': depenses_mensuelles.index,
        'Dépenses Fixes': df_filtered[
            (df_filtered['Catégorie'].isin(categories_fixes)) & 
            (df_filtered['Type de transaction'] == 'Dépense')
        ].groupby(df_filtered['Date'].dt.to_period('M'))[['Impacté à Caps', 'Impacté à Emilian']].sum().sum(axis=1),
        'Dépenses Variables': df_filtered[
            (~df_filtered['Catégorie'].isin(categories_fixes)) & 
            (df_filtered['Type de transaction'] == 'Dépense')
        ].groupby(df_filtered['Date'].dt.to_period('M'))[['Impacté à Caps', 'Impacté à Emilian']].sum().sum(axis=1),
        'Épargne': epargne_mensuelle
    })

# Convertir les dates en format lisible
repartition_mensuelle['Date'] = repartition_mensuelle['Date'].astype(str)

# Calculer les pourcentages
colonnes = ['Dépenses Fixes', 'Dépenses Variables', 'Épargne']
total_mensuel = repartition_mensuelle[colonnes].sum(axis=1)
for colonne in colonnes:
    repartition_mensuelle[f'{colonne} (%)'] = (repartition_mensuelle[colonne] / total_mensuel) * 100

# Création du graphique en barres 100%
fig_repartition = go.Figure()

# Ajouter les barres pour chaque catégorie
for colonne in colonnes:
    fig_repartition.add_trace(go.Bar(
        name=colonne,
        x=repartition_mensuelle['Date'],
        y=repartition_mensuelle[f'{colonne} (%)'],
        hovertemplate="%{x}<br>" + 
                     colonne + ": %{y:.1f}%<br>" +
                     "Montant: %{customdata:.0f} CHF<extra></extra>",
        customdata=repartition_mensuelle[colonne]
    ))

# Mise à jour du layout
fig_repartition.update_layout(
    title="Répartition Mensuelle des Dépenses et de l'Épargne",
    yaxis_title="Pourcentage",
    barmode='relative',
    barnorm='percent',
    showlegend=True,
    height=400,
    xaxis=dict(
        title="Mois",
        tickangle=45,
        type='category'
    ),
    yaxis=dict(
        title="Répartition (%)",
        gridcolor='lightgray'
    ),
    plot_bgcolor='white',
    bargap=0.2
)

# Ajout des colonnes pour les graphiques
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    st.plotly_chart(fig_trend, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig_bar, use_container_width=True)
with col4:
    st.plotly_chart(fig_stacked, use_container_width=True)

# Ajout du nouveau graphique en pleine largeur
st.plotly_chart(fig_repartition, use_container_width=True)
