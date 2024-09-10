# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    data = pd.read_csv('Tricount_Switzerland.csv')
    data['Date & heure'] = data['Date & heure'].astype(str)
    data['Année'] = data['Date & heure'].apply(extract_year_manual)
    data['Mois'] = data['Date & heure'].apply(extract_month_manual)

    mois_map = {
        'janvier': 1, 'février': 2, 'mars': 3,
        'avril': 4, 'mai': 5, 'juin': 6,
        'juillet': 7, 'août': 8, 'septembre': 9,
        'octobre': 10, 'novembre': 11, 'décembre': 12
    }

    data['Année'] = data['Année'].fillna(0).astype(int)
    data = data.dropna(subset=['Année', 'Mois'])
    data['Numéro Mois'] = data['Mois'].map(mois_map)
    data['Date'] = pd.to_datetime(
        data['Année'].astype(str) + '-' + 
        data['Numéro Mois'].astype(str) + '-01', 
        errors='coerce'
    )
    data = data.dropna(subset=['Date'])
    montant_cols = ['Montant', 'Montant dans la devise du tricount (CHF)', 'Payé par Caps', 'Payé par Emilian', 'Impacté à Caps', 'Impacté à Emilian']
    for col in montant_cols:
        if col in data.columns:
            data[col] = data[col].abs()
    columns = ["Date", "Année", "Mois", "Type de transaction", "Catégorie", "Impacté à Caps", "Impacté à Emilian"]
    df = data[columns]
    df = df.dropna(subset=['Date'])
    df = df.dropna(how='all')
    return df

df = load_data()

# Configuration de la page Streamlit
st.set_page_config(page_title="Dashboard de Suivi des Finances", layout="wide")

# Sidebar pour les filtres
st.sidebar.header("Filtres")
mois = st.sidebar.multiselect("Mois", options=df['Mois'].unique(), default=df['Mois'].unique())
annee = st.sidebar.multiselect("Année", options=df['Année'].unique(), default=df['Année'].unique())
personne = st.sidebar.selectbox("Filtrer par", options=["Caps", "Emilian", "Tous"])

# Filtrer les données en fonction des filtres sélectionnés
if personne == "Tous":
    df_filtered = df[(df['Mois'].isin(mois)) & (df['Année'].isin(annee))]
else:
    df_filtered = df[(df['Mois'].isin(mois)) & (df['Année'].isin(annee))]
    df_filtered['Montant'] = df_filtered[f'Impacté à {personne}']

nb_mois = len(df_filtered['Mois'].unique())

# Affichage des indicateurs clés
st.title("Dashboard de Suivi des Finances")

# Calcul des dépenses moyennes mensuelles et des revenus mensuels
if personne == "Caps":
    depenses_mensuelles = df_filtered[df_filtered['Type de transaction'] == 'Dépense'].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Caps'].sum()
    revenus_mensuels = df_filtered[df_filtered['Type de transaction'] == "Rentrée d'argent"].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Caps'].sum()
elif personne == "Emilian":
    depenses_mensuelles = df_filtered[df_filtered['Type de transaction'] == 'Dépense'].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Emilian'].sum()
    revenus_mensuels = df_filtered[df_filtered['Type de transaction'] == "Rentrée d'argent"].groupby(df_filtered['Date'].dt.to_period('M'))['Impacté à Emilian'].sum()
else:
    depenses_mensuelles = df_filtered[df_filtered['Type de transaction'] == 'Dépense'].groupby(df_filtered['Date'].dt.to_period('M'))[['Impacté à Caps', 'Impacté à Emilian']].sum().sum(axis=1)
    revenus_mensuels = df_filtered[df_filtered['Type de transaction'] == "Rentrée d'argent"].groupby(df_filtered['Date'].dt.to_period('M'))[['Impacté à Caps', 'Impacté à Emilian']].sum().sum(axis=1)

# Calcul du taux d'épargne moyen mensuel
epargne_mensuelle = revenus_mensuels - depenses_mensuelles
epargne_moyenne_mensuelle = epargne_mensuelle.sum() / nb_mois

# Calcul du revenu mensuel moyen
revenu_mensuel_moyen = revenus_mensuels.sum() / nb_mois
taux_epargne_moyen = (epargne_moyenne_mensuelle / revenu_mensuel_moyen) * 100

# Calcul des dépenses moyennes mensuelles
depenses_mensuelles_moyen = depenses_mensuelles.sum() / nb_mois

# Affichage des métriques
col1, col2, col3, col4 = st.columns(4)
#col1.metric("Revenu Mensuel Moyen", f"{revenu_mensuel_moyen:.0f} CHF")
col2.metric("Dépense Moyenne Mensuelle", f"{depenses_mensuelles_moyen:.0f} CHF")
col3.metric("Taux d'Épargne Moyen Mensuel", f"{taux_epargne_moyen:.0f} %")
col4.metric("Epargne Moyenne", f"{epargne_moyenne_mensuelle:.0f} CHF")



# Calcul du pourcentage des charges fixes sur le revenu
categories_fixes = ["Loyer & Charges", "Transport", "Courses", "Assurance"]
    
# Calcul des pourcentages des charges fixes et variables sur le revenu en prenant en compte les filtres
if personne == "Caps":
    charges_fixes = df_filtered[(df_filtered['Catégorie'].isin(categories_fixes)) & (df_filtered['Type de transaction'] == 'Dépense')]
    total_charges_fixes = charges_fixes.groupby(charges_fixes['Date'].dt.to_period('M'))['Impacté à Caps'].sum()
    charges_variables = df_filtered[(~df_filtered['Catégorie'].isin(categories_fixes)) & (df_filtered['Type de transaction'] == 'Dépense')]
    total_charges_variables = charges_variables.groupby(charges_variables['Date'].dt.to_period('M'))['Impacté à Caps'].sum()
elif personne == "Emilian":
    charges_fixes = df_filtered[(df_filtered['Catégorie'].isin(categories_fixes)) & (df_filtered['Type de transaction'] == 'Dépense')]
    total_charges_fixes = charges_fixes.groupby(charges_fixes['Date'].dt.to_period('M'))['Impacté à Emilian'].sum()
    charges_variables = df_filtered[(~df_filtered['Catégorie'].isin(categories_fixes)) & (df_filtered['Type de transaction'] == 'Dépense')]
    total_charges_variables = charges_variables.groupby(charges_variables['Date'].dt.to_period('M'))['Impacté à Emilian'].sum()
else:
    charges_fixes = df_filtered[(df_filtered['Catégorie'].isin(categories_fixes)) & (df_filtered['Type de transaction'] == 'Dépense')]
    total_charges_fixes = charges_fixes.groupby(charges_fixes['Date'].dt.to_period('M'))[['Impacté à Caps', 'Impacté à Emilian']].sum().sum(axis=1)
    charges_variables = df_filtered[(~df_filtered['Catégorie'].isin(categories_fixes)) & (df_filtered['Type de transaction'] == 'Dépense')]
    total_charges_variables = charges_variables.groupby(charges_variables['Date'].dt.to_period('M'))[['Impacté à Caps', 'Impacté à Emilian']].sum().sum(axis=1)

# Calcul des pourcentages
pourcentage_charges_fixes = (total_charges_fixes.sum() / revenus_mensuels.sum()) * 100 if revenus_mensuels.sum() > 0 else 0
pourcentage_charges_variables = (total_charges_variables.sum() / revenus_mensuels.sum()) * 100 if revenus_mensuels.sum() > 0 else 0


# Calcul de l'épargne cumulée et création de la jauge pour l'objectif annuel d'épargne
epargne_cumulee = epargne_mensuelle.sum()
objectif_annuel = 5000  # Objectif d'épargne annuel de 5K CHF

# Calcul du seuil pour être en ligne avec l'objectif à la date actuelle
import datetime
date_actuelle = datetime.datetime.now()
debut_annee = datetime.datetime(date_actuelle.year, 1, 1)
jours_ecoules = (date_actuelle - debut_annee).days
total_jours_annee = 365
objectif_a_date = (jours_ecoules / total_jours_annee) * objectif_annuel

objectif_MTD = (epargne_cumulee - objectif_a_date)*100 / objectif_a_date


# Affichage des autres métriques
col1.metric("% Charges Fixes", f"{pourcentage_charges_fixes:.0f} %")
col2.metric("% Charges Variables", f"{pourcentage_charges_variables:.0f} %")
col3.metric("Objectif Epargne MTD", f"{objectif_MTD:.0f} %")



# Création de la jauge pour montrer l'épargne cumulée par rapport à l'objectif
fig_gauge_epargne = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=epargne_cumulee,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Épargne Cumulée par rapport à l'Objectif Annuel"},
    delta={'reference': objectif_a_date, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
    gauge={
        'axis': {'range': [0, objectif_annuel], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, objectif_a_date], 'color': "lightgray"},
            {'range': [objectif_a_date, objectif_annuel], 'color': "lightgreen"}
        ],
        'threshold': {
            'line': {'color': "blue", 'width': 4},
            'thickness': 0.75,
            'value': objectif_a_date
        }
    },
    number={'suffix': " CHF"}
))

fig_gauge_epargne.update_layout(
    height=350
)




# Répartition des Dépenses par Catégorie (Moyenne Mensuelle)
depenses = df_filtered[df_filtered['Type de transaction'] == 'Dépense']

# Calculer la moyenne mensuelle des dépenses par catégorie
if personne == "Caps":
    depenses_par_categorie_moyenne = depenses.groupby(['Catégorie', depenses['Date'].dt.to_period('M')])['Impacté à Caps'].sum().reset_index()
    # Calculer la moyenne par catégorie
    depenses_par_categorie_moyenne = depenses_par_categorie_moyenne.groupby('Catégorie')['Impacté à Caps'].sum().reset_index()
    depenses_par_categorie_moyenne/=nb_mois
    depenses_par_categorie_moyenne.rename(columns={'Impacté à Caps': 'Total_Impact'}, inplace=True)
elif personne == "Emilian":
    depenses_par_categorie_moyenne = depenses.groupby(['Catégorie', depenses['Date'].dt.to_period('M')])['Impacté à Emilian'].sum().reset_index()
    # Calculer la moyenne par catégorie
    depenses_par_categorie_moyenne = depenses_par_categorie_moyenne.groupby('Catégorie')['Impacté à Emilian'].sum().reset_index()
    depenses_par_categorie_moyenne/=nb_mois
    depenses_par_categorie_moyenne.rename(columns={'Impacté à Emilian': 'Total_Impact'}, inplace=True)
else:
    depenses_caps = depenses.groupby(['Catégorie', depenses['Date'].dt.to_period('M')])['Impacté à Caps'].sum().reset_index()
    depenses_emilian = depenses.groupby(['Catégorie', depenses['Date'].dt.to_period('M')])['Impacté à Emilian'].sum().reset_index()

    # Fusionner les deux DataFrames pour obtenir une somme des deux impacts
    depenses_combined = pd.merge(depenses_caps, depenses_emilian, on=['Catégorie', 'Date'], suffixes=('_Caps', '_Emilian'))
    depenses_combined['Total_Impact'] = depenses_combined['Impacté à Caps'] + depenses_combined['Impacté à Emilian']
    
    # Calculer la moyenne mensuelle des dépenses par catégorie
    depenses_par_categorie_moyenne = depenses_combined.groupby('Catégorie')['Total_Impact'].sum().reset_index()
    depenses_par_categorie_moyenne/=nb_mois

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
charges_variables = depenses[~depenses['Catégorie'].isin(['Loyer & Charges', 'Transport', 'Courses'])]

# Convertir 'Date' en format périodique pour éviter les opérations non supportées
charges_variables['Date'] = charges_variables['Date'].dt.to_period('M')

if personne == "Caps":
    charges_variables_moyenne = charges_variables.groupby(['Date', 'Catégorie']).agg({
        'Impacté à Caps': 'sum'  # Utilisez 'sum' pour obtenir les totaux mensuels
    }).reset_index()
elif personne == "Emilian":
    charges_variables_moyenne = charges_variables.groupby(['Date', 'Catégorie']).agg({
        'Impacté à Emilian': 'sum'  # Utilisez 'sum' pour obtenir les totaux mensuels
    }).reset_index()
else:
    charges_variables_moyenne = charges_variables.groupby(['Date', 'Catégorie']).agg({
        'Impacté à Caps': 'sum',
        'Impacté à Emilian': 'sum'
    }).reset_index()

# Convertir la période en format datetime pour la visualisation
charges_variables_moyenne['Date'] = charges_variables_moyenne['Date'].dt.to_timestamp()
# Calculer la somme totale par catégorie pour identifier les 5 principales
if personne == "Caps":
    charges_variables_moyenne['Total_Impact'] = charges_variables_moyenne['Impacté à Caps']
elif personne == "Emilian":
    charges_variables_moyenne['Total_Impact'] = charges_variables_moyenne['Impacté à Emilian']
else:
    # Ajouter les deux colonnes pour obtenir le Total_Impact
    charges_variables_moyenne['Total_Impact'] = (
        charges_variables_moyenne['Impacté à Caps'] +
        charges_variables_moyenne['Impacté à Emilian']
    )

total_charges = charges_variables_moyenne.groupby('Catégorie')['Total_Impact'].sum()

# Identifier les 5 catégories avec les dépenses les plus élevées
top5_categories = total_charges.nlargest(5).index

# Filtrer les données pour ne conserver que les 5 catégories principales
charges_top5 = charges_variables_moyenne[charges_variables_moyenne['Catégorie'].isin(top5_categories)]

# Convertir 'Date' en format de chaîne de caractères pour les mois et années
charges_top5['Date'] = charges_top5['Date'].dt.to_period('M').astype(str)

# Définir l'ordre des mois et années pour le tri
charges_top5['Date'] = pd.Categorical(
    charges_top5['Date'],
    categories=sorted(charges_top5['Date'].unique(), key=lambda x: pd.to_datetime(x, format='%Y-%m')),
    ordered=True
)

# Création du graphique linéaire
fig_line = px.line(
    charges_top5,
    x='Date',
    y='Total_Impact',
    color='Catégorie',
    title="Évolution des 5 Premiers Postes des Charges Variables Moyennes",
    labels={'value': 'Montant', 'Date': 'Mois - Année'}
)

# Personnaliser le format d'affichage des dates sur l'axe des x
fig_line.update_xaxes(
    tickmode='array',
    tickvals=charges_top5['Date'].cat.categories,
    ticktext=[pd.to_datetime(date, format='%Y-%m').strftime('%B - %Y') for date in charges_top5['Date'].cat.categories]
)


# Ligne 1 : fig_pie et fig_gauge_epargne
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    st.plotly_chart(fig_gauge_epargne, use_container_width=True)

# Ligne 2 : fig_bar et fig_line
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig_bar, use_container_width=True)
with col4:
    st.plotly_chart(fig_line, use_container_width=True)
