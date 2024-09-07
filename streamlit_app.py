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

# Affichage des indicateurs clés
st.title("Dashboard de Suivi des Dépenses")

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
taux_epargne_moyen = epargne_mensuelle.mean() / revenus_mensuels.mean() * 100 if revenus_mensuels.mean() > 0 else 0

# Calcul du revenu mensuel moyen
revenu_mensuel_moyen = revenus_mensuels.mean()

# Affichage des métriques
col1, col2, col3 = st.columns(3)
col1.metric("Dépense Moyenne Mensuelle", f"{depenses_mensuelles.mean():.2f} CHF")
col2.metric("Revenu Mensuel Moyen", f"{revenu_mensuel_moyen:.2f} CHF")
col3.metric("Taux d'Épargne Moyen Mensuel", f"{taux_epargne_moyen:.2f} %")

# Répartition des Dépenses par Catégorie (Moyenne Mensuelle)
depenses = df_filtered[df_filtered['Type de transaction'] == 'Dépense']
if personne == "Caps":
    depenses_par_categorie_moyenne = depenses.groupby('Catégorie')['Impacté à Caps'].mean().reset_index()
elif personne == "Emilian":
    depenses_par_categorie_moyenne = depenses.groupby('Catégorie')['Impacté à Emilian'].mean().reset_index()
else:
    depenses_par_categorie_moyenne = depenses.groupby('Catégorie')[['Impacté à Caps', 'Impacté à Emilian']].mean().sum(axis=1).reset_index()

# Créer le camembert des dépenses moyennes mensuelles par catégorie
fig_pie = px.pie(
    depenses_par_categorie_moyenne,
    names='Catégorie',
    values=depenses_par_categorie_moyenne.columns[1],  # Utilise la colonne correcte selon le filtre appliqué (Caps ou Emilian)
    title="Répartition des Dépenses Moyennes Mensuelles par Catégorie"
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
    total_charges = charges_variables_moyenne.groupby('Catégorie')['Impacté à Caps'].sum()
elif personne == "Emilian":
    total_charges = charges_variables_moyenne.groupby('Catégorie')['Impacté à Emilian'].sum()
else:
    total_charges = charges_variables_moyenne.groupby('Catégorie')[['Impacté à Caps', 'Impacté à Emilian']].sum().sum(axis=1)

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

# Créer le graphique
fig_line = px.line(
    charges_top5,
    x='Date',
    y=charges_top5.columns[2],  # Utilisez la colonne correcte selon le filtre appliqué (Caps ou Emilian)
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


# Organisation des graphiques
st.plotly_chart(fig_pie, use_container_width=True)
st.plotly_chart(fig_bar, use_container_width=True)
st.plotly_chart(fig_line, use_container_width=True)
