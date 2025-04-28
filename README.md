# Dashboard de Suivi des Finances

Une application Streamlit pour suivre et visualiser vos dépenses et revenus, avec une interface intuitive et des visualisations interactives.

## Fonctionnalités

- 🔐 **Authentification sécurisée** : Accès protégé à vos données financières
- 📊 **Visualisations interactives** :
  - Graphique en camembert des dépenses par catégorie
  - Graphique de tendance d'épargne
  - Graphique des revenus et dépenses par mois
  - Graphique des principales dépenses variables
  - Graphique de répartition des dépenses et de l'épargne
- 📅 **Filtres personnalisables** :
  - Sélection de période (début et fin)
  - Filtrage par personne
- 📈 **Indicateurs clés** :
  - Revenu mensuel moyen
  - Taux d'épargne moyen
  - Dépenses mensuelles moyennes
  - Taux de dépenses fixes et variables
  - Épargne mensuelle moyenne

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-username/git_spending_tracker.git
cd git_spending_tracker
```

2. Créez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurez les secrets :
- Créez un fichier `.streamlit/secrets.toml`
- Ajoutez vos données d'authentification

5. Lancez l'application :
```bash
streamlit run streamlit_app.py
```

## Structure du Projet

```
git_spending_tracker/
├── .streamlit/
│   └── secrets.toml
├── streamlit_app.py
├── requirements.txt
├── runtime.txt
├── render.yaml
└── README.md
```

## Déploiement

L'application est déployée sur Render. Pour mettre à jour :

1. Poussez vos modifications sur GitHub
2. Render détectera automatiquement les changements
3. Un nouveau déploiement sera lancé

## Sécurité

- Les données sensibles sont stockées de manière sécurisée
- L'accès est protégé par authentification
- Le fichier CSV n'est pas inclus dans le dépôt Git

## Technologies Utilisées

- Python 3.10
- Streamlit
- Pandas
- Plotly
- Render (hébergement)

## Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Poussez vers la branche
5. Ouvrez une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub. 