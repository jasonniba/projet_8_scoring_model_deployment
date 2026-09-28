# Projet 8 - Mise en production d'un modèle de Credit Scoring

## Contexte

Ce projet s'inscrit dans la continuité du projet précédent
"Initiez-vous au MLOps".

Le modèle de Credit Scoring développé et versionné avec MLflow lors
du projet précédent est repris afin de construire une solution complète
de mise en production.

L'objectif est de permettre à l'entreprise fictive "Prêt à Dépenser"
d'utiliser le modèle de scoring pour traiter de nouvelles demandes de
crédit en quasi temps réel.

## Objectifs du projet

Le projet comprend :

- la mise en place d'une API de prédiction ;
- des tests unitaires automatisés ;
- la conteneurisation avec Docker ;
- un pipeline CI/CD ;
- le stockage et le monitoring des données de production ;
- la détection du Data Drift ;
- un tableau de bord de monitoring ;
- l'analyse des performances de l'API et du modèle ;
- l'optimisation du système avant/après tests de charge.

## Modèle utilisé

Le modèle repris du projet précédent est un pipeline Scikit-learn
enregistré avec MLflow.

Il contient notamment :

- un `SimpleImputer` avec stratégie médiane ;
- un `RandomForestClassifier` ;
- 401 variables d'entrée.

Le modèle est stocké dans :

`models/credit_scoring_model/`

## Structure actuelle

```text
Projet_8/
├── data/
├── models/
│   └── credit_scoring_model/
├── notebooks/
│   └── Outil de scoring Projet 6 Notebook.ipynb
├── src/
├── tests/
├── .gitignore
└── README.md