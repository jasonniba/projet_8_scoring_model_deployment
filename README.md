# Credit Scoring Model Deployment

Projet de mise en production d'un modèle de scoring de crédit réalisé dans le cadre du parcours Data Scientist OpenClassrooms.

L'objectif du projet est de déployer un modèle de Machine Learning derrière une API, de mettre en place son monitoring, de détecter le data drift et d'évaluer puis optimiser ses performances dans un environnement de production simulée.

---

# 1. Architecture du projet

```text
Projet_8/
│
├── .github/
│   └── workflows/
│
├── dashboard/
│   └── app.py
│
├── data/
│
├── models/
│   └── credit_scoring_model/
│
├── notebooks/
│   └── Outil de scoring Projet 6 Notebook .ipynb
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── db/
│   │   └── database.py
│   │
│   ├── monitoring/
│   │   ├── build_reference.py
│   │   ├── build_current.py
│   │   ├── logger.py
│   │   ├── run_drift.py
│   │   ├── simulate_production.py
│   │   └── store_drift_results.py
│   │
│   └── performance/
│       ├── baseline.py
│       ├── predictive_baseline.py
│       ├── profile_api.py
│       └── load_test.py
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

# 2. Modèle de scoring

Le modèle utilisé est un pipeline Scikit-learn enregistré avec MLflow.

Il contient :

- un `SimpleImputer(strategy="median")`
- un `RandomForestClassifier`
- 401 variables d'entrée
- une classification binaire

Les classes prédites sont :

- `0` : absence de défaut
- `1` : risque de défaut

Le modèle provient du projet précédent de scoring de crédit réalisé à partir des données Home Credit.

---

# 3. API FastAPI

Le modèle est exposé grâce à une API développée avec FastAPI.

L'API contient trois routes principales :

```text
GET  /
GET  /health
POST /predict
```

## Route `/`

Permet de vérifier que l'API est accessible.

## Route `/health`

Permet de vérifier que le modèle est correctement chargé.

Exemple de réponse :

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "Pipeline",
  "expected_features": 401
}
```

## Route `/predict`

La route `/predict` reçoit les 401 variables attendues par le modèle.

Exemple de réponse :

```json
{
  "prediction": 0,
  "default_probability": 0.24,
  "inference_time_ms": 15.2,
  "request_duration_ms": 18.4
}
```

L'API vérifie également que les variables reçues correspondent exactement aux variables attendues par le modèle.

Une requête contenant des variables manquantes ou inconnues retourne une erreur HTTP 422.

---

# 4. Lancement local

## Création de l'environnement virtuel

```powershell
python -m venv .venv
```

Activation sous PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

## Installation des dépendances

```powershell
pip install -r requirements.txt
```

## Lancement de l'API

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

L'API est ensuite disponible à l'adresse :

```text
http://127.0.0.1:8000
```

Documentation Swagger :

```text
http://127.0.0.1:8000/docs
```

Health check :

```text
http://127.0.0.1:8000/health
```

---

# 5. Tests automatisés

Les tests de l'API sont réalisés avec Pytest.

Pour les lancer :

```powershell
pytest -q
```

Les tests contrôlent notamment :

- le fonctionnement du health check
- une prédiction avec les 401 variables attendues
- le rejet d'une requête contenant des variables manquantes

Les tests permettent de détecter les régressions lors des modifications de l'API.

---

# 6. Docker

L'API peut être conteneurisée avec Docker.

## Construction de l'image

```powershell
docker build -t credit-scoring-api .
```

## Lancement du conteneur

```powershell
docker run -p 8000:8000 credit-scoring-api
```

L'application est alors accessible sur le port 8000.

---

# 7. CI/CD

Une pipeline CI/CD a été mise en place avec GitHub Actions.

Le workflow suit la chaîne suivante :

```text
Push sur main
      ↓
Tests Pytest
      ↓
Build de l'image Docker
      ↓
Déploiement sur Render
```

Le déploiement Render est déclenché uniquement si les étapes précédentes réussissent.

L'objectif est d'éviter de déployer une version de l'application qui ne passe pas les tests.

---

# 8. Déploiement

L'API est déployée sur Render.

Adresse publique :

```text
https://projet-8-scoring-model-deployment-1.onrender.com
```

Le endpoint de santé est disponible sur :

```text
https://projet-8-scoring-model-deployment-1.onrender.com/health
```

La documentation Swagger est disponible sur :

```text
https://projet-8-scoring-model-deployment-1.onrender.com/docs
```

Le déploiement utilisé dans ce projet constitue une démonstration de mise en production du modèle.

---

# 9. Monitoring de production

Le projet met en place une architecture de monitoring permettant de suivre le comportement du modèle et de l'API.

Architecture :

```text
Client
  ↓
FastAPI
  ↓
Modèle de scoring
  │
  ├── PostgreSQL
  │      ├── prédictions
  │      ├── probabilités
  │      ├── temps d'inférence
  │      ├── latence
  │      ├── erreurs
  │      └── résultats de drift
  │
  ├── Elasticsearch
  │      └── logs techniques structurés
  │
  └── Evidently
         └── détection du data drift

PostgreSQL + Elasticsearch
             ↓
         Streamlit
             ↓
      Dashboard monitoring
```

Le monitoring utilisé dans ce projet correspond à une production simulée / Proof of Concept.

---

# 10. PostgreSQL

PostgreSQL est utilisé pour stocker les données permettant de suivre le comportement de l'API.

La table principale contient notamment :

- la date de la requête
- la prédiction
- la probabilité de défaut
- le temps d'inférence
- la durée de traitement
- le statut HTTP
- les erreurs éventuelles
- les variables envoyées au modèle

Une seconde table permet de stocker les résultats de détection du data drift.

Pour améliorer les performances sous charge, l'API utilise un pool de connexions PostgreSQL.

Cela permet de réutiliser plusieurs connexions existantes plutôt que d'ouvrir une nouvelle connexion pour chaque prédiction.

---

# 11. Elasticsearch

Elasticsearch est utilisé pour stocker les logs techniques structurés de l'API.

Les événements enregistrés comprennent notamment :

- les prédictions réussies
- les erreurs
- les statuts HTTP
- les probabilités
- les temps d'inférence
- les durées de traitement

Afin de ne pas bloquer les réponses de l'API, les logs Elasticsearch sont envoyés en arrière-plan.

Le monitoring reste ainsi actif sans obliger le client à attendre la fin de l'écriture du log.

---

# 12. Data Drift

Evidently est utilisé pour comparer les données de référence aux données observées lors de la simulation de production.

Les données de référence contiennent :

```text
5000 observations
401 variables
```

Les données simulant la production contiennent :

```text
500 observations
401 variables
```

Résultat obtenu lors de l'analyse :

```text
Nombre de variables analysées : 401
Variables en drift            : 22
Part des variables en drift   : 5,49 %
Seuil global                  : 50 %
```

Le dataset global n'est donc pas considéré comme étant en drift.

Certaines variables individuelles présentent cependant une dérive.

Le seuil global utilisé par Evidently est de 50 % des colonnes en drift.

Avec environ 5,49 % de variables détectées en dérive, le seuil global n'est pas atteint.

Les résultats détaillés sont enregistrés dans PostgreSQL puis affichés dans le dashboard Streamlit.

---

# 13. Dashboard Streamlit

Un dashboard Streamlit permet de centraliser les indicateurs de monitoring.

Il affiche notamment :

- le nombre total de requêtes
- le nombre de succès
- le nombre d'erreurs
- le taux d'erreur
- la latence moyenne
- le P95
- le temps moyen d'inférence
- la distribution des scores
- les résultats de data drift
- les dernières requêtes
- les logs Elasticsearch

Lancement du dashboard :

```powershell
streamlit run dashboard/app.py
```

---

# 14. Baseline prédictive

Avant d'optimiser l'application, une baseline prédictive a été enregistrée.

Le modèle a été évalué sur un jeu de validation figé contenant :

```text
10000 clients
401 variables
```

La distribution réelle de la cible était :

```text
Classe 0 : 9193
Classe 1 : 807
```

Résultats :

```text
ROC AUC  : 0.7570
Recall   : 0.6518
F1 Score : 0.2762
Accuracy : 0.7243
```

Matrice de confusion :

```text
TN = 6717
FP = 2476
FN = 281
TP = 526
```

Le Recall de 65,18 % indique que le modèle identifie environ deux tiers des clients réellement en défaut dans ce jeu de validation.

L'Accuracy doit être interprétée avec prudence en raison du fort déséquilibre des classes.

Dans ce jeu de validation, plus de 91 % des observations appartiennent à la classe 0.

---

# 15. Baseline opérationnelle

Une première analyse des temps enregistrés par l'API a donné les résultats suivants.

## Temps d'inférence

```text
Moyenne : 15.737 ms
P50     : 15.715 ms
P95     : 16.451 ms
P99     : 18.695 ms
Min     : 12.831 ms
Max     : 25.719 ms
```

## Temps de traitement enregistré avant le monitoring externe

```text
Moyenne : 19.899 ms
P50     : 19.943 ms
P95     : 20.797 ms
P99     : 22.822 ms
Min     : 15.842 ms
Max     : 31.954 ms
```

Ces mesures ne représentaient cependant pas entièrement le temps réellement observé par le client car les écritures PostgreSQL et Elasticsearch étaient réalisées après certaines mesures internes.

Un profiling complet de la fonction de prédiction a donc été réalisé.

---

# 16. Profiling de l'API

Le profiling initial a permis d'identifier les parties les plus coûteuses de l'application.

Résultat initial :

```text
Nombre de requêtes : 20
Temps total         : 2.396 secondes
Temps moyen complet : 119.816 ms / requête
Débit séquentiel    : 8.35 req/s
```

Les principaux goulots identifiés étaient :

```text
Elasticsearch             ≈ 55 ms / requête
PostgreSQL                ≈ 19 ms / requête
model.predict_proba()     ≈ 19 ms / requête
model.predict()           ≈ 19 ms / requête
```

Le profiling a mis en évidence trois principaux axes d'optimisation :

1. le modèle était calculé deux fois
2. l'écriture Elasticsearch bloquait la réponse
3. les connexions PostgreSQL pouvaient être optimisées

---

# 17. Optimisation du modèle

Initialement, l'API exécutait :

```python
probabilities = model.predict_proba(client_df)
prediction = model.predict(client_df)
```

Pour un Random Forest, l'appel à `predict()` repose lui-même sur les probabilités produites par les arbres.

Le modèle effectuait donc inutilement deux calculs.

L'API utilise désormais uniquement :

```python
probabilities = model.predict_proba(client_df)

prediction_index = np.argmax(probabilities)

prediction = classes[prediction_index]
```

Cela permet d'obtenir la classe prédite à partir du résultat de `predict_proba()` sans parcourir deux fois le modèle.

Après cette première optimisation :

```text
Temps moyen complet : 97.554 ms / requête
Débit séquentiel    : 10.25 req/s
```

Par rapport au profiling initial :

```text
Temps moyen :
119.816 ms → 97.554 ms

Débit :
8.35 req/s → 10.25 req/s
```

---

# 18. Optimisation Elasticsearch

Le profiling montrait qu'Elasticsearch était le principal goulot d'étranglement.

Initialement :

```text
Prédiction
    ↓
Écriture Elasticsearch
    ↓
Attente de la réponse Elasticsearch
    ↓
Réponse au client
```

Cette écriture était synchrone.

Elle a été remplacée par un mécanisme en arrière-plan utilisant un pool de threads.

Nouvelle architecture :

```text
Prédiction
    ↓
Ajout du log dans une file
    ├──────────────→ Réponse au client
    │
    └──────────────→ Elasticsearch en arrière-plan
```

Le monitoring Elasticsearch est donc conservé, mais son écriture ne bloque plus directement la réponse envoyée au client.

Après cette optimisation, un profiling a obtenu :

```text
Temps moyen complet : 45.192 ms / requête
Débit séquentiel    : 22.13 req/s
```

Le profiling multi-thread doit cependant être interprété avec précaution car certaines opérations Elasticsearch continuent à s'exécuter parallèlement.

Les tests HTTP de charge sont donc utilisés comme mesure principale pour l'évaluation finale.

---

# 19. Tests de montée en charge

Avant l'optimisation complète, plusieurs niveaux de concurrence ont été testés.

| Concurrence | Débit | Latence moyenne | P95 | P99 | Erreurs |
|---:|---:|---:|---:|---:|---:|
| 5 | 31.56 req/s | 157.94 ms | 211.68 ms | 325.21 ms | 0 % |
| 10 | 34.37 req/s | 288.15 ms | 352.04 ms | 566.04 ms | 0 % |
| 20 | 35.16 req/s | 557.43 ms | 794.73 ms | 964.25 ms | 0 % |
| 50 | 34.45 req/s | 1366.13 ms | 3254.62 ms | 4352.35 ms | 0 % |

Ces tests montrent qu'avant l'optimisation complète, le débit atteignait un plateau autour de 35 requêtes par seconde dans l'environnement local utilisé.

Lorsque la concurrence passait de 20 à 50, le débit n'augmentait plus alors que la latence augmentait fortement.

Il s'agissait donc d'un signe de saturation.

---

# 20. Optimisation PostgreSQL

Une première approche consistant à réutiliser une seule connexion PostgreSQL permettait d'éviter les reconnexions.

Cette solution devenait cependant un goulot sous forte concurrence car les requêtes devaient partager une connexion unique.

La solution finale utilise donc un pool de connexions PostgreSQL.

Architecture :

```text
Requêtes concurrentes
        ↓
Pool PostgreSQL
 ├── connexion 1
 ├── connexion 2
 ├── connexion 3
 ├── ...
 └── connexion 10
```

Plusieurs écritures peuvent ainsi être traitées sans recréer systématiquement une connexion et sans imposer une connexion unique à toutes les requêtes.

---

# 21. Test prolongé avant optimisation du pool PostgreSQL

Un test prolongé de 5000 requêtes a été réalisé avec une concurrence de 20.

Résultats :

```text
Nombre de requêtes : 5000
Concurrence         : 20

Durée totale        : 222.146 sec
Succès              : 5000
Erreurs             : 0
Taux d'erreur       : 0.00 %

Débit               : 22.51 req/s

Latence moyenne     : 886.788 ms
P50                 : 887.179 ms
P95                 : 899.951 ms
P99                 : 923.485 ms
Maximum             : 1279.558 ms
```

L'API est restée stable et n'a retourné aucune erreur mais la connexion PostgreSQL partagée limitait fortement les performances.

---

# 22. Test prolongé après optimisation

Après la mise en place du pool PostgreSQL, exactement le même test a été exécuté :

```text
Nombre de requêtes : 5000
Concurrence         : 20
```

Résultats :

```text
Durée totale        : 81.262 sec
Succès              : 5000
Erreurs             : 0
Taux d'erreur       : 0.00 %

Débit               : 61.53 req/s

Latence moyenne     : 324.727 ms
P50                 : 321.257 ms
P95                 : 394.561 ms
P99                 : 437.375 ms
Minimum             : 124.902 ms
Maximum             : 734.946 ms
```

L'API a donc traité l'intégralité des 5000 requêtes sans aucune erreur.

---

# 23. Comparaison avant / après optimisation PostgreSQL

Sur exactement le même scénario de 5000 requêtes avec une concurrence de 20 :

| Indicateur | Avant | Après |
|---|---:|---:|
| Requêtes | 5000 | 5000 |
| Erreurs | 0 % | 0 % |
| Durée totale | 222.146 s | 81.262 s |
| Débit | 22.51 req/s | 61.53 req/s |
| Latence moyenne | 886.79 ms | 324.73 ms |
| P95 | 899.95 ms | 394.56 ms |
| P99 | 923.49 ms | 437.38 ms |

Les améliorations observées sont approximativement :

```text
Débit :
22.51 → 61.53 req/s
≈ +173 %

Latence moyenne :
886.79 → 324.73 ms
≈ -63 %

P95 :
899.95 → 394.56 ms
≈ -56 %

P99 :
923.49 → 437.38 ms
≈ -53 %

Taux d'erreur :
0 % → 0 %
```

Ces résultats montrent que l'optimisation améliore fortement le débit et la latence sans dégrader la stabilité de l'API.

---

# 24. Capacité sous charge

Le projet ne cherche pas à déterminer une capacité universelle indépendante de l'infrastructure.

Les performances dépendent notamment :

- du processeur
- de la mémoire disponible
- du nombre de workers
- du réseau
- de PostgreSQL
- d'Elasticsearch
- de l'infrastructure utilisée pour le déploiement

Dans l'environnement local du projet, l'API optimisée a néanmoins démontré sa capacité à traiter :

```text
5000 requêtes
20 requêtes concurrentes
61.53 requêtes/seconde
0 % d'erreur
```

Ce test permet de vérifier la stabilité de l'application sur un volume important de requêtes dans le cadre du Proof of Concept.

---

# 25. Synthèse des optimisations

Les optimisations ont été réalisées à partir des résultats du profiling et des tests de charge.

## Optimisation 1

Suppression du double calcul :

```text
predict_proba()
+
predict()
```

remplacé par :

```text
predict_proba()
+
argmax()
```

## Optimisation 2

Passage du logging Elasticsearch :

```text
synchrone
```

à :

```text
asynchrone / arrière-plan
```

## Optimisation 3

Passage de la gestion PostgreSQL à :

```text
pool de connexions
```

Ces modifications ont permis de réduire les temps d'attente et d'améliorer la capacité de traitement concurrente de l'API.

---

# 26. Technologies utilisées

Le projet utilise notamment :

```text
Python
FastAPI
Uvicorn
Scikit-learn
MLflow
Pandas
NumPy
PostgreSQL
Psycopg
Elasticsearch
Evidently
Streamlit
Docker
GitHub Actions
Render
Pytest
```

---

# 27. Conclusion

Ce projet met en œuvre une chaîne complète de mise en production et de suivi d'un modèle de Machine Learning.

```text
Versioning Git / GitHub
        ↓
Modèle MLflow
        ↓
API FastAPI
        ↓
Tests automatisés
        ↓
Docker
        ↓
CI/CD GitHub Actions
        ↓
Déploiement Render
        ↓
Monitoring PostgreSQL
        ↓
Logs Elasticsearch
        ↓
Data Drift Evidently
        ↓
Dashboard Streamlit
        ↓
Profiling
        ↓
Tests de charge
        ↓
Optimisation
```

Le monitoring permet de suivre les prédictions, les probabilités, les performances de l'API, les erreurs et le data drift.

Le profiling a permis d'identifier plusieurs goulots d'étranglement qui ont ensuite été corrigés.

Les tests de charge ont permis de vérifier quantitativement l'amélioration des performances.

Sur le test final de 5000 requêtes avec une concurrence de 20, l'API a traité toutes les requêtes sans erreur avec un débit de 61.53 requêtes par seconde.

Les performances présentées correspondent à l'environnement local utilisé pour les tests et ne doivent pas être interprétées comme une capacité universelle de l'application indépendamment de son infrastructure.