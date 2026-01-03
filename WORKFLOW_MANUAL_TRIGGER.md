# 🚀 Guide pour Déclencher le Workflow GitHub Actions

## Comment déclencher le pipeline MLOps sur GitHub (Manuel)

### Étape 1️⃣ : Accéder à GitHub Actions

1. Ouvre ton navigateur
2. Va à : **https://github.com/lkassous/MLOps-Housepricepredictor**
3. Clique sur l'onglet **"Actions"** (en haut)

```
Code | Issues | Pull requests | Actions ← CLIQUE ICI
```

---

### Étape 2️⃣ : Sélectionner le Workflow

1. Tu verras une liste des workflows à gauche
2. Clique sur **"mlops-pipeline.yml"**

```
Actions
├── All workflows
└── mlops-pipeline.yml ← CLIQUE ICI
```

---

### Étape 3️⃣ : Déclencher Manuellement

1. Tu verras un bouton bleu **"Run workflow"** (à droite)
2. Clique dessus

```
┌─────────────────────────────────────┐
│  Run workflow  ← CLIQUE ICI (bouton bleu)  │
└─────────────────────────────────────┘
```

3. Une popup aparaîtra, clique à nouveau sur **"Run workflow"**

```
Run workflow
[Dropdown: Branch: master]
┌──────────────┐
│ Run workflow │ ← CLIQUE ICI
└──────────────┘
```

---

### Étape 4️⃣ : Vérifier l'Exécution

1. Tu verras un nouveau run aparaître dans la liste
2. Clique dessus pour voir les logs en temps réel

```
mlops-pipeline.yml                          ⟳
├─ 🟡 Pending  (1 min ago) ← CLIQUE POUR VOIR LES LOGS
├─ ✅ Completed (5 min ago)
└─ ✅ Completed (10 min ago)
```

---

## Ce que le Workflow va faire

```
1. Setup Python 3.11
   ↓
2. Install dependencies (pip)
   ↓
3. Run pipeline.py
   ├─ Load data (train.csv)
   ├─ Validate data
   ├─ Preprocess (43 categorical columns)
   ├─ Train 3 models (Linear, RF, XGBoost)
   ├─ Compare & select best
   ├─ Log to MLflow
   └─ Promote to @production
   ↓
4. Upload artifacts to GitHub
   ↓
5. Build Docker image (optionnel)
   ↓
6. Deploy to Cloud Run (optionnel - nécessite GCP)
```

---

## Temps d'Exécution Attendu

- **Setup**: 30 secondes
- **Pipeline**: 2-3 minutes
- **Total**: 3-4 minutes

---

## Résultat du Workflow

### ✅ Si tout passe
```
✅ Setup Python 3.11
✅ Install dependencies
✅ Run MLOps Pipeline
   - Data loaded: 1460 samples
   - Models trained: 3 modèles
   - Best model: XGBoost (R²=0.9179)
   - Promoted to production
✅ Upload artifacts
```

### ❌ Si erreur
```
❌ Run MLOps Pipeline
   Error: [description]
   ...
```

Clique sur le step échoué pour voir les détails.

---

## Où Voir les Résultats

### Dans GitHub (Artifacts)
```
Actions → mlops-pipeline.yml → [Run] → Artifacts
├─ mlflow-data/
│  └─ mlruns/ (toutes les métriques)
└─ pipeline-report
   └─ pipeline_report.json (résumé)
```

**Télécharge les artifacts** pour voir les résultats détaillés.

---

## Prochaines Étapes (Après le Workflow)

### Option 1 : Automation complète (Recommandé)
```bash
# Ajoute les secrets GitHub
# Settings → Secrets → New repository secret
# - GCP_PROJECT_ID
# - GCP_SA_KEY

# Puis le workflow déploiera automatiquement sur Cloud Run
```

### Option 2 : Manuellement Local
```bash
# À chaque fois que tu veux entraîner les modèles
python pipeline.py --data-path train.csv

# Puis voir les résultats dans MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

---

## Troubleshooting

### ❓ Le workflow n'apparaît pas
- Assure-toi d'être sur la branche **master**
- Vérifie que le fichier `.github/workflows/mlops-pipeline.yml` existe dans le repo

### ❓ Le workflow échoue
- Clique sur le run échoué
- Vois les logs rouges pour le détail de l'erreur
- Les erreurs courantes:
  - Import erreur → Vérifiez `requirements.txt`
  - Fichier manquant → Vérifiez `train.csv` est pushé

### ❓ Où voir les modèles entraînés
- GitHub Artifacts → `mlflow-data/mlruns/`
- Ou localement → `mlruns/` après `python pipeline.py`

---

## 🎯 Résumé Rapide

| Étape | Action | URL |
|-------|--------|-----|
| 1 | Go to GitHub | https://github.com/lkassous/MLOps-Housepricepredictor |
| 2 | Click Actions tab | .../actions |
| 3 | Select mlops-pipeline.yml | .../actions?query=workflow%3Amlops-pipeline |
| 4 | Click "Run workflow" (blue button) | - |
| 5 | Watch logs in real-time | - |
| 6 | Download artifacts | Actions → [Run] → Artifacts |

---

**🎉 C'est tout ! Le pipeline s'exécutera automatiquement !**

