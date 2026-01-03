# 📋 AIDE-MÉMOIRE MLOPS - RÉFÉRENCE RAPIDE

## ⚡ DÉMARRAGE EN 3 ÉTAPES

```powershell
# 1. Installer (une seule fois)
.\install.ps1

# 2. Exécuter le pipeline
python src\quick_start.py

# 3. Visualiser
.\start_mlflow.ps1
# → http://localhost:5000
```

---

## 🎯 COMMANDES ESSENTIELLES

| Action | Commande |
|--------|----------|
| **Installer dépendances** | `python -m pip install -r requirements.txt` |
| **Tester installation** | `python test_installation.py` |
| **Préparer données** | `python src\data_preparation.py` |
| **Entraîner modèles** | `python src\train_models.py` |
| **Enregistrer meilleur** | `python src\register_model.py` |
| **Pipeline complet** | `python src\quick_start.py` |
| **Démarrer MLflow UI** | `mlflow ui` |
| **MLflow UI (autre port)** | `mlflow ui --port 5001` |

---

## 📊 STRUCTURE DU PROJET

```
📁 Projet/
├── 📄 train.csv              ← Dataset (1460 maisons, 81 features)
├── 📄 test.csv               ← Test dataset
├── 📄 requirements.txt       ← Dépendances Python
│
├── 📂 src/                   ← Code source
│   ├── data_preparation.py   ← Préparation données
│   ├── train_models.py       ← Entraînement + MLflow
│   ├── register_model.py     ← Enregistrement modèle
│   └── quick_start.py        ← Exécution automatique
│
├── 📂 mlruns/                ← Tracking MLflow (auto)
├── 📂 models/                ← Modèles sauvegardés
│
├── 📄 GUIDE_MLOPS_FR.md      ← 📚 Guide complet
├── 📄 COMMANDES_RAPIDES.md   ← ⚡ Commandes détaillées
├── 📄 WORKFLOW.md            ← 🔄 Architecture
└── 📄 AIDE_MEMOIRE.md        ← 📋 Vous êtes ici!
```

---

## 🤖 MODÈLES ENTRAÎNÉS (8 au total)

| # | Modèle | Description |
|---|--------|-------------|
| 1 | **Linear Regression** | Régression linéaire simple |
| 2 | **Ridge Regression** | Régression avec L2 |
| 3 | **Lasso Regression** | Régression avec L1 |
| 4 | **Decision Tree** | Arbre de décision |
| 5 | **Random Forest** | Ensemble de 100 arbres |
| 6 | **Gradient Boosting** | Boosting séquentiel |
| 7 | **XGBoost** | Gradient boosting optimisé |
| 8 | **LightGBM** | Boosting rapide et léger |

---

## 📈 MÉTRIQUES DE PERFORMANCE

| Métrique | Description | Objectif |
|----------|-------------|----------|
| **RMSE** | Erreur quadratique moyenne | ↓ Plus faible |
| **MAE** | Erreur absolue moyenne | ↓ Plus faible |
| **R²** | Coefficient de détermination | ↑ Plus proche de 1 |

### Interprétation du R²
- **0.90+** : Excellent
- **0.80-0.89** : Très bon
- **0.70-0.79** : Bon
- **<0.70** : À améliorer

---

## 🔄 WORKFLOW COMPLET

```
1. Données       → data_preparation.py
                    ↓
2. Entraînement  → train_models.py (8 modèles)
                    ↓
3. Tracking      → MLflow (automatique)
                    ↓
4. Visualisation → MLflow UI (localhost:5000)
                    ↓
5. Sélection     → Comparer dans UI
                    ↓
6. Enregistrement→ register_model.py
                    ↓
7. Déploiement   → API / Cloud / Docker
```

---

## 🌐 MLFLOW UI - NAVIGATION

### Accès
```
http://localhost:5000
```

### Pages principales
1. **Experiments** : Liste des expériences
2. **Runs** : Tous les runs d'entraînement
3. **Compare** : Comparer plusieurs modèles
4. **Models** : Model Registry

### Actions rapides
- **Trier** : Cliquer sur l'en-tête de colonne (RMSE, R², etc.)
- **Filtrer** : Utiliser la barre de recherche
- **Comparer** : Cocher ☑ plusieurs runs → "Compare"
- **Détails** : Cliquer sur un run
- **Télécharger** : Dans Artifacts → Download

---

## 🔧 DIAGNOSTIC RAPIDE

### Vérifier Python
```powershell
python --version
# Attendu: Python 3.10+ ou supérieur
```

### Vérifier MLflow
```powershell
python -c "import mlflow; print(f'MLflow {mlflow.__version__}')"
# Attendu: MLflow 2.10.0
```

### Vérifier tout
```powershell
python test_installation.py
# Tous les tests doivent passer ✓
```

---

## 🚨 PROBLÈMES COURANTS

| Problème | Solution |
|----------|----------|
| `python` non reconnu | Installer Python et cocher "Add to PATH" |
| `pip` non reconnu | Utiliser `python -m pip` |
| Port 5000 occupé | `mlflow ui --port 5001` |
| Dataset non trouvé | `cd` vers le bon dossier |
| Erreur d'import | Réinstaller: `pip install -r requirements.txt` |

---

## 💾 SCRIPTS D'INSTALLATION

| Script | Usage |
|--------|-------|
| `install.ps1` | Installation complète automatique |
| `start_mlflow.ps1` | Démarrer MLflow UI facilement |
| `test_installation.py` | Vérifier que tout fonctionne |

---

## 📚 DOCUMENTATION

| Fichier | Contenu |
|---------|---------|
| **GUIDE_MLOPS_FR.md** | 📖 Guide complet (LIRE EN PREMIER) |
| **COMMANDES_RAPIDES.md** | ⚡ Toutes les commandes détaillées |
| **WORKFLOW.md** | 🔄 Architecture et workflow visuel |
| **AIDE_MEMOIRE.md** | 📋 Référence rapide (ce fichier) |
| **README.md** | 📝 Documentation principale |

---

## 🎓 LIENS UTILES

| Ressource | URL |
|-----------|-----|
| MLflow Docs | https://mlflow.org/docs/latest/ |
| Scikit-learn | https://scikit-learn.org/ |
| XGBoost | https://xgboost.readthedocs.io/ |
| LightGBM | https://lightgbm.readthedocs.io/ |
| Pandas | https://pandas.pydata.org/ |

---

## 🎯 OBJECTIFS DU PROJET

✅ Prédire les prix des maisons avec précision  
✅ Comparer 8 modèles différents  
✅ Utiliser MLflow pour le tracking  
✅ Enregistrer le meilleur modèle  
✅ Préparer pour le déploiement  

---

## 📊 EXEMPLE DE RÉSULTATS ATTENDUS

```
Model                 Test RMSE    Test MAE     Test R²
──────────────────────────────────────────────────────
XGBoost              $25,123.45   $16,789.12    0.8956  ← Meilleur
LightGBM             $26,234.56   $17,234.56    0.8912
Random Forest        $28,456.78   $18,234.56    0.8734
Gradient Boosting    $29,123.45   $19,456.78    0.8678
Ridge                $32,456.78   $21,234.56    0.8456
Lasso                $33,234.56   $22,123.45    0.8398
Linear               $34,567.89   $23,456.78    0.8234
Decision Tree        $38,123.45   $25,678.90    0.7956
```

---

## 🔄 CYCLE DE DÉVELOPPEMENT

```
1. Développer     → Modifier train_models.py
2. Expérimenter   → python src\train_models.py
3. Comparer       → MLflow UI
4. Itérer         → Ajuster hyperparamètres
5. Valider        → Meilleurs résultats
6. Enregistrer    → python src\register_model.py
7. Déployer       → Production
```

---

## 💡 CONSEILS PRO

✨ **Nommage** : Donnez des noms descriptifs à vos expériences  
✨ **Tags** : Utilisez des tags dans MLflow pour organiser  
✨ **Notes** : Documentez vos découvertes directement dans MLflow UI  
✨ **Sauvegarde** : Sauvegardez régulièrement le dossier `mlruns/`  
✨ **Environnement** : Utilisez un environnement virtuel Python  

---

## 🚀 ALLER PLUS LOIN

1. **Feature Engineering** : Créer de nouvelles features
2. **Hyperparameter Tuning** : GridSearchCV, RandomizedSearchCV
3. **Cross-Validation** : K-Fold validation
4. **Ensemble Methods** : Stacking, Blending
5. **Deep Learning** : Neural Networks
6. **Déploiement** : API REST, Cloud, Docker
7. **CI/CD** : Automatisation avec GitHub Actions

---

**Besoin d'aide? Consultez [GUIDE_MLOPS_FR.md](GUIDE_MLOPS_FR.md)**

**Bon MLOps! 🎉**
