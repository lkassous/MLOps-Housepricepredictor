# 🎯 ÉTAPES D'INSTALLATION ET D'UTILISATION

## 📋 INSTALLATION (À FAIRE UNE SEULE FOIS)

### Étape 1: Installer Python
1. Téléchargez Python depuis: https://www.python.org/downloads/
2. **IMPORTANT**: Pendant l'installation, cochez ☑ "Add Python to PATH"
3. Redémarrez votre ordinateur après l'installation

### Étape 2: Vérifier Python
Ouvrez PowerShell et tapez:
```powershell
python --version
```
Vous devriez voir: `Python 3.10.x` ou supérieur

### Étape 3: Naviguer vers le projet
```powershell
cd c:\Users\USERµ\Desktop\MLOPS_PROJECT\house-prices-advanced-regression-techniques
```

### Étape 4: Installer les dépendances
**Option A - Script automatique (RECOMMANDÉ):**
```powershell
.\install.ps1
```

**Option B - Installation manuelle:**
```powershell
python -m pip install -r requirements.txt
```

### Étape 5: Vérifier l'installation
```powershell
python test_installation.py
```
Tous les tests doivent afficher ✓

---

## 🚀 UTILISATION (APRÈS INSTALLATION)

### Option 1: Démarrage Rapide (TOUT EN UN)
```powershell
# Lance tout le pipeline automatiquement
python src\quick_start.py
```

### Option 2: Étape par Étape

**1. Préparer les données**
```powershell
python src\data_preparation.py
```

**2. Entraîner les modèles avec MLflow**
```powershell
python src\train_models.py
```

**3. Visualiser dans MLflow UI**
```powershell
# Ouvrir un NOUVEAU PowerShell
.\start_mlflow.ps1
# Ou: mlflow ui
```

**4. Ouvrir le navigateur**
- Allez sur: http://localhost:5000
- Comparez les 8 modèles entraînés
- Identifiez le meilleur modèle

**5. Enregistrer le meilleur modèle**
```powershell
# Dans le premier PowerShell
python src\register_model.py
```

---

## 📊 QUE FAIT CHAQUE SCRIPT?

| Script | Description |
|--------|-------------|
| `data_preparation.py` | Charge et nettoie le dataset (train.csv) |
| `train_models.py` | Entraîne 8 modèles et log dans MLflow |
| `register_model.py` | Enregistre le meilleur modèle |
| `quick_start.py` | Exécute les 3 scripts ci-dessus automatiquement |

---

## 🎓 APPRENDRE MLFLOW

### 1. Exécutez le pipeline
```powershell
python src\quick_start.py
```

### 2. Ouvrez MLflow UI
```powershell
mlflow ui
```

### 3. Dans votre navigateur (http://localhost:5000)

**Ce que vous verrez:**
- Liste de tous les modèles entraînés
- Métriques de performance (RMSE, MAE, R²)
- Paramètres de chaque modèle
- Graphiques de comparaison

**Actions possibles:**
- ☑ Sélectionner plusieurs modèles
- Cliquer sur "Compare" pour les comparer
- Voir les graphiques de performance
- Télécharger les modèles entraînés

---

## 📚 DOCUMENTATION DISPONIBLE

Tous ces fichiers sont dans votre projet:

1. **GUIDE_MLOPS_FR.md** 
   - 📖 Guide complet en français
   - **À LIRE EN PREMIER** pour tout comprendre

2. **COMMANDES_RAPIDES.md**
   - ⚡ Référence de toutes les commandes
   - Solutions aux problèmes courants

3. **WORKFLOW.md**
   - 🔄 Diagramme du workflow MLOps
   - Architecture du projet

4. **AIDE_MEMOIRE.md**
   - 📋 Référence rapide
   - Tableau des commandes essentielles

5. **DEMARRER_ICI.md** (ce fichier)
   - 🎯 Instructions de démarrage
   - Par où commencer

---

## ✅ CHECKLIST DE DÉMARRAGE

- [ ] Python 3.10+ installé (avec "Add to PATH" coché)
- [ ] PowerShell ouvert dans le bon dossier
- [ ] Dépendances installées (`.\install.ps1`)
- [ ] Test d'installation réussi (`python test_installation.py`)
- [ ] Pipeline exécuté (`python src\quick_start.py`)
- [ ] MLflow UI ouvert (`mlflow ui`)
- [ ] Navigateur sur http://localhost:5000
- [ ] Modèles visibles dans l'interface MLflow
- [ ] Guide GUIDE_MLOPS_FR.md lu

---

## 🆘 BESOIN D'AIDE?

### Si Python n'est pas reconnu:
1. Réinstallez Python
2. Cochez "Add Python to PATH" pendant l'installation
3. Redémarrez PowerShell

### Si les imports échouent:
```powershell
python -m pip install --upgrade -r requirements.txt
```

### Si MLflow UI ne démarre pas:
```powershell
# Essayez un autre port
mlflow ui --port 5001
```

### Si le dataset n'est pas trouvé:
```powershell
# Vérifiez que vous êtes dans le bon dossier
Get-Location
# Devrait afficher: ...\house-prices-advanced-regression-techniques
```

---

## 🎯 PROCHAINES ÉTAPES

Après avoir tout installé et testé:

1. **Expérimenter**: Modifiez les paramètres dans `train_models.py`
2. **Comparer**: Utilisez MLflow UI pour comparer les résultats
3. **Optimiser**: Ajustez les hyperparamètres des meilleurs modèles
4. **Approfondir**: Lisez le guide complet (GUIDE_MLOPS_FR.md)
5. **Déployer**: Créez une API avec le meilleur modèle

---

## 📞 RESSOURCES

- **Documentation MLflow**: https://mlflow.org/docs/latest/
- **Tutoriels Python**: https://docs.python.org/fr/3/tutorial/
- **Scikit-learn**: https://scikit-learn.org/stable/

---

**Vous êtes prêt! Commencez avec `.\install.ps1` 🚀**
