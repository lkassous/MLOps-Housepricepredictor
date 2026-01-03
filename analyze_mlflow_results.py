"""
Analyse des résultats MLflow - Hyperparameter Tuning
Visualise l'impact des paramètres sur les performances
"""
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

print("=" * 70)
print("ANALYSE DES RÉSULTATS MLFLOW - HYPERPARAMETER TUNING")
print("=" * 70)

# 1. Récupérer tous les runs de l'expérience d'hyperparameter tuning
experiment_name = "House-Prices-Hyperparameter-Tuning"
try:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"\nExpérience '{experiment_name}' non trouvée.")
        print("Utilisation de l'expérience principale...")
        experiment_name = "House-Prices-Regression"
        experiment = client.get_experiment_by_name(experiment_name)
    
    experiment_id = experiment.experiment_id
    print(f"\nExpérience analysée: {experiment_name}")
    print(f"ID: {experiment_id}")
    
    # Récupérer tous les runs
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.test_rmse ASC"],
        max_results=500
    )
    
    print(f"\nNombre total de runs: {len(runs)}")
    
    # 2. Extraire les données dans un DataFrame
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        
        data.append({
            'run_id': run.info.run_id,
            'n_estimators': int(params.get('n_estimators', 0)),
            'max_depth': int(params.get('max_depth', 0)),
            'learning_rate': float(params.get('learning_rate', 0)),
            'subsample': float(params.get('subsample', 1.0)),
            'colsample_bytree': float(params.get('colsample_bytree', 1.0)),
            'train_rmse': metrics.get('train_rmse', 0),
            'test_rmse': metrics.get('test_rmse', 0),
            'train_r2': metrics.get('train_r2', 0),
            'test_r2': metrics.get('test_r2', 0),
            'train_mae': metrics.get('train_mae', 0),
            'test_mae': metrics.get('test_mae', 0)
        })
    
    df = pd.DataFrame(data)
    
    # 3. Statistiques descriptives
    print("\n" + "=" * 70)
    print("STATISTIQUES DES MÉTRIQUES")
    print("=" * 70)
    print("\nTest RMSE:")
    print(df['test_rmse'].describe())
    print(f"\nMeilleur RMSE: ${df['test_rmse'].min():,.2f}")
    print(f"Pire RMSE: ${df['test_rmse'].max():,.2f}")
    print(f"Écart: ${df['test_rmse'].max() - df['test_rmse'].min():,.2f}")
    
    print("\nTest R²:")
    print(df['test_r2'].describe())
    
    # 4. Meilleurs runs
    print("\n" + "=" * 70)
    print("TOP 5 MEILLEURS MODÈLES")
    print("=" * 70)
    top5 = df.nlargest(5, 'test_r2')[['n_estimators', 'max_depth', 'learning_rate', 
                                        'subsample', 'colsample_bytree', 'test_rmse', 'test_r2']]
    print(top5.to_string(index=False))
    
    # 5. Analyse de l'impact des paramètres
    print("\n" + "=" * 70)
    print("IMPACT DES PARAMÈTRES SUR TEST RMSE (moyenne par valeur)")
    print("=" * 70)
    
    for param in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']:
        print(f"\n{param}:")
        impact = df.groupby(param)['test_rmse'].agg(['mean', 'std', 'count'])
        print(impact.to_string())
    
    # 6. Corrélations
    print("\n" + "=" * 70)
    print("CORRÉLATIONS ENTRE PARAMÈTRES ET MÉTRIQUES")
    print("=" * 70)
    
    correlation_cols = ['n_estimators', 'max_depth', 'learning_rate', 
                       'subsample', 'colsample_bytree', 'test_rmse', 'test_r2']
    corr_matrix = df[correlation_cols].corr()
    print("\nCorrélation avec Test RMSE:")
    print(corr_matrix['test_rmse'].sort_values().to_string())
    
    # 7. Visualisations
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)
    
    # Créer une figure avec plusieurs subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analyse Hyperparameter Tuning - Impact sur Test RMSE', fontsize=16, y=1.00)
    
    # 1. n_estimators vs RMSE
    ax = axes[0, 0]
    df.groupby('n_estimators')['test_rmse'].mean().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Impact de n_estimators')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Test RMSE (moyenne)')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. max_depth vs RMSE
    ax = axes[0, 1]
    df.groupby('max_depth')['test_rmse'].mean().plot(kind='bar', ax=ax, color='darkorange')
    ax.set_title('Impact de max_depth')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('Test RMSE (moyenne)')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. learning_rate vs RMSE
    ax = axes[0, 2]
    df.groupby('learning_rate')['test_rmse'].mean().plot(kind='bar', ax=ax, color='green')
    ax.set_title('Impact de learning_rate')
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('Test RMSE (moyenne)')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. subsample vs RMSE
    ax = axes[1, 0]
    df.groupby('subsample')['test_rmse'].mean().plot(kind='bar', ax=ax, color='purple')
    ax.set_title('Impact de subsample')
    ax.set_xlabel('subsample')
    ax.set_ylabel('Test RMSE (moyenne)')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. colsample_bytree vs RMSE
    ax = axes[1, 1]
    df.groupby('colsample_bytree')['test_rmse'].mean().plot(kind='bar', ax=ax, color='brown')
    ax.set_title('Impact de colsample_bytree')
    ax.set_xlabel('colsample_bytree')
    ax.set_ylabel('Test RMSE (moyenne)')
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Distribution des RMSE
    ax = axes[1, 2]
    ax.hist(df['test_rmse'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(df['test_rmse'].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: ${df["test_rmse"].mean():,.0f}')
    ax.axvline(df['test_rmse'].min(), color='green', linestyle='--', linewidth=2, label=f'Meilleur: ${df["test_rmse"].min():,.0f}')
    ax.set_title('Distribution des Test RMSE')
    ax.set_xlabel('Test RMSE')
    ax.set_ylabel('Fréquence')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlflow_analysis_results.png', dpi=300, bbox_inches='tight')
    print("\nGraphiques sauvegardés: mlflow_analysis_results.png")
    
    # 8. Heatmap de corrélation
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax2)
    ax2.set_title('Matrice de Corrélation - Paramètres et Métriques', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('mlflow_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap sauvegardée: mlflow_correlation_heatmap.png")
    
    # 9. Exporter les résultats en CSV
    df_export = df.sort_values('test_rmse')
    df_export.to_csv('hyperparameter_tuning_results.csv', index=False)
    print("\nRésultats exportés: hyperparameter_tuning_results.csv")
    
    # 10. Recommandations
    print("\n" + "=" * 70)
    print("RECOMMANDATIONS")
    print("=" * 70)
    
    best_params = df.loc[df['test_rmse'].idxmin()]
    print("\nMeilleure configuration trouvée:")
    print(f"  - n_estimators: {int(best_params['n_estimators'])}")
    print(f"  - max_depth: {int(best_params['max_depth'])}")
    print(f"  - learning_rate: {best_params['learning_rate']}")
    print(f"  - subsample: {best_params['subsample']}")
    print(f"  - colsample_bytree: {best_params['colsample_bytree']}")
    print(f"\nPerformances:")
    print(f"  - Test RMSE: ${best_params['test_rmse']:,.2f}")
    print(f"  - Test R²: {best_params['test_r2']:.4f}")
    print(f"  - Test MAE: ${best_params['test_mae']:,.2f}")
    
    print("\n" + "=" * 70)
    print("ANALYSE TERMINÉE")
    print("=" * 70)
    
except Exception as e:
    print(f"\nErreur: {e}")
    import traceback
    traceback.print_exc()
