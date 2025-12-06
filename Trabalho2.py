import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score)

RANDOM_STATE = 42
CV_FOLDS = 5
matplotlib.use('Agg')
sns.set(style="whitegrid")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def carregar_dados(caminho_csv):
    caminho = Path(caminho_csv)
    if not caminho.exists():
        print(f"Erro: arquivo nao encontrado em {caminho}")
        sys.exit(1)
    print(f"Carregando dados de: {caminho.name}")
    return pd.read_csv(caminho)

def preprocessamento_avancado(df):
    df_proc = df.copy()

    for col in df_proc.columns:
        if df_proc[col].dtype == 'object':
            if df_proc[col].isna().all():
                df_proc[col] = df_proc[col].fillna('unknown')
            else:
                df_proc[col] = df_proc[col].fillna(df_proc[col].mode().iloc[0])
        else:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())

    if 'daily_social_media_time' in df_proc.columns and 'work_hours_per_day' in df_proc.columns:
        df_proc['social_minutes_per_work_hour'] = df_proc['daily_social_media_time'] / (df_proc['work_hours_per_day'] * 60 + 1)
    
    if 'stress_level' in df_proc.columns and 'work_hours_per_day' in df_proc.columns:
        df_proc['stress_x_work'] = df_proc['stress_level'] * df_proc['work_hours_per_day']

    if 'sleep_hours' in df_proc.columns:
        df_proc['sleep_deficit'] = (8 - df_proc['sleep_hours']).clip(lower=0)

    if 'breaks_during_work' in df_proc.columns and 'work_hours_per_day' in df_proc.columns:
        df_proc['breaks_per_work_hour'] = df_proc['breaks_during_work'] / (df_proc['work_hours_per_day'] + 1)
        
    if 'work_hours_per_day' in df_proc.columns:
        df_proc['work_hours_sq'] = df_proc['work_hours_per_day']**2

    cat_cols = [c for c in ['gender', 'social_platform_preference', 'job_type'] if c in df_proc.columns]
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
        
    return df_proc

def run_analise_regressao(df_proc_avancado, output_dir, colunas_vazamento):
    print("\n--- [TRABALHO 1] Rodando Analise de Regressao ---")
    target = 'perceived_productivity_score'
    
    if target not in df_proc_avancado.columns:
        return

    cols_drop = [c for c in colunas_vazamento if c in df_proc_avancado.columns]
    X = df_proc_avancado.drop(columns=cols_drop, errors='ignore')
    y = df_proc_avancado[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    
    model = XGBRegressor(random_state=RANDOM_STATE, n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"Regressão Baseline RMSE: {rmse(y_test, preds):.4f}")
    print(f"Regressão Baseline R2: {r2_score(y_test, preds):.4f}")

def run_classificacao_otimizada(df_proc, output_dir, colunas_vazamento):
    print("\n--- [TRABALHO 2] Otimizacao e Mudanca de Metodologia (Classificacao) ---")
    
    target_original = 'perceived_productivity_score'
    if target_original not in df_proc.columns:
        print("Alvo nao encontrado.")
        return

    df_class = df_proc.copy()
    df_class['productivity_class'] = pd.qcut(df_class[target_original], q=3, labels=[0, 1, 2])
    
    print("Distribuição das Classes:")
    print(df_class['productivity_class'].value_counts(normalize=True))

    cols_drop = [c for c in colunas_vazamento if c in df_class.columns] + ['productivity_class']
    
    X = df_class.drop(columns=cols_drop, errors='ignore')
    y = df_class['productivity_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

    print("\nIniciando Grid Search (Otimizacao de Hiperparametros)...")
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\nMelhores Parametros Encontrados: {grid_search.best_params_}")
    print(f"Melhor Acuracia no Treino (CV): {grid_search.best_score_:.4f}")

    y_pred = best_model.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nResultados no Conjunto de Teste (Otimizado):")
    print(f"Acuracia: {acuracia:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', normalize='true')
    plt.title('Matriz de Confusao Normalizada (XGBoost Otimizado)')
    plt.grid(False)
    
    output_fig = output_dir / "matriz_confusao_otimizada.png"
    plt.savefig(output_fig, bbox_inches='tight')
    plt.close()
    print(f"Grafico salvo: {output_fig}")

    res_df = pd.DataFrame([{
        'Experimento': 'Classificacao Otimizada (Trab 2)',
        'Best Params': str(grid_search.best_params_),
        'Acuracia': acuracia,
        'F1_Weighted': f1
    }])
    res_df.to_csv(output_dir / "resultados_trabalho2.csv", index=False)
    print("\n=== RESULTADOS FINAIS TRABALHO 2 ===")
    print("Melhores parâmetros:", grid_search.best_params_)
    print("Acurácia:", acuracia)
    print("F1-Weighted:", f1)

def main():
    if len(sys.argv) != 2:
        print("Erro: forneca o caminho para o arquivo csv")
        print(f"Uso: python {sys.argv[0]} <caminho_para_o_dataset.csv>")
        sys.exit(1)
        
    caminho_csv = sys.argv[1]
    output_dir = Path("saida")
    output_dir.mkdir(exist_ok=True)
    
    df_bruto = carregar_dados(caminho_csv)
    df_proc = preprocessamento_avancado(df_bruto)
    
    colunas_vazamento = ['actual_productivity_score', 'perceived_productivity_score', 'job_satisfaction_score']

    run_analise_regressao(df_proc, output_dir, colunas_vazamento)
    
    run_classificacao_otimizada(df_proc, output_dir, colunas_vazamento)
    
    print("\nProcesso concluido com sucesso.")

if __name__ == "__main__":
    main()