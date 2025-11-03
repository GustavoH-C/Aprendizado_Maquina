import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
CV_FOLDS = 5
matplotlib.use('Agg')
sns.set(style="whitegrid")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def carregar_dados(caminho_csv):
    caminho = Path(caminho_csv)
    if not caminho.exists():
        print(f"Erro: Arquivo nao encontrado em {caminho}")
        sys.exit(1)
    print(f"Carregando dados de: {caminho.name}")
    return pd.read_csv(caminho)

def preprocessamento_baseline(df):
    df_proc = df.copy()
    
    for col in df_proc.columns:
        if df_proc[col].dtype == 'object':
            df_proc[col] = df_proc[col].fillna(df_proc[col].mode().iloc[0])
        else:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())
    
    if 'daily_social_media_time' in df_proc.columns and 'work_hours_per_day' in df_proc.columns:
        df_proc['social_minutes_per_work_hour'] = df_proc['daily_social_media_time'] / (df_proc['work_hours_per_day'] * 60 + 1)
        
    cat_cols = [c for c in ['gender', 'social_platform_preference', 'job_type'] if c in df_proc.columns]
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
        
    return df_proc

def run_experimento_1_regressao_linear(df_proc):
    print("\nIniciando experimento 1: baseline com regressao linear")
    
    target = 'perceived_productivity_score'
    if target not in df_proc.columns:
        print(f"Alvo '{target}' nao encontrado, pulando experimento 1")
        return

    X = df_proc.drop(target, axis=1)
    y = df_proc[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    model_r2 = r2_score(y_test, preds)
    model_rmse = rmse(y_test, preds)
    
    print(f"Resultado (Regressao Linear):")
    print(f"  R² (Coef. Determinaçao): {model_r2:.4f}")
    print(f"  RMSE (Erro Quadratico): {model_rmse:.4f}")
    
    try:
        coefs = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
        print("\n  Top 5 Coeficientes do Modelo:")
        print(coefs.head(5).to_string())
    except Exception as e:
        print(f"Nao foi possivel mostrar coeficientes: {e}")

def run_experimento_2_gradient_boosting(df_proc, output_dir):
    print("\nIniciando experimento 2: analise com Gradient Boosting")
    
    target = 'perceived_productivity_score'
    
    if target not in df_proc.columns:
        print(f"Colunas de score nao encontradas, pulando experimento 2")
        return

    X = df_proc.drop(target, axis=1, errors='ignore')
    y = df_proc[target]
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    model = GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=200)
    model.fit(X_train, y_train)

    try:
        importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        importances.head(25).plot(kind='barh')
        plt.gca().invert_yaxis()
        plt.xlabel("Importancia")
        plt.title("Feature Importances - Gradient Boosting")
        plt.tight_layout()
        
        output_filename = output_dir / "feature_importances_inicial.png"
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()
        
        print(f"Resultado (Gradient Boosting):")
        print(f"Grafico salvo: {output_filename}")
        print("Analisando grafico de importancia...")
        
    except Exception as e:
        print(f"Erro ao gerar grafico do experimento 2: {e}")

def main():
    if len(sys.argv) != 2:
        print("Erro: Forneça o caminho para o arquivo CSV.")
        print(f"Uso: python {sys.argv[0]} <caminho_para_o_dataset.csv>")
        sys.exit(1)
        
    caminho_csv = sys.argv[1]
    
    output_dir = Path("saida")
    output_dir.mkdir(exist_ok=True)
    print(f"Salvando resultados em: {output_dir.resolve()}")
    
    df_bruto = carregar_dados(caminho_csv)
    
    df_processado = preprocessamento_baseline(df_bruto)
    
    run_experimento_1_regressao_linear(df_processado)
    run_experimento_2_gradient_boosting(df_processado, output_dir)
    
if __name__ == "__main__":
    main()