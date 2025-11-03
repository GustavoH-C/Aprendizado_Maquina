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

def get_models_para_comparacao():
    return {
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=200),
        "Xgboost": XGBRegressor(random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1, eval_metric='rmse')
    }

def run_analise_corrigida(df_raw, output_dir):
    print("\nIniciando analise corrigida (pos-vazamento)")
    
    colunas_de_vazamento = ['actual_productivity_score', 'perceived_productivity_score', 'job_satisfaction_score']
    
    alvos_de_teste = ['perceived_productivity_score', 'actual_productivity_score']
    
    df_proc_avancado = preprocessamento_avancado(df_raw)
    
    for target in alvos_de_teste:
        if target not in df_proc_avancado.columns:
            print(f"Alvo '{target}' nao encontrado pulando sub-experimento")
            continue
            
        print(f"\nRodando analise corrigida para o alvo: [{target}]")
        
        cols_to_drop_from_X = [c for c in colunas_de_vazamento if c in df_proc_avancado.columns]
        
        X = df_proc_avancado.drop(columns=cols_to_drop_from_X, errors='ignore')
        y = df_proc_avancado[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
        
        models = get_models_para_comparacao()
        results = []
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        best_model = None
        best_model_name = ""
        
        for name, model in models.items():
            neg_mse_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
            cv_rmse = np.sqrt(-neg_mse_scores).mean()
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if name == "Xgboost":
                best_model = model
                best_model_name = name
            
            results.append({
                'modelo': name,
                'cv_rmse': cv_rmse,
                'test_rmse': rmse(y_test, preds),
                'test_mae': mean_absolute_error(y_test, preds),
                'test_r2': r2_score(y_test, preds)
            })
        
        df_results = pd.DataFrame(results).sort_values('test_rmse')
        tabela_filename = output_dir / f"tabela_resultados_sem_vazamento{target}.csv"
        df_results.to_csv(tabela_filename, index=False)
        
        print(f"    Tabela de resultados salva: {tabela_filename}")
        print(df_results.to_string())
        
        if target == 'perceived_productivity_score' and best_model:
            print(f"Gerando graficos finais para o alvo '{target}'")
            
            try:
                final_preds = best_model.predict(X_test)
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, final_preds, alpha=0.6)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel(f"Real ({target})")
                plt.ylabel("Predito")
                plt.title(f"Previsoes vs reais - {best_model_name} (sem vazamento)")
                plt.tight_layout()
                
                output_fig2 = output_dir / "previsoes_vs_reais_XGB_SEM_VAZAMENTO.png"
                plt.savefig(output_fig2, bbox_inches='tight')
                plt.close()
                print(f"    Grafico salvo: {output_fig2}")
            except Exception as e:
                print(f"Erro ao salvar grafico 'preds_vs_real': {e}")

            try:
                importances_final = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                plt.figure(figsize=(10, 8))
                importances_final.head(25).plot(kind='barh')
                plt.gca().invert_yaxis()
                plt.xlabel("Importancia (f-score)")
                plt.title(f"Feature importances - {best_model_name} (sem vazamento)")
                plt.tight_layout()
                
                output_fig3 = output_dir / "feature_importance_XGB_SEM_VAZAMENTO.png"
                plt.savefig(output_fig3, bbox_inches='tight')
                plt.close()
                print(f"    Grafico salvo: {output_fig3}")
            except Exception as e:
                print(f"Erro ao salvar grafico 'feature_importance': {e}")
                
            model_path = output_dir / "modelo_xgboost_final.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            print(f"Modelo final salvo: {model_path}")

def main():
    if len(sys.argv) != 2:
        print("Erro: forneca o caminho para o arquivo csv")
        print(f"Uso: python {sys.argv[0]} <caminho_para_o_datasetcsv>")
        sys.exit(1)
        
    caminho_csv = sys.argv[1]
    
    output_dir = Path("saida")
    output_dir.mkdir(exist_ok=True)
    print(f"Salvando resultados em: {output_dir.resolve()}")
    
    df_bruto = carregar_dados(caminho_csv)
    run_analise_corrigida(df_bruto, output_dir)

if __name__ == "__main__":
    main()