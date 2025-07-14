# ============================================================
# 0. Librerías principales
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from google.colab import files
import io
import plotly.express as px

# ============================================================
# 1. Subir archivo CSV (selecciona tasas_bancos_completo.csv)
# ============================================================
uploaded  = files.upload()
filename  = next(iter(uploaded))
df        = pd.read_csv(io.BytesIO(uploaded[filename]))
df['Fecha'] = pd.to_datetime(df['Fecha'])

# ============================================================
# 2. Promedios por banco (summary)
# ============================================================
summary = (
    df.groupby('Banco')
      .agg({
          'Tasa_Activa': 'mean',
          'Clientes': 'mean',
          'Satisfaccion_NPS': 'mean',
          'Tiempo_Espera_Min': 'mean',
          'Reclamos_Mensuales': 'mean',
          'Uso_Canales_Digitales_%': 'mean'
      })
      .reset_index()
)

# ============================================================
# 3. Normalización y Score ponderado
# ============================================================
beneficial_cols  = ['Clientes', 'Satisfaccion_NPS', 'Uso_Canales_Digitales_%']
detrimental_cols = ['Tasa_Activa', 'Tiempo_Espera_Min', 'Reclamos_Mensuales']

scaler = MinMaxScaler()
summary_norm = summary.copy()
summary_norm[beneficial_cols]  = scaler.fit_transform(summary[beneficial_cols])
summary_norm[detrimental_cols] = scaler.fit_transform(summary[detrimental_cols])
summary_norm[detrimental_cols] = 1 - summary_norm[detrimental_cols]           # menor = mejor

weights = {
    'Clientes':0.20,
    'Satisfaccion_NPS':0.25,
    'Uso_Canales_Digitales_%':0.15,
    'Tasa_Activa':0.20,
    'Tiempo_Espera_Min':0.10,
    'Reclamos_Mensuales':0.10
}

summary_norm['Score'] = (
      summary_norm['Clientes']              * weights['Clientes']
    + summary_norm['Satisfaccion_NPS']      * weights['Satisfaccion_NPS']
    + summary_norm['Uso_Canales_Digitales_%'] * weights['Uso_Canales_Digitales_%']
    + summary_norm['Tasa_Activa']           * weights['Tasa_Activa']
    + summary_norm['Tiempo_Espera_Min']     * weights['Tiempo_Espera_Min']
    + summary_norm['Reclamos_Mensuales']    * weights['Reclamos_Mensuales']
)

summary_final = summary.merge(summary_norm[['Banco','Score']], on='Banco') \
                       .sort_values('Score', ascending=False)

print('### Ranking ponderado de bancos ###\n')
print(summary_final[['Banco','Score']].to_string(index=False))

# ============================================================
# 4. VISUALIZACIONES
# ============================================================

## 4.1 Barra horizontal – Ranking
plt.figure(figsize=(8,4))
sns.barplot(data=summary_final, y='Banco', x='Score',
            palette='Blues_r', order=summary_final['Banco'])
plt.title('Ranking ponderado de bancos (mayor = mejor)')
plt.xlabel('Score'); plt.ylabel('')
for i, v in enumerate(summary_final['Score']):
    plt.text(v + 0.01, i, f'{v:.2f}', va='center')
plt.xlim(0, summary_final['Score'].max()+0.1)
plt.tight_layout(); plt.show()

## 4.2 Panel comparativo – Tasa, NPS, Reclamos
melt = summary.melt(id_vars='Banco',
                    value_vars=['Tasa_Activa','Satisfaccion_NPS','Reclamos_Mensuales'],
                    var_name='Métrica', value_name='Valor')
g = sns.catplot(data=melt, kind='bar',
                x='Valor', y='Banco', hue='Métrica',
                palette='Set2', height=6, aspect=1.4)
g.fig.suptitle('Comparativo de métricas clave por banco', fontsize=14)
g.set_xlabels(''); g.set_ylabels('')
plt.tight_layout(); plt.show()

## 4.3 Heatmap de correlaciones
corr_cols = ['Tasa_Activa','Clientes','Satisfaccion_NPS',
             'Tiempo_Espera_Min','Reclamos_Mensuales',
             'Uso_Canales_Digitales_%']
plt.figure(figsize=(6,5))
sns.heatmap(summary[corr_cols].corr(), annot=True, cmap='BrBG', fmt='.2f')
plt.title('Correlaciones entre métricas (promedio por banco)')
plt.tight_layout(); plt.show()

## 4.4 Bubble chart – Tasa vs Digitalización
plt.figure(figsize=(7,5))
sizes = summary['Clientes'] / 2_000_000
scatter = plt.scatter(summary['Tasa_Activa'],
                      summary['Uso_Canales_Digitales_%'],
                      s=sizes, alpha=0.65,
                      c=summary['Satisfaccion_NPS'],
                      cmap='coolwarm')
for _, row in summary.iterrows():
    plt.text(row['Tasa_Activa']+0.05,
             row['Uso_Canales_Digitales_%']+0.4,
             row['Banco'], fontsize=9)
cbar = plt.colorbar(scatter); cbar.set_label('NPS (%)')
plt.xlabel('Tasa de interés (%)')
plt.ylabel('Uso canales digitales (%)')
plt.title('Tasa vs Digitalización (tamaño = clientes, color = NPS)')
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

## 4.5 Serie temporal de tasas
plt.figure(figsize=(10,5))
for banco, d in df.groupby('Banco'):
    plt.plot(d['Fecha'], d['Tasa_Activa'], label=banco)
plt.fill_between(df['Fecha'],
                 df['Tasa_Activa'].min(),
                 df['Tasa_Activa'].max(),
                 alpha=0.05, color='gray')
plt.legend(); plt.ylabel('Tasa de interés (%)')
plt.title('Evolución mensual de la tasa por banco')
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

# ============================================================
# 5. Ejemplo de filtros
# ============================================================
## – Filtrar un banco
banco_objetivo = 'Bancolombia'
df_banco = df[df['Banco'] == banco_objetivo]

plt.figure(figsize=(8,4))
plt.plot(df_banco['Fecha'], df_banco['Reclamos_Mensuales'], marker='o')
plt.title(f'Reclamos mensuales – {banco_objetivo}')
plt.ylabel('Reclamos'); plt.xlabel('')
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()

## – Filtrar últimos 6 meses
ultimos_6m = df[df['Fecha'] >= df['Fecha'].max() - pd.DateOffset(months=6)]
print('\n=== Últimos 6 meses – muestra ===')
print(ultimos_6m[['Fecha','Banco','Tasa_Activa']].head())
