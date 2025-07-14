# 🏦 Análisis de Ranking de Bancos en Colombia

Este proyecto presenta un análisis exploratorio y visual de entidades bancarias en Colombia con base en múltiples métricas como tasas de interés, número de clientes, nivel de satisfacción, reclamos y nivel de digitalización. El objetivo es generar un **ranking ponderado** que combine estas variables y facilite la comparación entre bancos.

---

## 📊 Objetivos

- Calcular promedios por banco para tasas, reclamos, satisfacción y uso de canales digitales.
- Normalizar las métricas para construir un ranking comparativo.
- Visualizar relaciones entre tasa de interés, digitalización y nivel de reclamos.
- Explorar la evolución temporal de métricas clave.
- Filtrar y comparar bancos específicos o periodos de tiempo (ej. últimos 6 meses).

---

## 📁 Dataset

El archivo utilizado es: `tasas_bancos_completo.csv`, que incluye las siguientes columnas:

- `Fecha`
- `Banco`
- `Tasa_Activa` (anual)
- `Clientes`
- `Satisfaccion_NPS` (Net Promoter Score)
- `Tiempo_Espera_Min`
- `Reclamos_Mensuales`
- `Uso_Canales_Digitales_%`

> ⚠️ Este dataset fue generado a partir de datos reales y simulados, siguiendo patrones publicados por la Superintendencia Financiera de Colombia y otras fuentes.

---

## ⚙️ Tecnologías

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly (interactivo)
- Google Colab

---

## 📈 Visualizaciones incluidas

- Ranking ponderado de bancos (gráfico de barras horizontal)
- Comparativo de Tasa, NPS y Reclamos
- Heatmap de correlaciones entre métricas
- Gráfico burbuja: Tasa vs. Digitalización vs. NPS
- Serie temporal de tasas de interés
- Filtros por banco y por fechas recientes (últimos 6 meses)

---

## 📌 ¿Qué demuestra este proyecto?

Este análisis refleja habilidades en:

- Limpieza y transformación de datos
- Agregaciones por categoría
- Normalización y construcción de indicadores compuestos
- Visualización avanzada y narrativa de datos
- Comparaciones temporales y segmentadas

---

## 👩‍💻 Autor

**Geraldin Carriazo**  
[LinkedIn](https://www.linkedin.com/in/geraldin-carriazo)
