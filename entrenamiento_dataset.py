# MODELO TOP 5 CULTIVOS - CAMBIO RÁPIDO
# =====================================
# Usar dataset limpio existente, solo filtrar a Top 5

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🎯 MODELO TOP 5 CULTIVOS - PRUEBA RÁPIDA")
print("=" * 50)

# 1. CARGAR DATASET LIMPIO (YA PERFECTO DEL PASO 2)
print("\n📂 1. CARGANDO DATASET LIMPIO...")
df_clean = pd.read_csv('dataset_cleaned_ready_for_ml.csv')
print(f"✅ Dataset limpio cargado: {len(df_clean):,} registros")

# 2. FILTRAR SOLO TOP 5 CULTIVOS
print("\n🔍 2. FILTRANDO A TOP 5 CULTIVOS...")

# Los 5 cultivos más frecuentes de tu análisis
TOP_5_CULTIVOS = ['Cacao', 'Pastos', 'Aguacate', 'Caña panelera/azucar', 'Café']

print("🏆 Top 5 cultivos seleccionados:")
for i, cultivo in enumerate(TOP_5_CULTIVOS, 1):
    count = (df_clean['Cultivo'] == cultivo).sum()
    percentage = (count / len(df_clean)) * 100
    print(f"   {i}. {cultivo:<25} | {count:5,} registros ({percentage:4.1f}%)")

# Filtrar dataset
df_top5 = df_clean[df_clean['Cultivo'].isin(TOP_5_CULTIVOS)].copy()

print(f"\n📊 Resultado del filtro:")
print(f"   Dataset original: {len(df_clean):,} registros (20 cultivos)")
print(f"   Dataset filtrado: {len(df_top5):,} registros (5 cultivos)")
print(f"   Conservado: {len(df_top5)/len(df_clean)*100:.1f}% de los datos")

# 3. PREPARAR FEATURES Y TARGET
print("\n🔧 3. PREPARANDO DATOS...")

# Features (las mismas 8 variables)
feature_columns = [
    'pH agua:suelo 2,5:1,0',
    'Materia orgánica (MO) %',
    'Fósforo (P) Bray II mg/kg',
    'Potasio (K) intercambiable cmol(+)/kg',
    'Calcio (Ca) intercambiable cmol(+)/kg',
    'Magnesio (Mg) intercambiable cmol(+)/kg',
    'Topografia_encoded',
    'Drenaje_encoded'
]

X = df_top5[feature_columns]
y = df_top5['Cultivo']

print(f"✅ Features: {X.shape}")
print(f"✅ Target: {y.shape}")
print(f"✅ Cultivos únicos: {y.nunique()}")

# Verificar distribución de cultivos
print(f"\n📈 Distribución final de cultivos:")
cultivo_dist = y.value_counts()
for cultivo, count in cultivo_dist.items():
    percentage = (count / len(y)) * 100
    print(f"   🌱 {cultivo:<25} | {count:5,} registros ({percentage:4.1f}%)")

# 4. DIVISIÓN TRAIN/TEST
print("\n🔀 4. DIVISIÓN TRAIN/TEST...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Train: {len(X_train):,} registros")
print(f"✅ Test:  {len(X_test):,} registros")

# 5. ENTRENAR MODELO OPTIMIZADO
print("\n🚀 5. ENTRENANDO MODELO TOP 5...")

# Usar parámetros balanceados para 5 clases
rf_top5 = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,              # Un poco más profundo para 5 clases
    min_samples_split=5,       # Menos restrictivo
    min_samples_leaf=2,        # Menos restrictivo
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("🌲 Parámetros del modelo:")
print(f"   🌳 Árboles: {rf_top5.n_estimators}")
print(f"   📏 Profundidad: {rf_top5.max_depth}")
print(f"   🔢 Min split: {rf_top5.min_samples_split}")
print(f"   🍃 Min hoja: {rf_top5.min_samples_leaf}")

# Entrenar
import time
start_time = time.time()
rf_top5.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"✅ Modelo entrenado en {training_time:.2f} segundos")

# 6. EVALUAR MODELO
print("\n📊 6. EVALUACIÓN DEL MODELO TOP 5:")
print("-" * 40)

# Predicciones
y_train_pred = rf_top5.predict(X_train)
y_test_pred = rf_top5.predict(X_test)

# Métricas principales
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
overfitting = abs(train_accuracy - test_accuracy)

print(f"🎯 Precisión Train: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"🎯 Precisión Test:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"📈 Overfitting:    {overfitting:.4f} ({overfitting*100:.2f}%)")

# Métricas detalladas
precision_weighted = precision_score(y_test, y_test_pred, average='weighted')
recall_weighted = recall_score(y_test, y_test_pred, average='weighted')
f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

print(f"\n📈 Métricas detalladas:")
print(f"🎯 Precisión: {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
print(f"🔄 Recall:    {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
print(f"⚖️ F1-Score:  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")

# 7. COMPARACIÓN CON MODELO ANTERIOR
print(f"\n🆚 7. COMPARACIÓN CON MODELOS ANTERIORES:")
print("-" * 40)

modelos_anteriores = {
    '20 cultivos (original)': {'precision': 0.5156, 'f1': 0.5062},
    '20 cultivos (conservador)': {'precision': 0.3776, 'f1': 0.3770}
}

print("Modelo                     | Precisión | F1-Score | Cultivos")
print("-" * 65)
for nombre, metricas in modelos_anteriores.items():
    print(f"{nombre:<26} | {metricas['precision']:8.4f} | {metricas['f1']:8.4f} | 20")

print(f"{'Top 5 cultivos (nuevo)':<26} | {test_accuracy:8.4f} | {f1_weighted:8.4f} | 5")

# Calcular mejoras
mejor_anterior = max([m['precision'] for m in modelos_anteriores.values()])
mejora = test_accuracy - mejor_anterior

print(f"\n🚀 Mejora vs mejor anterior: {mejora:+.4f} ({mejora*100:+.2f}%)")

# 8. PERFORMANCE POR CULTIVO
print(f"\n🌱 8. PERFORMANCE POR CULTIVO:")
print("-" * 40)

# Reporte detallado
classification_rep = classification_report(y_test, y_test_pred, output_dict=True)

print("Cultivo                  | Precisión | Recall | F1-Score | Soporte")
print("-" * 70)

for cultivo in TOP_5_CULTIVOS:
    if cultivo in classification_rep:
        metrics = classification_rep[cultivo]
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1-score']
        support = int(metrics['support'])
        print(f"{cultivo:<25} | {precision:8.3f} | {recall:6.3f} | {f1:8.3f} | {support:7d}")

# 9. IMPORTANCIA DE VARIABLES
print(f"\n🔍 9. IMPORTANCIA DE VARIABLES:")
print("-" * 40)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_top5.feature_importances_
}).sort_values('importance', ascending=False)

print("Ranking de importancia:")
for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
    importance_pct = row['importance'] * 100
    bar = "█" * max(1, int(importance_pct / 3))
    print(f"{i}. {row['feature']:<35} | {importance_pct:5.2f}% {bar}")

# 10. GUARDAR MODELO TOP 5
print(f"\n💾 10. GUARDANDO MODELO TOP 5...")

joblib.dump(rf_top5, 'modelo_top5_cultivos.pkl')
print("✅ Modelo Top 5 guardado: 'modelo_top5_cultivos.pkl'")

# Información completa
model_info_top5 = {
    'model': rf_top5,
    'cultivos': TOP_5_CULTIVOS,
    'feature_names': list(X.columns),
    'target_classes': list(rf_top5.classes_),
    'metrics': {
        'test_accuracy': test_accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'overfitting_score': overfitting
    },
    'data_info': {
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'total_samples': len(X),
        'original_samples': len(df_clean),
        'data_retention': len(df_top5)/len(df_clean)
    }
}

joblib.dump(model_info_top5, 'modelo_top5_info.pkl')
print("✅ Info del modelo guardada: 'modelo_top5_info.pkl'")

# 11. RESUMEN FINAL
print(f"\n🎯 RESUMEN FINAL - MODELO TOP 5:")
print("=" * 50)
print(f"✅ Cultivos: {len(TOP_5_CULTIVOS)} (vs 20 anteriores)")
print(f"✅ Registros: {len(df_top5):,} ({len(df_top5)/len(df_clean)*100:.1f}% del dataset)")
print(f"✅ Precisión: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"✅ F1-Score: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
print(f"✅ Overfitting: {overfitting:.4f} ({'Controlado' if overfitting < 0.1 else 'Alto'})")
print(f"✅ Tiempo: {training_time:.2f} segundos")

# Evaluación del resultado
if test_accuracy > 0.80:
    status = "🎉 ¡EXCELENTE!"
    recommendation = "Listo para producción"
elif test_accuracy > 0.65:
    status = "👍 BUENO"
    recommendation = "Modelo útil, se puede optimizar más"
elif test_accuracy > 0.50:
    status = "😐 REGULAR"
    recommendation = "Funciona, pero necesita mejoras"
else:
    status = "😞 BAJO"
    recommendation = "Necesita trabajo adicional"

print(f"\n{status}")
print(f"📋 Evaluación: {recommendation}")

# Comparación específica
baseline_random = 1.0 / len(TOP_5_CULTIVOS)  # 20% para 5 clases
improvement_vs_random = test_accuracy - baseline_random

print(f"\n📊 Comparación con predicción aleatoria:")
print(f"   🎲 Aleatoria: {baseline_random:.4f} ({baseline_random*100:.2f}%)")
print(f"   🤖 Tu modelo: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   🚀 Mejora: {improvement_vs_random:+.4f} ({improvement_vs_random*100:+.2f}%)")

if test_accuracy > 0.60:
    print(f"\n🚀 SIGUIENTE PASO: Crear función de predicción")
else:
    print(f"\n🔧 CONSIDERACIONES: Revisar si se necesitan más variables")

print(f"\n✨ ¡MODELO TOP 5 COMPLETADO!")
