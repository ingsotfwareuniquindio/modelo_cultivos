# PASO 2: LIMPIEZA Y PREPARACIÓN DE DATOS
# =========================================
# Ejecutar en Google Colab después del Paso 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("🧹 PASO 2: LIMPIEZA Y PREPARACIÓN DE DATOS")
print("=" * 60)

# 1. CARGAR DATASET FILTRADO DEL PASO 1
print("\n📂 1. CARGANDO DATASET FILTRADO...")
df = pd.read_csv('dataset_top20_cultivos.csv')
print(f"✅ Dataset cargado: {len(df):,} registros, {len(df.columns)} columnas")
print(f"✅ Cultivos únicos: {df['Cultivo'].nunique()}")

# Variables que vamos a usar para el modelo
FEATURES_NUMERICAS = [
    'pH agua:suelo 2,5:1,0',
    'Materia orgánica (MO) %',
    'Fósforo (P) Bray II mg/kg',
    'Potasio (K) intercambiable cmol(+)/kg',
    'Calcio (Ca) intercambiable cmol(+)/kg',
    'Magnesio (Mg) intercambiable cmol(+)/kg'
]

FEATURES_CATEGORICAS = ['Topografia', 'Drenaje']
TARGET = 'Cultivo'

ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS

print(f"\n📊 Variables para el modelo:")
print(f"   • Numéricas: {len(FEATURES_NUMERICAS)}")
print(f"   • Categóricas: {len(FEATURES_CATEGORICAS)}")
print(f"   • Target: {TARGET}")

# 2. ANÁLISIS INICIAL DE DATOS FALTANTES
print(f"\n🔍 2. ANÁLISIS DE DATOS FALTANTES:")
print("-" * 50)

def analyze_missing_data(df, features):
    """Analiza datos faltantes de manera detallada"""
    missing_info = []

    for feature in features:
        if feature in df.columns:
            # Contar diferentes tipos de valores faltantes
            total_rows = len(df)

            if feature in FEATURES_NUMERICAS:
                # Para variables numéricas
                numeric_series = pd.to_numeric(df[feature], errors='coerce')
                nan_count = numeric_series.isna().sum()
                nd_count = (df[feature] == 'ND').sum() if df[feature].dtype == 'object' else 0
                empty_count = (df[feature] == '').sum() if df[feature].dtype == 'object' else 0
            else:
                # Para variables categóricas
                nan_count = df[feature].isna().sum()
                nd_count = (df[feature] == 'ND').sum()
                empty_count = (df[feature] == '').sum()
                no_indica_count = (df[feature] == 'No indica').sum()
                nd_count += no_indica_count

            total_missing = nan_count + nd_count + empty_count
            valid_count = total_rows - total_missing
            valid_percentage = (valid_count / total_rows) * 100

            missing_info.append({
                'feature': feature,
                'valid': valid_count,
                'missing': total_missing,
                'percentage_valid': valid_percentage
            })

            # Mostrar información
            status = "🟢" if valid_percentage >= 90 else "🟡" if valid_percentage >= 75 else "🔴"
            print(f"{status} {feature:<45} | {valid_count:6,} válidos ({valid_percentage:5.1f}%)")

    return missing_info

missing_analysis = analyze_missing_data(df, ALL_FEATURES)

# 3. ELIMINAR REGISTROS CRÍTICOS
print(f"\n🗑️  3. ELIMINANDO REGISTROS CRÍTICOS:")

def count_missing_per_row(row, features):
    """Cuenta cuántas variables faltan por registro"""
    missing_count = 0
    for feature in features:
        if feature in df.columns:
            value = row[feature]
            if pd.isna(value) or value == 'ND' or value == '' or value == 'No indica':
                missing_count += 1
    return missing_count

# Contar variables faltantes por registro
df['missing_count'] = df.apply(lambda row: count_missing_per_row(row, ALL_FEATURES), axis=1)

# Mostrar distribución de registros por cantidad de variables faltantes
print("   Distribución de registros por variables faltantes:")
missing_distribution = df['missing_count'].value_counts().sort_index()
for missing_vars, count in missing_distribution.items():
    percentage = (count / len(df)) * 100
    print(f"   📊 {missing_vars} variables faltantes: {count:5,} registros ({percentage:4.1f}%)")

# Eliminar registros con más de 4 variables faltantes (de 8 totales)
THRESHOLD_MISSING = 4
before_elimination = len(df)
df_clean = df[df['missing_count'] <= THRESHOLD_MISSING].copy()
df_clean = df_clean.drop('missing_count', axis=1)

eliminated = before_elimination - len(df_clean)
print(f"\n   ✅ Eliminados: {eliminated:,} registros con >{THRESHOLD_MISSING} variables faltantes")
print(f"   ✅ Conservados: {len(df_clean):,} registros ({len(df_clean)/before_elimination*100:.1f}%)")

# 4. IMPUTACIÓN INTELIGENTE POR CULTIVO
print(f"\n🔧 4. IMPUTACIÓN POR MEDIANA DE CULTIVO:")
print("-" * 50)

# Diccionario para guardar las medianas por cultivo
medians_by_crop = {}

for feature in FEATURES_NUMERICAS:
    if feature in df_clean.columns:
        print(f"\n   🧮 Procesando: {feature}")

        # Convertir a numérico
        df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')

        # Calcular medianas por cultivo
        medians_by_crop[feature] = df_clean.groupby(TARGET)[feature].median().to_dict()

        # Mostrar algunas medianas de ejemplo
        sample_crops = list(medians_by_crop[feature].keys())[:5]
        for crop in sample_crops:
            median_val = medians_by_crop[feature][crop]
            if not pd.isna(median_val):
                print(f"       📈 {crop}: mediana = {median_val:.2f}")

        # Imputar valores faltantes por cultivo
        missing_before = df_clean[feature].isna().sum()

        for cultivo in df_clean[TARGET].unique():
            # Máscara para este cultivo con valores faltantes
            mask = (df_clean[TARGET] == cultivo) & (df_clean[feature].isna())

            if mask.any() and cultivo in medians_by_crop[feature]:
                median_value = medians_by_crop[feature][cultivo]
                if not pd.isna(median_value):
                    df_clean.loc[mask, feature] = median_value

        # Imputar con mediana global cualquier valor que quede faltante
        remaining_missing = df_clean[feature].isna().sum()
        if remaining_missing > 0:
            global_median = df_clean[feature].median()
            df_clean[feature].fillna(global_median, inplace=True)

        missing_after = df_clean[feature].isna().sum()
        imputed = missing_before - missing_after
        print(f"       ✅ Imputados: {imputed:,} valores (quedan {missing_after} faltantes)")

# 5. TRATAMIENTO DE OUTLIERS EXTREMOS
print(f"\n⚠️  5. TRATAMIENTO DE OUTLIERS EXTREMOS:")
print("-" * 50)

# Definir rangos de valores posibles (no ideales, sino POSIBLES)
OUTLIER_RANGES = {
    'pH agua:suelo 2,5:1,0': (2.5, 9.5),  # Rango posible en suelos naturales
    'Materia orgánica (MO) %': (0, 25),   # Rango posible
    'Fósforo (P) Bray II mg/kg': (0, 500), # Rango posible
    'Potasio (K) intercambiable cmol(+)/kg': (0, 5),
    'Calcio (Ca) intercambiable cmol(+)/kg': (0, 50),
    'Magnesio (Mg) intercambiable cmol(+)/kg': (0, 10)
}

total_outliers_corrected = 0

for feature, (min_val, max_val) in OUTLIER_RANGES.items():
    if feature in df_clean.columns:
        # Identificar outliers extremos
        outliers_mask = (df_clean[feature] < min_val) | (df_clean[feature] > max_val)
        n_outliers = outliers_mask.sum()

        if n_outliers > 0:
            print(f"\n   🔍 {feature}:")
            print(f"       ⚠️ Outliers extremos encontrados: {n_outliers}")

            # Mostrar algunos ejemplos
            outlier_values = df_clean.loc[outliers_mask, feature].head(5)
            print(f"       📊 Ejemplos de valores extremos: {outlier_values.tolist()}")

            # Reemplazar outliers con mediana del cultivo correspondiente
            for cultivo in df_clean[TARGET].unique():
                cultivo_outliers_mask = outliers_mask & (df_clean[TARGET] == cultivo)
                if cultivo_outliers_mask.any():
                    # Usar mediana del cultivo si está disponible
                    if cultivo in medians_by_crop[feature]:
                        replacement_value = medians_by_crop[feature][cultivo]
                    else:
                        replacement_value = df_clean[feature].median()

                    df_clean.loc[cultivo_outliers_mask, feature] = replacement_value

            total_outliers_corrected += n_outliers
            print(f"       ✅ Corregidos: {n_outliers} valores extremos")

print(f"\n   📊 Total de outliers extremos corregidos: {total_outliers_corrected}")

# 6. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
print(f"\n🏷️  6. CODIFICACIÓN DE VARIABLES CATEGÓRICAS:")
print("-" * 50)

# Diccionario para guardar los encoders
encoders = {}

# Topografía (orden lógico: fácil → difícil)
print(f"\n   🗺️  Codificando Topografía:")
topografia_values = df_clean['Topografia'].value_counts()
print(f"       📊 Valores únicos encontrados: {topografia_values.to_dict()}")

topografia_mapping = {
    'Plano': 0,
    'Plano y ondulado': 1,
    'Ondulado': 2,
    'Pendiente': 3,
    'No indica': 1,  # Default a intermedio
    '': 1,           # Default a intermedio
    np.nan: 1        # Default a intermedio
}

df_clean['Topografia_encoded'] = df_clean['Topografia'].map(topografia_mapping)
df_clean['Topografia_encoded'].fillna(1, inplace=True)  # Ondulado como default
encoders['Topografia'] = topografia_mapping

# Convert np.nan key to string for printing
printable_topografia_mapping = {str(k) if pd.isna(k) else k: v for k, v in topografia_mapping.items()}
print(f"       ✅ Codificación aplicada: {dict(sorted(printable_topografia_mapping.items()))}")
print(f"       📈 Distribución codificada: {df_clean['Topografia_encoded'].value_counts().sort_index().to_dict()}")

# Drenaje (orden lógico: malo → excelente)
print(f"\n   💧 Codificando Drenaje:")
drenaje_values = df_clean['Drenaje'].value_counts()
print(f"       📊 Valores únicos encontrados: {drenaje_values.to_dict()}")

drenaje_mapping = {
    'Malo': 0,
    'Regular': 1,
    'Bueno': 2,
    'Excelente': 3,
    'No indica': 2,  # Default a bueno
    '': 2,           # Default a bueno
    np.nan: 2        # Default a bueno
}

df_clean['Drenaje_encoded'] = df_clean['Drenaje'].map(drenaje_mapping)
df_clean['Drenaje_encoded'].fillna(2, inplace=True)  # Bueno como default
encoders['Drenaje'] = drenaje_mapping

# Convert np.nan key to string for printing
printable_drenaje_mapping = {str(k) if pd.isna(k) else k: v for k, v in drenaje_mapping.items()}
print(f"       ✅ Codificación aplicada: {dict(sorted(printable_drenaje_mapping.items()))}")
print(f"       📈 Distribución codificada: {df_clean['Drenaje_encoded'].value_counts().sort_index().to_dict()}")

# 7. PREPARAR FEATURES FINALES
print(f"\n📋 7. PREPARANDO FEATURES FINALES:")

# Lista final de features para el modelo
FINAL_FEATURES = [
    'pH agua:suelo 2,5:1,0',
    'Materia orgánica (MO) %',
    'Fósforo (P) Bray II mg/kg',
    'Potasio (K) intercambiable cmol(+)/kg',
    'Calcio (Ca) intercambiable cmol(+)/kg',
    'Magnesio (Mg) intercambiable cmol(+)/kg',
    'Topografia_encoded',
    'Drenaje_encoded'
]

# Verificar que todas las features estén disponibles
available_features = [col for col in FINAL_FEATURES if col in df_clean.columns]
print(f"   ✅ Features finales disponibles: {len(available_features)}/{len(FINAL_FEATURES)}")

for i, feature in enumerate(available_features, 1):
    print(f"   {i:2d}. {feature}")

# 8. NORMALIZACIÓN CON STANDARDSCALER
print(f"\n📊 8. NORMALIZACIÓN DE VARIABLES NUMÉRICAS:")
print("-" * 50)

# Separar features numéricas de las categóricas codificadas
numeric_features = [col for col in available_features if not col.endswith('_encoded')]
categorical_features = [col for col in available_features if col.endswith('_encoded')]

print(f"   🔢 Variables numéricas a normalizar: {len(numeric_features)}")
print(f"   🏷️  Variables categóricas (no normalizar): {len(categorical_features)}")

# Mostrar estadísticas antes de normalizar
print(f"\n   📈 Estadísticas ANTES de normalizar:")
print("   Variable                                    | Min      | Max       | Media    | Std      ")
print("-" * 90)

for feature in numeric_features:
    values = df_clean[feature]
    print(f"   {feature:<42} | {values.min():8.2f} | {values.max():9.2f} | {values.mean():8.2f} | {values.std():8.2f}")

# Aplicar StandardScaler solo a variables numéricas
scaler = StandardScaler()
X_numeric = df_clean[numeric_features]
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Reemplazar valores normalizados en el DataFrame
for i, feature in enumerate(numeric_features):
    df_clean[feature] = X_numeric_scaled[:, i]

print(f"\n   📈 Estadísticas DESPUÉS de normalizar:")
print("   Variable                                    | Min      | Max       | Media    | Std      ")
print("-" * 90)

for feature in numeric_features:
    values = df_clean[feature]
    print(f"   {feature:<42} | {values.min():8.2f} | {values.max():9.2f} | {values.mean():8.2f} | {values.std():8.2f}")

# 9. VERIFICACIÓN FINAL DE CALIDAD
print(f"\n✅ 9. VERIFICACIÓN FINAL DE CALIDAD:")
print("-" * 50)

# Preparar matrices X e y finales
X = df_clean[available_features].copy()
y = df_clean[TARGET].copy()

# Verificar datos faltantes
missing_final = X.isnull().sum().sum()
print(f"   📊 Datos faltantes en X: {missing_final}")
print(f"   📊 Datos faltantes en y: {y.isnull().sum()}")

# Verificar distribución de cultivos
cultivos_distribution = y.value_counts()
print(f"   🌱 Cultivos únicos: {len(cultivos_distribution)}")
print(f"   📈 Registro mínimo por cultivo: {cultivos_distribution.min()}")
print(f"   📈 Registro máximo por cultivo: {cultivos_distribution.max()}")

# Mostrar top 10 cultivos con más registros
print(f"\n   🏆 Top 10 cultivos en dataset limpio:")
for i, (cultivo, count) in enumerate(cultivos_distribution.head(10).items(), 1):
    percentage = (count / len(y)) * 100
    print(f"   {i:2d}. {cultivo:<25} | {count:4,} registros ({percentage:4.1f}%)")

# 10. CREAR VISUALIZACIONES DE CALIDAD
print(f"\n📈 10. CREANDO VISUALIZACIONES DE CALIDAD...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Gráfico 1: Distribución de cultivos
ax1 = axes[0, 0]
cultivos_distribution.head(15).plot(kind='bar', ax=ax1, color='lightblue')
ax1.set_title('Distribución de Cultivos (Dataset Limpio)', fontweight='bold')
ax1.set_xlabel('Cultivos')
ax1.set_ylabel('Número de Registros')
ax1.tick_params(axis='x', rotation=45, labelsize=8)

# Gráfico 2: Distribución de pH normalizado
ax2 = axes[0, 1]
ax2.hist(df_clean['pH agua:suelo 2,5:1,0'], bins=30, color='green', alpha=0.7, edgecolor='black')
ax2.set_title('Distribución de pH (Normalizado)', fontweight='bold')
ax2.set_xlabel('pH (normalizado)')
ax2.set_ylabel('Frecuencia')

# Gráfico 3: Distribución de Materia Orgánica normalizada
ax3 = axes[0, 2]
ax3.hist(df_clean['Materia orgánica (MO) %'], bins=30, color='brown', alpha=0.7, edgecolor='black')
ax3.set_title('Distribución de Materia Orgánica (Normalizado)', fontweight='bold')
ax3.set_xlabel('Materia Orgánica (normalizada)')
ax3.set_ylabel('Frecuencia')

# Gráfico 4: Boxplot de Fósforo por top 5 cultivos
ax4 = axes[1, 0]
top5_cultivos = cultivos_distribution.head(5).index
df_top5 = df_clean[df_clean[TARGET].isin(top5_cultivos)]
df_top5.boxplot(column='Fósforo (P) Bray II mg/kg', by=TARGET, ax=ax4)
ax4.set_title('Distribución de Fósforo por Cultivo (Top 5)', fontweight='bold')
ax4.set_xlabel('Cultivo')
ax4.set_ylabel('Fósforo (normalizado)')

# Gráfico 5: Distribución de Topografía codificada
ax5 = axes[1, 1]
topografia_dist = df_clean['Topografia_encoded'].value_counts().sort_index()
topografia_dist.plot(kind='bar', ax=ax5, color='orange')
ax5.set_title('Distribución de Topografía (Codificada)', fontweight='bold')
ax5.set_xlabel('Código de Topografía')
ax5.set_ylabel('Frecuencia')

# Gráfico 6: Distribución de Drenaje codificado
ax6 = axes[1, 2]
drenaje_dist = df_clean['Drenaje_encoded'].value_counts().sort_index()
drenaje_dist.plot(kind='bar', ax=ax6, color='cyan')
ax6.set_title('Distribución de Drenaje (Codificado)', fontweight='bold')
ax6.set_xlabel('Código de Drenaje')
ax6.set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# 11. GUARDAR RESULTADOS
print(f"\n💾 11. GUARDANDO RESULTADOS:")

# Guardar dataset limpio completo
df_clean.to_csv('dataset_cleaned_ready_for_ml.csv', index=False)
print(f"   ✅ Dataset limpio guardado: 'dataset_cleaned_ready_for_ml.csv'")

# Guardar matrices X e y separadas
X.to_csv('X_features_cleaned.csv', index=False)
y.to_csv('y_target_cleaned.csv', index=False)
print(f"   ✅ Features (X) guardadas: 'X_features_cleaned.csv'")
print(f"   ✅ Target (y) guardado: 'y_target_cleaned.csv'")

# Guardar información de preprocesamiento
import joblib

preprocessing_info = {
    'scaler': scaler,
    'medians_by_crop': medians_by_crop,
    'encoders': encoders,
    'feature_names': available_features,
    'outlier_ranges': OUTLIER_RANGES
}

joblib.dump(preprocessing_info, 'preprocessing_info.pkl')
print(f"   ✅ Info de preprocesamiento guardada: 'preprocessing_info.pkl'")

# 12. RESUMEN FINAL
print(f"\n🎯 RESUMEN FINAL DEL PASO 2:")
print("=" * 50)
print(f"✅ Registros procesados: {len(df):,} → {len(df_clean):,} ({len(df_clean)/len(df)*100:.1f}% conservado)")
print(f"✅ Variables faltantes imputadas por mediana de cultivo")
print(f"✅ {total_outliers_corrected:,} outliers extremos corregidos")
print(f"✅ Variables categóricas codificadas ordinalmente")
print(f"✅ Variables numéricas normalizadas con StandardScaler")
print(f"✅ Dataset final: {X.shape[0]:,} registros × {X.shape[1]} features")
print(f"✅ Cultivos únicos: {len(y.unique())} clases")
print(f"✅ Datos faltantes restantes: {missing_final} (0%)")

print(f"\n🚀 SIGUIENTE PASO: División train/test y entrenamiento del modelo")
print("   Ejecutar: Paso 3 - Entrenamiento del Modelo")

print(f"\n✨ ¡PASO 2 COMPLETADO EXITOSAMENTE!")
print(f"   📊 Dataset limpio y listo para machine learning")
print(f"   🎯 Siguiente: Entrenar modelo con Random Forest")
