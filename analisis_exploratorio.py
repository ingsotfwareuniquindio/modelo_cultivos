# PASO 1: ANÁLISIS EXPLORATORIO Y TOP 20 CULTIVOS
# =================================================
# Ejecutar en Google Colab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("🚀 PASO 1: ANÁLISIS EXPLORATORIO Y SELECCIÓN TOP 20")
print("=" * 60)

# 1. CARGAR DATASET
print("\n📊 1. CARGANDO DATASET...")
df = pd.read_excel('cultivos.xlsx')
print(f"✅ Dataset cargado: {len(df):,} registros, {len(df.columns)} columnas")

# 2. INFORMACIÓN BÁSICA
print(f"\n📈 2. INFORMACIÓN BÁSICA:")
print(f"   • Registros totales: {len(df):,}")
print(f"   • Cultivos únicos: {df['Cultivo'].nunique():,}")
print(f"   • Departamentos: {df['Departamento'].nunique()}")
print(f"   • Periodo de datos: {df['FechaAnalisis'].dropna().count():,} fechas registradas")

# 3. ANÁLISIS DE CULTIVOS - ENCONTRAR TOP 20
print(f"\n🏆 3. ANÁLISIS DE CULTIVOS:")

# Contar cultivos y limpiar datos básicos
cultivos_count = df['Cultivo'].value_counts()

# Filtrar cultivos válidos (eliminar 'No indica', vacíos, etc.)
cultivos_validos = cultivos_count[
    ~cultivos_count.index.isin(['No indica', '', 'NO INDICA', 'Sin información'])
].dropna()

print(f"   📋 Cultivos válidos encontrados: {len(cultivos_validos)}")

# TOP 20 CULTIVOS
top_20_cultivos = cultivos_validos.head(20)

print(f"\n🥇 TOP 20 CULTIVOS MÁS FRECUENTES:")
print("-" * 50)
total_registros_top20 = 0

for i, (cultivo, count) in enumerate(top_20_cultivos.items(), 1):
    porcentaje = (count / len(df)) * 100
    total_registros_top20 += count
    print(f"{i:2d}. {cultivo:<25} | {count:5,} registros ({porcentaje:4.1f}%)")

cobertura_top20 = (total_registros_top20 / len(df)) * 100
print(f"\n📊 COBERTURA TOP 20: {total_registros_top20:,} registros ({cobertura_top20:.1f}% del dataset)")

# 4. ANÁLISIS DE DISTRIBUCIÓN POR RANGOS
print(f"\n📈 4. DISTRIBUCIÓN POR RANGOS DE FRECUENCIA:")

rangos = [
    (">1000", cultivos_validos[cultivos_validos > 1000]),
    ("500-1000", cultivos_validos[(cultivos_validos >= 500) & (cultivos_validos <= 1000)]),
    ("100-499", cultivos_validos[(cultivos_validos >= 100) & (cultivos_validos < 500)]),
    ("50-99", cultivos_validos[(cultivos_validos >= 50) & (cultivos_validos < 100)]),
    ("10-49", cultivos_validos[(cultivos_validos >= 10) & (cultivos_validos < 50)]),
    ("<10", cultivos_validos[cultivos_validos < 10])
]

for rango_nombre, rango_data in rangos:
    if len(rango_data) > 0:
        registros_rango = rango_data.sum()
        porcentaje_rango = (registros_rango / len(df)) * 100
        print(f"   🏷️  {rango_nombre:>8} registros: {len(rango_data):3d} cultivos | {registros_rango:6,} registros ({porcentaje_rango:4.1f}%)")

# 5. FILTRAR DATASET CON TOP 20
print(f"\n🔧 5. CREANDO DATASET FILTRADO:")

# Crear lista de cultivos top 20
lista_top20 = top_20_cultivos.index.tolist()

# Filtrar dataset
df_top20 = df[df['Cultivo'].isin(lista_top20)].copy()

print(f"   ✅ Dataset original: {len(df):,} registros")
print(f"   ✅ Dataset filtrado: {len(df_top20):,} registros")
print(f"   ✅ Reducción: {((len(df) - len(df_top20)) / len(df) * 100):.1f}%")

# 6. VERIFICAR CALIDAD DE DATOS EN VARIABLES CLAVE
print(f"\n🔍 6. CALIDAD DE DATOS EN VARIABLES PREDICTORAS:")

variables_predictoras = [
    'pH agua:suelo 2,5:1,0',
    'Materia orgánica (MO) %',
    'Fósforo (P) Bray II mg/kg',
    'Potasio (K) intercambiable cmol(+)/kg',
    'Calcio (Ca) intercambiable cmol(+)/kg',
    'Magnesio (Mg) intercambiable cmol(+)/kg',
    'Topografia',
    'Drenaje'
]

print("   Variable                                    | Válidos    | Faltantes  | % Válidos")
print("-" * 85)

for var in variables_predictoras:
    if var in df_top20.columns:
        # Contar valores válidos (no NaN, no 'ND', no vacíos)
        if var in ['Topografia', 'Drenaje']:
            # Variables categóricas
            validos = df_top20[var].notna().sum()
            validos -= df_top20[var].isin(['', 'ND', 'No indica']).sum()
        else:
            # Variables numéricas
            validos = pd.to_numeric(df_top20[var], errors='coerce').notna().sum()

        faltantes = len(df_top20) - validos
        porcentaje_validos = (validos / len(df_top20)) * 100

        status = "🟢" if porcentaje_validos >= 90 else "🟡" if porcentaje_validos >= 70 else "🔴"

        print(f"{status} {var:<42} | {validos:8,} | {faltantes:8,} | {porcentaje_validos:6.1f}%")

# 7. ANÁLISIS POR DEPARTAMENTO (TOP 20)
print(f"\n🗺️  7. DISTRIBUCIÓN GEOGRÁFICA (TOP 20 CULTIVOS):")

dep_top20 = df_top20['Departamento'].value_counts().head(10)
print("   Top 10 Departamentos con más registros:")
for dep, count in dep_top20.items():
    porcentaje = (count / len(df_top20)) * 100
    print(f"   📍 {dep:<20} | {count:5,} registros ({porcentaje:4.1f}%)")

# 8. ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS
print(f"\n📊 8. ESTADÍSTICAS DE VARIABLES DEL SUELO:")

variables_numericas = [
    'pH agua:suelo 2,5:1,0',
    'Materia orgánica (MO) %',
    'Fósforo (P) Bray II mg/kg',
    'Potasio (K) intercambiable cmol(+)/kg'
]

print("   Variable                  | Min     | Max      | Promedio | Mediana  | Desv.Est")
print("-" * 85)

for var in variables_numericas:
    if var in df_top20.columns:
        serie = pd.to_numeric(df_top20[var], errors='coerce').dropna()
        if len(serie) > 0:
            print(f"   {var:<25} | {serie.min():7.2f} | {serie.max():8.2f} | {serie.mean():8.2f} | {serie.median():8.2f} | {serie.std():8.2f}")

# 9. CREAR VISUALIZACIONES
print(f"\n📈 9. CREANDO VISUALIZACIONES...")

# Configurar estilo
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Top 20 cultivos
ax1 = axes[0, 0]
top_20_cultivos.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Top 20 Cultivos Más Frecuentes', fontsize=14, fontweight='bold')
ax1.set_xlabel('Cultivos', fontsize=12)
ax1.set_ylabel('Número de Registros', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=10)

# Gráfico 2: Distribución por departamento
ax2 = axes[0, 1]
dep_top20.head(8).plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Top 8 Departamentos (Top 20 Cultivos)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Departamentos', fontsize=12)
ax2.set_ylabel('Número de Registros', fontsize=12)
ax2.tick_params(axis='x', rotation=45, labelsize=10)

# Gráfico 3: Distribución de pH
ax3 = axes[1, 0]
ph_data = pd.to_numeric(df_top20['pH agua:suelo 2,5:1,0'], errors='coerce').dropna()
ax3.hist(ph_data, bins=30, color='orange', alpha=0.7, edgecolor='black')
ax3.set_title('Distribución de pH del Suelo', fontsize=14, fontweight='bold')
ax3.set_xlabel('pH', fontsize=12)
ax3.set_ylabel('Frecuencia', fontsize=12)

# Gráfico 4: Distribución de Materia Orgánica
ax4 = axes[1, 1]
mo_data = pd.to_numeric(df_top20['Materia orgánica (MO) %'], errors='coerce').dropna()
ax4.hist(mo_data, bins=30, color='brown', alpha=0.7, edgecolor='black')
ax4.set_title('Distribución de Materia Orgánica', fontsize=14, fontweight='bold')
ax4.set_xlabel('Materia Orgánica (%)', fontsize=12)
ax4.set_ylabel('Frecuencia', fontsize=12)

plt.tight_layout()
plt.show()

# 10. GUARDAR DATASET FILTRADO
print(f"\n💾 10. GUARDANDO DATASET FILTRADO:")

# Guardar dataset con top 20 cultivos
df_top20.to_csv('dataset_top20_cultivos.csv', index=False)
print(f"   ✅ Dataset guardado: 'dataset_top20_cultivos.csv'")
print(f"   📊 Registros: {len(df_top20):,}")
print(f"   🏷️  Cultivos: {len(lista_top20)}")

# Guardar lista de cultivos top 20
with open('lista_top20_cultivos.txt', 'w', encoding='utf-8') as f:
    f.write("TOP 20 CULTIVOS SELECCIONADOS PARA EL MODELO\n")
    f.write("=" * 50 + "\n\n")
    for i, (cultivo, count) in enumerate(top_20_cultivos.items(), 1):
        f.write(f"{i:2d}. {cultivo}: {count:,} registros\n")
    f.write(f"\nTotal registros: {total_registros_top20:,}")
    f.write(f"\nCobertura del dataset: {cobertura_top20:.1f}%")

print(f"   ✅ Lista guardada: 'lista_top20_cultivos.txt'")

# 11. RESUMEN FINAL
print(f"\n🎯 RESUMEN DEL PASO 1:")
print("=" * 50)
print(f"✅ Dataset original procesado: {len(df):,} registros")
print(f"✅ Top 20 cultivos identificados: {cobertura_top20:.1f}% cobertura")
print(f"✅ Dataset filtrado creado: {len(df_top20):,} registros")
print(f"✅ Variables predictoras evaluadas: {len(variables_predictoras)}")
print(f"✅ Calidad de datos verificada")
print(f"✅ Archivos guardados para siguientes pasos")

print(f"\n🚀 SIGUIENTE PASO: Limpieza y preparación de datos")
print("   Ejecutar: Paso 2 - Limpieza de Datos")

# Mostrar lista final para confirmación
print(f"\n📋 CULTIVOS SELECCIONADOS PARA EL MODELO:")
for i, cultivo in enumerate(lista_top20, 1):
    print(f"   {i:2d}. {cultivo}")

# Retornar datos para siguiente paso
print(f"\n✨ ¡PASO 1 COMPLETADO EXITOSAMENTE!")
