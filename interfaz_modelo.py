# INTERFAZ HTML SIMPLE - FUNCIONA 100% SIN DEPENDENCIAS
# ====================================================
# Copiar y pegar en UNA SOLA CELDA

from IPython.display import HTML, display
import json

# Función de predicción
def predecir_cultivo_final(ph, mo, fosforo, potasio, calcio, magnesio, topografia, drenaje):
    """Sistema de recomendación integrado"""
    
    scores = {'Cacao': 0, 'Pastos': 0, 'Aguacate': 0, 'Caña panelera': 0, 'Café': 0}
    
    # Evaluación por pH
    if ph < 5.5:
        scores['Café'] += 4; scores['Cacao'] += 2
    elif ph < 6.5:
        scores['Aguacate'] += 4; scores['Cacao'] += 3; scores['Café'] += 1
    elif ph < 7.5:
        scores['Pastos'] += 4; scores['Aguacate'] += 2; scores['Cacao'] += 1
    else:
        scores['Caña panelera'] += 4; scores['Pastos'] += 2
    
    # Evaluación por materia orgánica
    if mo >= 4.0:
        scores['Aguacate'] += 3; scores['Cacao'] += 3; scores['Café'] += 1
    elif mo >= 2.0:
        scores['Cacao'] += 2; scores['Pastos'] += 2; scores['Café'] += 1
    else:
        scores['Caña panelera'] += 2; scores['Pastos'] += 1
    
    # Evaluación por fósforo
    if fosforo >= 30:
        scores['Cacao'] += 2; scores['Aguacate'] += 2
    elif fosforo >= 15:
        scores['Aguacate'] += 1; scores['Café'] += 1; scores['Cacao'] += 1
    
    # Evaluación por topografía
    if topografia == "Pendiente":
        scores['Café'] += 3; scores['Cacao'] += 1
    elif topografia == "Plano":
        scores['Caña panelera'] += 3; scores['Pastos'] += 2
    else:
        scores['Aguacate'] += 2; scores['Cacao'] += 1
    
    # Evaluación por drenaje
    if drenaje in ["Bueno", "Excelente"]:
        scores['Aguacate'] += 2; scores['Cacao'] += 1
    elif drenaje == "Regular":
        scores['Pastos'] += 2; scores['Café'] += 1
    
    # Calcular resultado
    cultivo_ganador = max(scores, key=scores.get)
    score_max = scores[cultivo_ganador]
    confianza = min(score_max / 15.0, 0.95)
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return cultivo_ganador, confianza, ranking

# HTML completamente funcional
html_interface = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌱 Recomendador de Cultivos IA</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4CAF50, #2E7D32);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .content {
            padding: 40px;
        }
        
        .form-section {
            margin-bottom: 30px;
        }
        
        .form-section h3 {
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #4CAF50;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            font-weight: 600;
            color: #34495e;
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
        }
        
        .btn-container {
            text-align: center;
            margin: 30px 0;
        }
        
        .btn-predict {
            padding: 15px 40px;
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        }
        
        .examples {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .example-btn {
            padding: 8px 15px;
            margin: 5px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        
        .example-btn:hover {
            background: #0056b3;
        }
        
        .result {
            margin-top: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            text-align: center;
            display: none;
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result h2 {
            font-size: 2.5em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .confidence {
            font-size: 1.5em;
            margin: 20px 0;
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
        }
        
        .ranking {
            margin-top: 25px;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
        }
        
        .ranking h3 {
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .crop-item {
            background: rgba(255,255,255,0.15);
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 16px;
        }
        
        .crop-item.winner {
            background: rgba(255,215,0,0.3);
            border: 2px solid #FFD700;
        }
        
        .interpretation {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
            font-size: 16px;
            line-height: 1.5;
        }
        
        @media (max-width: 768px) {
            .content {
                padding: 20px;
            }
            .header h1 {
                font-size: 2em;
            }
            .form-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌱 Recomendador de Cultivos IA</h1>
            <p>Sistema inteligente para recomendación de cultivos basado en análisis de suelo</p>
        </div>
        
        <div class="content">
            <div class="examples">
                <h3>🧪 Ejemplos Rápidos para Probar</h3>
                <button class="example-btn" onclick="loadExample(6.5, 4.2, 28, 0.8, 12, 3.5, 'Ondulado', 'Bueno')">🥑 Aguacate</button>
                <button class="example-btn" onclick="loadExample(4.8, 6.1, 15, 0.4, 5.2, 1.8, 'Pendiente', 'Regular')">☕ Café</button>
                <button class="example-btn" onclick="loadExample(7.8, 2.1, 45, 1.2, 18, 4.2, 'Plano', 'Excelente')">🌾 Caña</button>
                <button class="example-btn" onclick="loadExample(6.0, 3.5, 20, 0.6, 8, 2.5, 'Ondulado', 'Bueno')">🍫 Cacao</button>
            </div>
            
            <form id="soilForm">
                <div class="form-section">
                    <h3>🧪 Análisis Químico del Suelo</h3>
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="ph">pH del suelo (3.0 - 9.0)</label>
                            <input type="number" id="ph" step="0.1" min="3" max="9" value="6.5" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="mo">Materia Orgánica (%)</label>
                            <input type="number" id="mo" step="0.1" min="0" max="15" value="4.0" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="fosforo">Fósforo disponible (mg/kg)</label>
                            <input type="number" id="fosforo" step="0.1" min="0" value="25.0" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="potasio">Potasio intercambiable (cmol+/kg)</label>
                            <input type="number" id="potasio" step="0.01" min="0" value="0.7" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="calcio">Calcio intercambiable (cmol+/kg)</label>
                            <input type="number" id="calcio" step="0.1" min="0" value="10.0" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="magnesio">Magnesio intercambiable (cmol+/kg)</label>
                            <input type="number" id="magnesio" step="0.1" min="0" value="3.0" required>
                        </div>
                    </div>
                </div>
                
                <div class="form-section">
                    <h3>🌍 Características del Terreno</h3>
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="topografia">Topografía del terreno</label>
                            <select id="topografia" required>
                                <option value="Plano">Plano</option>
                                <option value="Ondulado" selected>Ondulado</option>
                                <option value="Pendiente">Pendiente</option>
                                <option value="Plano y ondulado">Plano y ondulado</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="drenaje">Capacidad de drenaje</label>
                            <select id="drenaje" required>
                                <option value="Malo">Malo</option>
                                <option value="Regular">Regular</option>
                                <option value="Bueno" selected>Bueno</option>
                                <option value="Excelente">Excelente</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="btn-container">
                    <button type="submit" class="btn-predict">🔮 Obtener Recomendación</button>
                </div>
            </form>
            
            <div class="result" id="result">
                <h2 id="cropName">🌱 Cultivo Recomendado</h2>
                <div class="confidence" id="confidence">Confianza: 0%</div>
                
                <div class="ranking">
                    <h3>🏆 Ranking Completo de Cultivos</h3>
                    <div id="rankingList"></div>
                </div>
                
                <div class="interpretation" id="interpretation">
                    <strong>💡 Interpretación:</strong> <span id="interpretationText"></span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Función de predicción integrada en JavaScript
        function predictCrop(ph, mo, fosforo, potasio, calcio, magnesio, topografia, drenaje) {
            const scores = {
                'Cacao': 0,
                'Pastos': 0, 
                'Aguacate': 0,
                'Caña panelera': 0,
                'Café': 0
            };
            
            // Evaluación por pH
            if (ph < 5.5) {
                scores['Café'] += 4;
                scores['Cacao'] += 2;
            } else if (ph < 6.5) {
                scores['Aguacate'] += 4;
                scores['Cacao'] += 3;
                scores['Café'] += 1;
            } else if (ph < 7.5) {
                scores['Pastos'] += 4;
                scores['Aguacate'] += 2;
                scores['Cacao'] += 1;
            } else {
                scores['Caña panelera'] += 4;
                scores['Pastos'] += 2;
            }
            
            // Evaluación por materia orgánica
            if (mo >= 4.0) {
                scores['Aguacate'] += 3;
                scores['Cacao'] += 3;
                scores['Café'] += 1;
            } else if (mo >= 2.0) {
                scores['Cacao'] += 2;
                scores['Pastos'] += 2;
                scores['Café'] += 1;
            } else {
                scores['Caña panelera'] += 2;
                scores['Pastos'] += 1;
            }
            
            // Evaluación por fósforo
            if (fosforo >= 30) {
                scores['Cacao'] += 2;
                scores['Aguacate'] += 2;
            } else if (fosforo >= 15) {
                scores['Aguacate'] += 1;
                scores['Café'] += 1;
                scores['Cacao'] += 1;
            }
            
            // Evaluación por topografía
            if (topografia === "Pendiente") {
                scores['Café'] += 3;
                scores['Cacao'] += 1;
            } else if (topografia === "Plano") {
                scores['Caña panelera'] += 3;
                scores['Pastos'] += 2;
            } else {
                scores['Aguacate'] += 2;
                scores['Cacao'] += 1;
            }
            
            // Evaluación por drenaje
            if (drenaje === "Bueno" || drenaje === "Excelente") {
                scores['Aguacate'] += 2;
                scores['Cacao'] += 1;
            } else if (drenaje === "Regular") {
                scores['Pastos'] += 2;
                scores['Café'] += 1;
            }
            
            // Calcular resultado
            const maxScore = Math.max(...Object.values(scores));
            const winner = Object.keys(scores).find(key => scores[key] === maxScore);
            const confidence = Math.min(maxScore / 15.0, 0.95);
            
            // Crear ranking
            const ranking = Object.entries(scores)
                .sort((a, b) => b[1] - a[1])
                .map(([crop, score]) => [crop, score / 15.0]);
            
            return {
                cultivo: winner,
                confianza: confidence,
                ranking: ranking
            };
        }
        
        // Manejar envío del formulario
        document.getElementById('soilForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Obtener valores
            const ph = parseFloat(document.getElementById('ph').value);
            const mo = parseFloat(document.getElementById('mo').value);
            const fosforo = parseFloat(document.getElementById('fosforo').value);
            const potasio = parseFloat(document.getElementById('potasio').value);
            const calcio = parseFloat(document.getElementById('calcio').value);
            const magnesio = parseFloat(document.getElementById('magnesio').value);
            const topografia = document.getElementById('topografia').value;
            const drenaje = document.getElementById('drenaje').value;
            
            // Hacer predicción
            const result = predictCrop(ph, mo, fosforo, potasio, calcio, magnesio, topografia, drenaje);
            
            // Mostrar resultado
            displayResult(result);
        });
        
        function displayResult(result) {
            // Mostrar cultivo ganador
            document.getElementById('cropName').textContent = `🌱 ${result.cultivo}`;
            document.getElementById('confidence').textContent = `Confianza: ${(result.confianza * 100).toFixed(1)}%`;
            
            // Mostrar ranking
            const rankingList = document.getElementById('rankingList');
            rankingList.innerHTML = '';
            
            result.ranking.forEach((item, index) => {
                const [crop, score] = item;
                const emoji = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '🏅';
                const div = document.createElement('div');
                div.className = `crop-item ${index === 0 ? 'winner' : ''}`;
                div.innerHTML = `
                    <span>${emoji} ${index + 1}. ${crop}</span>
                    <span>${(score * 100).toFixed(1)}%</span>
                `;
                rankingList.appendChild(div);
            });
            
            // Interpretación
            let interpretation;
            if (result.confianza >= 0.7) {
                interpretation = `Excelente recomendación. Tu suelo presenta condiciones ideales para ${result.cultivo}.`;
            } else if (result.confianza >= 0.5) {
                interpretation = `Buena recomendación. ${result.cultivo} debería prosperar en tu suelo.`;
            } else if (result.confianza >= 0.3) {
                interpretation = `Recomendación moderada. ${result.cultivo} podría funcionar, considera también las alternativas.`;
            } else {
                interpretation = `Recomendación incierta. Te sugerimos consultar con un agrónomo local.`;
            }
            
            document.getElementById('interpretationText').textContent = interpretation;
            
            // Mostrar resultado con animación
            document.getElementById('result').style.display = 'block';
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
        }
        
        // Función para cargar ejemplos
        function loadExample(ph, mo, fosforo, potasio, calcio, magnesio, topografia, drenaje) {
            document.getElementById('ph').value = ph;
            document.getElementById('mo').value = mo;
            document.getElementById('fosforo').value = fosforo;
            document.getElementById('potasio').value = potasio;
            document.getElementById('calcio').value = calcio;
            document.getElementById('magnesio').value = magnesio;
            document.getElementById('topografia').value = topografia;
            document.getElementById('drenaje').value = drenaje;
        }
    </script>
</body>
</html>
"""

print("🚀 Creando interfaz HTML...")
print("✨ ¡Interfaz completamente funcional!")
print("📱 No requiere conexión externa - Todo funciona en el navegador")

# Mostrar la interfaz
display(HTML(html_interface))

print("\n🎉 ¡INTERFAZ CREADA EXITOSAMENTE!")
print("📋 Características:")
print("   ✅ Formulario completo con todos los campos")
print("   ✅ Botones de ejemplo para probar rápido")
print("   ✅ Sistema de predicción integrado")
print("   ✅ Resultados con ranking y interpretación")
print("   ✅ Diseño profesional y responsive")
print("   ✅ Funciona 100% sin dependencias externas")
print("\n🔥 ¡Tu recomendador de cultivos está listo para usar!")
