import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. DICCIONARIOS Y DATOS PRECARGADOS
# ==========================================
dict_especie = {"Perro": "Dog", "Gato": "Cat", "Pájaro": "Bird", "Otro": "Other"}
iconos_especie = {"Perro": "🐶", "Gato": "🐱", "Pájaro": "🐦", "Otro": "🐾"}

top_razas = {
    "Perro": ['Pit Bull Mix', 'Chihuahua Shorthair Mix', 'Labrador Retriever Mix', 'German Shepherd Mix', 'Australian Cattle Dog Mix', 'Dachshund Mix', 'Boxer Mix', 'Border Collie Mix', 'Miniature Poodle Mix', 'Otra (Especificar)'],
    "Gato": ['Domestic Shorthair Mix', 'Domestic Medium Hair Mix', 'Domestic Longhair Mix', 'Siamese Mix', 'Domestic Shorthair', 'American Shorthair Mix', 'Snowshoe Mix', 'Otra (Especificar)'],
    "Pájaro": ['Chicken Mix', 'Parakeet Mix', 'Duck Mix', 'Cockatiel Mix', 'Pigeon Mix', 'Otra (Especificar)'],
    "Otro": ['Otra (Especificar)']
}

dict_sexo = {
    "Macho Intacto (No castrado)": "Intact Male", 
    "Hembra Intacta (No esterilizada)": "Intact Female", 
    "Macho Castrado": "Neutered Male", 
    "Hembra Esterilizada": "Spayed Female", 
    "Desconocido": "Unknown"
}

dict_ingreso = {"Callejero / Rescate (Stray)": "Stray", "Entregado por Dueño (Owner Surrender)": "Owner Surrender", "Asistencia Pública / Legal": "Public Assist"}
iconos_ingreso = {"Callejero / Rescate (Stray)": "🛣️", "Entregado por Dueño (Owner Surrender)": "🏠", "Asistencia Pública / Legal": "⚖️"}

dict_condicion = {"Normal / Sano": "Normal", "Enfermo": "Sick", "Herido": "Injured", "Lactante": "Nursing", "Anciano (Aged)": "Aged", "Feral": "Feral"}
iconos_condicion = {"Normal / Sano": "✅", "Enfermo": "🤒", "Herido": "🩹", "Lactante": "🍼", "Anciano (Aged)": "👴", "Feral": "🦁"}

# ==========================================
# 2. FUNCIONES REACTIVAS
# ==========================================
def calcular_heuristicas(breed_name, age_years):
    b = breed_name.lower()
    is_puppy = 1 if age_years < 1.0 else 0
    is_senior = 1 if age_years >= 8.0 else 0
    is_mix = 1 if 'mix' in b else 0
    is_pitbull = 1 if any(k in b for k in ['pit bull', 'staffordshire']) else 0
    is_chihuahua = 1 if 'chihuahua' in b else 0
    is_labrador = 1 if any(k in b for k in ['labrador', 'retriever']) else 0
    is_shepherd = 1 if 'shepherd' in b else 0
    is_husky = 1 if any(k in b for k in ['husky', 'malamute']) else 0

    large_keys = ['labrador', 'german shepherd', 'husky', 'pyrenees', 'rottweiler', 'mastiff', 'dane', 'bernard', 'retriever', 'hound']
    medium_keys = ['pit bull', 'cattle dog', 'border collie', 'boxer', 'australian shepherd', 'beagle', 'staffordshire', 'catahoula', 'pointer', 'bulldog']
    small_keys = ['chihuahua', 'dachshund', 'poodle', 'yorkshire', 'rat terrier', 'jack russell', 'schnauzer', 'shih tzu', 'pug', 'pomeranian', 'cairn']
    
    if any(k in b for k in large_keys): size = 'Large'
    elif any(k in b for k in medium_keys): size = 'Medium'
    elif any(k in b for k in small_keys): size = 'Small'
    else: size = 'Medium'
    
    return size, is_puppy, is_senior, is_mix, is_pitbull, is_chihuahua, is_labrador, is_shepherd, is_husky

# ==========================================
# 3. CONFIGURACIÓN Y ESTILOS
# ==========================================
st.set_page_config(page_title="AAC | Triaje Predictivo", layout="wide", page_icon="🐾")

st.markdown("""
    <style>
    .stApp { background-color: #022135; color: #ffffff; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    .size-container { display: flex; justify-content: space-around; background: #022135; padding: 15px; border-radius: 10px; margin-top: 5px; margin-bottom: 15px; border: 1px solid #113a5c;}
    .paw { display: flex; flex-direction: column; align-items: center; transition: all 0.3s ease; }
    .paw-active { color: #b5b462; transform: scale(1.15); opacity: 1; text-shadow: 0px 0px 10px rgba(181,180,98,0.5); }
    .paw-inactive { color: #ffffff; opacity: 0.2; transform: scale(0.9); filter: grayscale(100%); }
    .paw-icon { font-size: 2rem; }
    .paw-text { font-size: 0.75rem; font-weight: bold; margin-top: 5px; }

    .tree-container { display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 10px; background-color: #113a5c; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .tree-node { display: flex; flex-direction: column; align-items: center; width: 110px; text-align: center; transition: all 0.3s ease;}
    .tree-icon { font-size: 2.5rem; background-color: #022135; padding: 15px; border-radius: 50%; border: 2px solid #b5b462; width: 75px; height: 75px; display: flex; justify-content: center; align-items: center; margin-bottom: 10px; }
    .tree-label { font-size: 0.85rem; color: #ffffff; font-weight: bold; }
    .tree-arrow { font-size: 2rem; color: #b5b462; padding-bottom: 30px; }

    .alerta-container { margin-top: 20px; animation: fadeIn 0.5s ease-out; }
    .alerta-alta { background-color: #e07a5f; padding: 30px; border-radius: 12px; color: white; text-align: center; border-left: 10px solid #a83232; box-shadow: 0 0 15px rgba(224,122,95,0.4); }
    .alerta-media { background-color: #b5b462; padding: 30px; border-radius: 12px; color: #022135; text-align: center; border-left: 10px solid #8a883b; }
    .alerta-baja { background-color: #5a8b8b; padding: 30px; border-radius: 12px; color: white; text-align: center; border-left: 10px solid #2b4f4f; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    try: return joblib.load('motor_triaje_aac.pkl')
    except: return None

modelo_ia = cargar_modelo()

# ==========================================
# 4. UI REACTIVA: LAYOUT OPTIMIZADO
# ==========================================
st.title("Panel de Optimización de Triaje | AAC")
st.markdown("---")

col_form, col_espacio, col_resultados = st.columns([1.3, 0.1, 2])

with col_form:
    st.markdown("### 📋 Perfil de Ingreso")
    
    # Fila 1: Especie y Edad (Mitad y Mitad)
    f1_c1, f1_c2 = st.columns(2)
    with f1_c1:
        ui_especie = st.selectbox("Especie", list(dict_especie.keys()))
    with f1_c2:
        age_in_years = st.number_input("Edad (Años)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
    
    # Fila 2: Raza (Ancho completo para el input dinámico)
    raza_seleccionada = st.selectbox("Raza Principal", top_razas[ui_especie])
    if raza_seleccionada == 'Otra (Especificar)':
        raza_final = st.text_input("Escriba la raza:", value="Mix")
    else:
        raza_final = raza_seleccionada
        
    breed_size, is_puppy, is_senior, is_mix, is_pitbull, is_chihuahua, is_labrador, is_shepherd, is_husky = calcular_heuristicas(raza_final, age_in_years)
    
    # Fila 3: Indicador Morfológico
    st.markdown("<p style='font-size:0.85rem; margin-bottom: 2px; color: #b5b462; text-align: center;'>Estimación de Morfología</p>", unsafe_allow_html=True)
    c_small = "paw-active" if breed_size == 'Small' else "paw-inactive"
    c_medium = "paw-active" if breed_size == 'Medium' else "paw-inactive"
    c_large = "paw-active" if breed_size == 'Large' else "paw-inactive"
    
    st.markdown(f"""
    <div class='size-container'>
        <div class='paw {c_small}'><div class='paw-icon'>🐕</div><div class='paw-text'>PEQUEÑO</div></div>
        <div class='paw {c_medium}'><div class='paw-icon'>🦮</div><div class='paw-text'>MEDIANO</div></div>
        <div class='paw {c_large}'><div class='paw-icon'>🐕‍🦺</div><div class='paw-text'>GRANDE</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Fila 4: Condición y Vía de Ingreso (Mitad y Mitad)
    f4_c1, f4_c2 = st.columns(2)
    with f4_c1:
        ui_condicion = st.selectbox("Estado de Salud", list(dict_condicion.keys()))
    with f4_c2:
        ui_ingreso = st.selectbox("Vía de Ingreso", list(dict_ingreso.keys()), index=1)
    
    # Fila 5: Sexo (Ancho completo por la longitud de los textos)
    ui_sexo = st.selectbox("Sexo y Estado Reproductivo", list(dict_sexo.keys()))
    
    st.markdown("</div><br>", unsafe_allow_html=True)
    ejecutar_btn = st.button("🚀 Calcular Pronóstico de Estadía (LOS)", use_container_width=True, type="primary")

# ==========================================
# 5. RENDERIZADO VISUAL EN VIVO
# ==========================================
with col_resultados:
    st.markdown("### 🔍 Vista Previa del Triaje en Vivo")
    
    icono_edad = "🍼" if is_puppy else ("👴" if is_senior else "⏱️")
    etiqueta_edad = "Cachorro/Cría" if is_puppy else ("Senior" if is_senior else "Adulto")
    
    arbol_html = f"""
    <div class="tree-container">
        <div class="tree-node">
            <div class="tree-icon">{iconos_especie[ui_especie]}</div>
            <div class="tree-label">{ui_especie}</div>
        </div>
        <div class="tree-arrow">➔</div>
        <div class="tree-node">
            <div class="tree-icon">{icono_edad}</div>
            <div class="tree-label">{etiqueta_edad} <br>({age_in_years} años)</div>
        </div>
        <div class="tree-arrow">➔</div>
        <div class="tree-node">
            <div class="tree-icon">{iconos_ingreso[ui_ingreso]}</div>
            <div class="tree-label">Vía de Ingreso</div>
        </div>
        <div class="tree-arrow">➔</div>
        <div class="tree-node">
            <div class="tree-icon">{iconos_condicion[ui_condicion]}</div>
            <div class="tree-label">Estado de Salud</div>
        </div>
    </div>
    """
    st.markdown(arbol_html, unsafe_allow_html=True)
    
    # ==========================================
    # 6. INFERENCIA DEL MODELO
    # ==========================================
    if ejecutar_btn and modelo_ia is not None:
        
        input_data = pd.DataFrame({
            'animal_type': [dict_especie[ui_especie]], 'intake_type': [dict_ingreso[ui_ingreso]], 
            'intake_condition': [dict_condicion[ui_condicion]], 'sex_upon_intake': [dict_sexo[ui_sexo]], 
            'breed_size': [breed_size], 'age_in_years': [age_in_years],
            'is_puppy_kitten': [is_puppy], 'is_senior': [is_senior], 'is_mix': [is_mix],
            'is_pitbull': [is_pitbull], 'is_chihuahua': [is_chihuahua], 'is_labrador': [is_labrador],
            'is_shepherd': [is_shepherd], 'is_husky': [is_husky]
        })

        pred_log = modelo_ia.predict(input_data)[0]
        dias_esperados = np.expm1(pred_log)
        
        st.markdown("### 📊 Resultado Analítico de Operaciones")
        
        if dias_esperados <= 14:
            st.markdown(f"""
            <div class="alerta-container alerta-baja">
                <h1 style='font-size: 3.5rem; margin: 0;'>{dias_esperados:.1f} Días Proyectados</h1>
                <h2>🟢 Riesgo Bajo (Flujo Estándar)</h2>
                <p style='font-size: 1.1rem;'>El paciente tiene un perfil de alta rotación logística. Procesar mediante los canales regulares de adopción.</p>
            </div>
            """, unsafe_allow_html=True)
        elif 14 < dias_esperados <= 30:
            st.markdown(f"""
            <div class="alerta-container alerta-media">
                <h1 style='font-size: 3.5rem; margin: 0; color: #022135;'>{dias_esperados:.1f} Días Proyectados</h1>
                <h2>🟡 Riesgo Medio (Monitoreo Estricto)</h2>
                <p style='font-size: 1.1rem; color: #022135;'>Probabilidad moderada de estancamiento. Programar evaluación fotográfica estándar e ingresar a lista de espera de hogares temporales (Foster Network).</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alerta-container alerta-alta">
                <h1 style='font-size: 3.5rem; margin: 0;'>{dias_esperados:.1f} Días Proyectados</h1>
                <h2>🔴 RIESGO LOGÍSTICO ALTO</h2>
                <p style='font-size: 1.1rem;'><b>ACCIÓN INMEDIATA:</b> Detonar protocolo de marketing cruzado el Día 0. Asignar fotografía prioritaria y enviar alerta temprana a fundaciones asociadas de rescate.</p>
            </div>
            """, unsafe_allow_html=True)
            
    elif not ejecutar_btn:
        st.markdown("""
        <div style='text-align: center; margin-top: 30px; opacity: 0.6;'>
            <p><i>👆 Los indicadores cambian en tiempo real. Configura el perfil a la izquierda y calcula el pronóstico de impacto.</i></p>
        </div>
        """, unsafe_allow_html=True)