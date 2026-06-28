import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. DICCIONARIOS DE TRADUCCIÓN (UI -> Modelo)
# ==========================================
dict_especie = {"Perro": "Dog", "Gato": "Cat", "Pájaro": "Bird", "Otro": "Other"}

dict_sexo = {
    "Macho Intacto (No castrado)": "Intact Male", 
    "Hembra Intacta (No esterilizada)": "Intact Female", 
    "Macho Castrado": "Neutered Male", 
    "Hembra Esterilizada": "Spayed Female", 
    "Desconocido": "Unknown"
}

dict_ingreso = {
    "Callejero / Rescate (Stray)": "Stray", 
    "Entregado por Dueño (Owner Surrender)": "Owner Surrender", 
    "Asistencia Pública / Legal": "Public Assist"
}

dict_condicion = {
    "Normal / Sano": "Normal", 
    "Enfermo": "Sick", 
    "Herido": "Injured", 
    "Lactante": "Nursing", 
    "Anciano (Aged)": "Aged", 
    "Feral / Salvaje": "Feral", 
    "Otro": "Other"
}

# ==========================================
# 2. CONFIGURACIÓN VISUAL CORPORATIVA
# ==========================================
st.set_page_config(page_title="AAC | Triaje Predictivo", layout="wide", page_icon="🐾")

st.markdown("""
    <style>
    .stApp { background-color: #022135; color: #ffffff; }
    .alerta-alta { background-color: #e07a5f; padding: 25px; border-radius: 8px; color: white; text-align: center; border-left: 8px solid #a83232;}
    .alerta-media { background-color: #b5b462; padding: 25px; border-radius: 8px; color: #022135; text-align: center; border-left: 8px solid #8a883b;}
    .alerta-baja { background-color: #5a8b8b; padding: 25px; border-radius: 8px; color: white; text-align: center; border-left: 8px solid #2b4f4f;}
    hr {border-color: #113a5c;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. CARGA DEL MODELO (CACHÉ)
# ==========================================
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('motor_triaje_aac.pkl')
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

modelo_ia = cargar_modelo()

# ==========================================
# 4. INTERFAZ DE USUARIO (TRIAJE DÍA 0)
# ==========================================
st.title("🐾 Optimización de Triaje | Austin Animal Center")
st.markdown("### Motor de Predicción Logística y Asignación de Recursos")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Datos Demográficos")
    ui_especie = st.selectbox("Especie", list(dict_especie.keys()))
    age_in_years = st.number_input("Edad Estimada (Años)", min_value=0.0, max_value=25.0, value=5.0, step=0.5)
    ui_sexo = st.selectbox("Sexo y Estado Reproductivo", list(dict_sexo.keys()))
    breed = st.text_input("Raza (Texto Libre, ej. 'Pit Bull Mix')", value="Domestic Shorthair Mix")

with col2:
    st.subheader("🏥 Contexto de Admisión")
    ui_ingreso = st.selectbox("Vía de Ingreso", list(dict_ingreso.keys()), index=1)
    ui_condicion = st.selectbox("Condición Clínica Inicial", list(dict_condicion.keys()))

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. INGENIERÍA DE CARACTERÍSTICAS Y PREDICCIÓN
# ==========================================
if st.button("Ejecutar Análisis Predictivo", use_container_width=True, type="primary"):
    if modelo_ia is not None:
        
        # A. Traducción UI -> Backend
        animal_type = dict_especie[ui_especie]
        sex_upon_intake = dict_sexo[ui_sexo]
        intake_type = dict_ingreso[ui_ingreso]
        intake_condition = dict_condicion[ui_condicion]
        
        # B. Extracción de Heurísticas de Raza y Edad
        breed_lower = breed.lower()
        is_mix = 1 if 'mix' in breed_lower else 0
        is_pitbull = 1 if any(k in breed_lower for k in ['pit bull', 'staffordshire']) else 0
        is_chihuahua = 1 if 'chihuahua' in breed_lower else 0
        is_labrador = 1 if any(k in breed_lower for k in ['labrador', 'retriever']) else 0
        is_shepherd = 1 if 'shepherd' in breed_lower else 0
        is_husky = 1 if any(k in breed_lower for k in ['husky', 'malamute']) else 0
        
        is_puppy_kitten = 1 if age_in_years < 1.0 else 0
        is_senior = 1 if age_in_years >= 8.0 else 0

        # Lógica de tamaño
        large_keys = ['labrador', 'german shepherd', 'husky', 'pyrenees', 'rottweiler', 'mastiff', 'dane', 'bernard', 'retriever', 'hound']
        medium_keys = ['pit bull', 'cattle dog', 'border collie', 'boxer', 'australian shepherd', 'beagle', 'staffordshire', 'catahoula', 'pointer', 'bulldog']
        small_keys = ['chihuahua', 'dachshund', 'poodle', 'yorkshire', 'rat terrier', 'jack russell', 'schnauzer', 'shih tzu', 'pug', 'pomeranian', 'cairn']
        
        if any(k in breed_lower for k in large_keys): breed_size = 'Large'
        elif any(k in breed_lower for k in medium_keys): breed_size = 'Medium'
        elif any(k in breed_lower for k in small_keys): breed_size = 'Small'
        else: breed_size = 'Medium'

        # C. Construcción del Vector de Entrada Matemático
        input_data = pd.DataFrame({
            'animal_type': [animal_type],
            'intake_type': [intake_type],
            'intake_condition': [intake_condition],
            'sex_upon_intake': [sex_upon_intake],
            'breed_size': [breed_size],
            'age_in_years': [age_in_years],
            'is_puppy_kitten': [is_puppy_kitten],
            'is_senior': [is_senior],
            'is_mix': [is_mix],
            'is_pitbull': [is_pitbull],
            'is_chihuahua': [is_chihuahua],
            'is_labrador': [is_labrador],
            'is_shepherd': [is_shepherd],
            'is_husky': [is_husky]
        })

        # D. Inferencia (Logarítmica a Días Reales)
        pred_log = modelo_ia.predict(input_data)[0]
        dias_esperados = np.expm1(pred_log)
        
        # E. Dashboard de Resultados
        st.markdown("---")
        st.subheader("🎯 Diagnóstico Operativo")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Estadía Proyectada (LOS)", value=f"{dias_esperados:.1f} Días")
            st.caption("Margen de error (MedAE) estimado: ±3.9 días")
            
        with res_col2:
            if dias_esperados <= 14:
                st.markdown('<div class="alerta-baja"><h2>🟢 Riesgo Bajo (Flujo Estándar)</h2><p>El paciente tiene un perfil de alta rotación. Procesar mediante los canales regulares de adopción.</p></div>', unsafe_allow_html=True)
            elif 14 < dias_esperados <= 30:
                st.markdown('<div class="alerta-media"><h2>🟡 Riesgo Medio (Monitoreo)</h2><p>Probabilidad moderada de estancamiento. Programar evaluación fotográfica estándar e ingresar a lista de espera de hogares temporales (Fosters).</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alerta-alta"><h2>🔴 RIESGO ALTO (Alerta Logística)</h2><p><b>ACCIÓN INMEDIATA REQUERIDA:</b> Detonar protocolo de marketing agresivo el Día 0. Asignar sesión fotográfica prioritaria y contactar a socios comunitarios.</p></div>', unsafe_allow_html=True)