import os
import json
import time
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_distances
from model_trainer import train_new_model

# --- 1. CONFIGURACIÓN ---
load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# --- 2. FUNCIONES DE BÚSQUEDA (Sin cambios) ---
def fetch_recent_embeddings(days_ago=7):
    start_time = datetime.now() - timedelta(days=days_ago)
    try:
        response = supabase.table('embeddings_log') \
            .select('embedding') \
            .gt('timestamp', start_time.isoformat()) \
            .execute()
        
        data = []
        for record in response.data:
            vec = record['embedding']
            if isinstance(vec, str): vec = json.loads(vec)
            data.append(vec)
        return data
    except Exception as e:
        print("Error fetching recent:", e)
        return []

def fetch_reference_embeddings(days_ago=7):
    end_time = datetime.now() - timedelta(days=days_ago)
    try:
        response = (
            supabase.table('embeddings_log')
            .select('embedding')
            .lt('timestamp', end_time.isoformat())
            .limit(100) 
            .execute()
        )
        data = []
        for record in response.data:
            vec = record['embedding']
            if isinstance(vec, str): vec = json.loads(vec)
            data.append(vec)
        return data
    except Exception as e:
        print("Error fetching reference:", e)
        return []

# --- 3. LÓGICA DE DETECCIÓN (Sin cambios) ---
def detect_drift(reference_data, current_data, threshold=0.1):
    if not reference_data or not current_data:
        return False, 0.0
    ref_matrix = np.array(reference_data)
    curr_matrix = np.array(current_data)
    ref_centroid = np.mean(ref_matrix, axis=0)
    curr_centroid = np.mean(curr_matrix, axis=0)
    distance = cosine_distances([ref_centroid], [curr_centroid])[0][0]
    return distance > threshold, distance

# --- 4. AUTOMATIZACIÓN (NUEVO: Acciones y Logging) ---
def log_event(event_type, severity, details, action):
    """Guarda el incidente en la tabla drift_events para el tablero de BI."""
    print(f"📝 Registrando evento: {event_type}...")
    try:
        data = {
            "event_type": event_type,
            "severity_score": float(severity), # Convertir a float simple para JSON
            "details": details,
            "action_taken": action,
            "timestamp": datetime.now().isoformat()
        }
        supabase.table('drift_events').insert(data).execute()
        print("✅ Evento guardado en base de datos.")
    except Exception as e:
        print(f"❌ Error guardando evento: {e}")


# --- 5. EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print("🔎 Monitor ejecutándose...")
    
    # Paso A: Obtener Datos
    recent = fetch_recent_embeddings(7)
    reference = fetch_reference_embeddings(7)
    
    # Paso B: Detectar
    is_drift, score = detect_drift(reference, recent)
    print(f"📊 Score de Drift: {score:.4f}")

    # Paso C: Decidir y Actuar
    if is_drift:
        print("🚨 DRIFT DETECTADO. Iniciando MLOps Pipeline...")
        
        # 1. Juntamos TODOS los datos (Viejos + Nuevos) para re-entrenar
        # Esto es lo que se hace en la realidad: aprender de todo lo disponible
        full_training_data = reference + recent
        
        if len(full_training_data) > 0:
            # 2. Llamamos al script de entrenamiento REAL
            model_path, new_accuracy = train_new_model(full_training_data)
            
            # 3. Registramos el evento con datos reales
            log_event(
                event_type="model_retraining", 
                severity=score, 
                details=f"Drift detectado. Modelo reentrenado con {len(full_training_data)} registros. Nueva precisión: {new_accuracy}",
                action=f"Created artifact: {model_path}"
            )
        else:
            print("⚠️ Error: No hay suficientes datos para reentrenar.")

    else:
        print("✅ Sistema estable. No se requieren acciones.")