import os
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. Configuración inicial (igual que antes)
load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Tamaño estándar de un vector de OpenAI (1536 dimensiones)
VECTOR_SIZE = 1536

def generate_fake_embedding(center_val=0.0):
    """
    Crea un vector falso.
    Truco: Si cambiamos el 'center_val', el vector se mueve a otro lugar del mapa matemático.
    """
    # Generamos ruido aleatorio centrado en un valor específico
    # loc=center_val define el "centro" del vector.
    vector = np.random.normal(loc=center_val, scale=0.1, size=VECTOR_SIZE)
    return vector.tolist()

def seed_database():
    print("🌱 Iniciando siembra de datos falsos...")

    # --- LOTE 1: DATOS HISTÓRICOS (NORMALES) ---
    # Simulamos datos de hace 10 días
    old_date = datetime.now() - timedelta(days=10)
    print(f"Generando 50 registros 'normales' con fecha: {old_date}...")
    
    old_data = []
    for _ in range(50):
        old_data.append({
            "content": "Pregunta normal sobre el negocio",
            "type": "query",
            "timestamp": old_date.isoformat(),
            # Usamos 0.0 como centro para los datos "normales"
            "embedding": generate_fake_embedding(center_val=0.0) 
        })
    
    # Insertamos en Supabase en una sola llamada
    supabase.table('embeddings_log').insert(old_data).execute()
    print("✅ Datos históricos insertados.")

    # --- LOTE 2: DATOS RECIENTES (DRIFT / DESVIADOS) ---
    # Simulamos datos de HOY
    current_date = datetime.now()
    print(f"Generando 20 registros 'raros' (drift) con fecha: {current_date}...")

    new_data = []
    for _ in range(20):
        new_data.append({
            "content": "Pregunta extraña sobre cocina o videojuegos",
            "type": "query",
            "timestamp": current_date.isoformat(),
            # ¡AQUÍ ESTÁ EL TRUCO! Usamos 0.5 como centro. 
            # Matemáticamente, esto está "lejos" de 0.0
            "embedding": generate_fake_embedding(center_val=0.5)
        })

    supabase.table('embeddings_log').insert(new_data).execute()
    print("✅ Datos recientes (Drift) insertados.")
    print("🏁 ¡Siembra terminada! Tu base de datos ya tiene historia.")

if __name__ == "__main__":
    seed_database()