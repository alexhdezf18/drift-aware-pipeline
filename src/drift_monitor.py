import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timedelta

# 1. Cargar los secretos de la bóveda (.env)
load_dotenv()

# 2. Obtener las credenciales
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

# 3. Conectar con Supabase
# Esta variable 'supabase' es nuestro "bibliotecario" que puede leer y escribir datos.
supabase: Client = create_client(url, key)

print("--- Iniciando prueba de conexión ---")


def fetch_recent_embeddings(days_ago=7):
    """
    Obtiene los embeddings registrados en los últimos X días.
    """
    # 1. Calculamos la fecha de corte
    start_time = datetime.now() - timedelta(days=days_ago)
    
    print(f"--- Buscando datos desde: {start_time} ---")

    # 2. Hacemos la consulta a Supabase
    # .select("embedding") -> Solo queremos la columna de vectores, no todo el texto.
    # .gt("timestamp", ...) -> "Greater Than" (Mayor que). Significa "danos fechas DESPUÉS de start_time"
    try:
        response = supabase.table('embeddings_log') \
            .select('embedding') \
            .gt('timestamp', start_time.isoformat()) \
            .execute()
            
        return response.data
    except Exception as e:
        print("Error al buscar datos:", e)
        return []


def fetch_reference_embeddings(days_ago=7):
    """
    Obtiene los embeddings 'viejos' (históricos) para usarlos como referencia.
    Busca todo lo que sea ANTERIOR a la fecha de corte.
    """
    end_time = datetime.now() - timedelta(days=days_ago)
    
    print(f"--- Buscando referencia (datos anteriores a {end_time}) ---")

    try:
        response = supabase.table('embeddings_log') \
            .select('embedding') \
            .lt('timestamp', end_time.isoformat()) \
            .execute()
            
        return response.data
    except Exception as e:
        print("Error al buscar referencia:", e)
        return []

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # Probamos la función
    recent_data = fetch_recent_embeddings(7)
    
    # Como la base de datos está vacía, esto debería imprimir una lista vacía []
    print(f"Registros encontrados: {len(recent_data)}")
    print(f"Data: {recent_data}")