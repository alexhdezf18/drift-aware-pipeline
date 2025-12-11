import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime

# Simulamos que tenemos etiquetas (labels) para entrenar
# En la vida real, estas vendrían de tus usuarios marcando "like/dislike"
def train_new_model(embeddings_data):
    """
    Recibe datos (embeddings), entrena un modelo real y lo guarda en disco.
    """
    print("   ↳ 🧠 Iniciando entrenamiento de Random Forest...")
    
    # 1. Preparar los datos
    # Convertimos la lista de embeddings en un DataFrame de Pandas (Formato tabla)
    X = pd.DataFrame(embeddings_data)
    
    # Generamos etiquetas ficticias para que el código funcione (0 = Malo, 1 = Bueno)
    # (En un proyecto real, esto vendría de la base de datos interaction_log)
    y = [1 if i % 2 == 0 else 0 for i in range(len(embeddings_data))]

    # 2. Dividir para entrenar y probar (80% entrenar, 20% examen)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 3. Entrenar el modelo (Aquí ocurre el aprendizaje máquina real)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # 4. Evaluar qué tan bueno es
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"   ↳ 🎯 Precisión del nuevo modelo: {accuracy:.2f}")

    # 5. Guardar el artefacto (El archivo del modelo)
    # Creamos una carpeta 'models' si no existe
    if not os.path.exists('models'):
        os.makedirs('models')
        
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/model_{version}.joblib"
    
    joblib.dump(clf, filename)
    print(f"   ↳ 💾 Modelo guardado físicamente en: {filename}")
    
    return filename, accuracy