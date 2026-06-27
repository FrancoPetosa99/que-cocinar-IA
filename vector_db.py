# vector_db.py
import chromadb
import os

# Definimos la ruta local para la persistencia de la base de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DATA_PATH = os.path.join(BASE_DIR, ".chroma_data")


# Inicializamos la conexión a ChromaDB en modo persistente
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

def get_chroma_client():
    """
    Retorna la instancia global del cliente de ChromaDB.
    """
    return chroma_client

def check_connection():
    """
    Verifica que la conexión a la base de datos esté activa y lista.
    """
    try:
        heartbeat = chroma_client.heartbeat()
        print(f"✅ Conexión a ChromaDB exitosa. Heartbeat: {heartbeat}")
        print(f"📂 Los datos están persistiendo correctamente en: {CHROMA_DATA_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error al conectar con ChromaDB: {e}")
        return False

if __name__ == "__main__":
    print("Iniciando prueba de conexión a la base de datos vectorial...")
    check_connection()