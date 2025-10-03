📄 Comparador de Documentos Word

Este proyecto permite comparar varios documentos Word (.docx) para detectar similitudes entre ellos.
Se utiliza tanto la coincidencia literal como métodos estadísticos, probabilísticos y semánticos, lo que lo convierte en una herramienta útil para:

🔎 Detección de plagio o paráfrasis

📑 Control de versiones de informes

🧪 Investigación académica y empresarial

🛡 Auditoría de documentación en consultoría

Incluye una interfaz gráfica (GUI) para facilitar la selección de carpetas y la ejecución del análisis, así como un script CLI para ejecución directa en terminal.

⚙️ Algoritmos implementados

El comparador aplica cinco enfoques distintos:

Literal – Distancia de Levenshtein (RapidFuzz)

Compara cadenas carácter por carácter.

Útil para detectar copias exactas o mínimas ediciones.

Conjuntos – Jaccard

Compara conjuntos de palabras compartidas entre documentos.

Detecta si dos textos usan un vocabulario similar.

Vectorial – TF-IDF + Coseno (Scikit-Learn)

Representa los documentos como vectores según la frecuencia de las palabras.

Permite medir similitudes estadísticas incluso con redacción diferente.

Probabilístico – LDA (Latent Dirichlet Allocation)

Descubre temas ocultos en los documentos.

Cada documento se representa como una mezcla de temas.

Útil para comparar documentos a nivel conceptual, aunque no compartan palabras.
⚠ Nota: es no determinista, los resultados pueden variar si no se fija random_state.

Semántico – SBERT (Sentence-BERT)

Convierte los documentos en embeddings (vectores de significado) mediante modelos de deep learning.

Detecta paráfrasis y similitud semántica aunque las frases estén redactadas de forma distinta.

Modelos incluidos:

paraphrase-MiniLM-L6-v2 (rápido, por defecto)

all-MiniLM-L12-v2

all-mpnet-base-v2 (más preciso, más pesado)

🖥️ Interfaz gráfica (GUI)

La GUI permite:

Seleccionar la carpeta de documentos .docx.

Incluir/excluir subcarpetas.

Elegir el nombre del Excel de salida.

Seleccionar el modelo SBERT.

Definir número de temas LDA (auto o manual).

Ajustar cuántos pares mostrar en los rankings.

Ver en la propia ventana los mensajes del proceso en tiempo real.

Exportar resultados a Excel.

Cada control tiene tooltip de ayuda al pasar el ratón.

📊 Resultados

El programa genera un archivo Excel con:

Matriz de similitud de cada método (1_Literal, 2_Conjuntos, 3_Vectorial, 4_Probabilistico, 5_Semantico).

Hojas con los TOP K pares de documentos más similares por método (TOP_*).

Los valores de similitud se expresan en porcentaje (0–100%).

🚀 Instalación

Clonar el repositorio o copiar los scripts.

Crear un entorno virtual (opcional pero recomendado):

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows


Instalar dependencias:

pip install -r requirements.txt

Dependencias principales

python-docx → lectura de Word

unidecode, nltk → normalización de texto

rapidfuzz → distancia de Levenshtein

scikit-learn → TF-IDF, LDA

pandas, openpyxl → exportar resultados a Excel

sentence-transformers → embeddings SBERT

gensim → soporte extra para modelos de tópicos

🏃 Uso por consola

Ejecutar el comparador directamente:

python comparadorv2.py "C:\ruta\a\carpeta\docs" --recursivo --salida resultados.xlsx --modelo paraphrase-MiniLM-L6-v2 --lda_topics auto --topk 50


Parámetros:

--recursivo → incluir subcarpetas

--salida → nombre del Excel

--modelo → modelo SBERT (MiniLM, mpnet, etc.)

--lda_topics → nº de temas (auto o entero)

--topk → nº de pares más similares por ranking

🖥️ Uso con interfaz gráfica
python gui_comparador.py


Desde la GUI puedes configurar todas las opciones y ejecutar el análisis con un clic.

📦 Generar ejecutable

Se recomienda usar PyInstaller para distribuir la aplicación sin necesidad de instalar Python:

Empaquetar el motor CLI:

pyinstaller --console --name comparador_cli --collect-all sentence_transformers comparadorv2.py


Empaquetar la GUI:

pyinstaller --windowed --name ComparadorGUI --add-data "%APPDATA%\nltk_data;nltk_data" gui_comparador.py


Entregar ambos EXE en una misma carpeta.

⚠️ Limitaciones conocidas

El método LDA es aproximado y no determinista si no se fija semilla (random_state).

El modelo SBERT puede tardar la primera vez porque descarga embeddings desde HuggingFace.

Archivos .docx muy grandes pueden ralentizar el cálculo.

📚 Referencias

Sentence-Transformers

Scikit-Learn – Text Feature Extraction

Gensim – Topic Modeling with LDA

NLTK Stopwords