# Comparador de documentos Word (.docx)

Este proyecto contiene un script en Python que permite **comparar múltiples documentos Word** y calcular su **similitud** usando distintos enfoques, desde métodos clásicos hasta modelos semánticos modernos.  

El resultado se guarda en un archivo Excel (`resultados_similitud.xlsx`) con una hoja por método y un ranking de pares de documentos más similares.

---

## 🚀 Funcionalidad

El script lee todos los `.docx` en una carpeta y genera:

1. **Literal (Levenshtein):** compara los textos carácter a carácter (útil para copias exactas).  
2. **Conjuntos (Jaccard):** compara las palabras comunes entre documentos.  
3. **Vectorial (TF-IDF + coseno):** analiza la frecuencia relativa de las palabras.  
4. **Probabilístico (LDA):** representa cada documento como una mezcla de *temas* y compara las distribuciones.  
5. **Semántico (SBERT + coseno):** usa embeddings de `Sentence-BERT` para detectar similitud de significado, incluso con paráfrasis.  

El Excel generado incluye:  
- Una hoja con la **matriz de similitud** para cada método (valores en %).  
- Una hoja con el **TOP N pares de documentos más similares** por cada método.  
- Una hoja **INFO** con parámetros de ejecución y lista de documentos analizados.

---

## 📂 Estructura de salida

Ejemplo de hojas en el Excel:

- `1_Literal_Levenshtein`  
- `2_Conjuntos_Jaccard`  
- `3_Vectorial_TFIDF`  
- `4_Probabilistico_LDA`  
- `5_Semantico_SBERT`  
- `TOP_Literal`, `TOP_Conjuntos`, `TOP_Vectorial`, `TOP_Probabilistico`, `TOP_Semantico`  
- `INFO`

---

## 🛠️ Requisitos

- Python **3.9+** (compatible con 3.13)  
- Librerías necesarias:  

```bash
pip install python-docx unidecode pandas numpy scikit-learn rapidfuzz \
            sentence-transformers nltk openpyxl
