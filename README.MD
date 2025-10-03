Informe técnico

Script de comparación de similitud entre documentos Word (.docx)

1. Objetivo del script

El script permite comparar múltiples documentos Word y calcular un porcentaje de similitud entre ellos mediante dos enfoques complementarios:

Similitud TF-IDF (coseno): mide coincidencia literal de vocabulario y expresiones frecuentes.

Similitud semántica (embeddings SBERT): mide cercanía en significado, incluso si se usan palabras distintas (detección de paráfrasis).

De esta manera, el analista puede identificar documentos duplicados, parcialmente copiados o reformulados, útil para detección de plagio, control de calidad documental o auditoría de informes.

2. Funcionamiento del script
a) Carga y normalización

Se leen todos los archivos .docx en la carpeta indicada.

Se extrae el texto de párrafos y tablas.

El texto se normaliza:

Minúsculas.

Eliminación de acentos/diacríticos.

Compactación de espacios.

b) Similitud TF-IDF

Se aplica el modelo TF-IDF (Term Frequency – Inverse Document Frequency):

Cada documento se convierte en un vector numérico que refleja la importancia relativa de cada palabra o combinación de palabras (ngrams).

Se calcula la similitud del coseno entre los vectores de dos documentos.

Este método detecta coincidencias literales: cuanto mayor sea el vocabulario en común y su relevancia, mayor será la similitud.

c) Similitud semántica (SBERT)

Se utiliza Sentence-BERT, un modelo de lenguaje preentrenado que convierte cada documento en un embedding (vector de alta dimensión que representa el significado del texto).

Se calcula el coseno entre los embeddings de los documentos.

Este método captura la similitud semántica: dos frases que expresan lo mismo con palabras distintas producen vectores cercanos.

Ejemplo: “El coche azul estaba estacionado” ≈ “El automóvil de color azul se aparcó”.

d) Resultados presentados

Matriz TF-IDF: muestra porcentajes de similitud literal entre todos los pares de documentos.

Matriz semántica: muestra porcentajes de similitud de significado.

Top pares más similares: ranking con los pares de documentos que presentan mayor similitud semántica, para una revisión rápida.

3. Teoría de fondo

TF-IDF

Ventaja: sencillo, rápido y útil para textos que comparten vocabulario exacto.

Limitación: no detecta paráfrasis ni sinónimos, porque depende de palabras exactas.

Embeddings semánticos (SBERT)

Ventaja: entiende el significado del texto, permite detectar reformulaciones.

Limitación: requiere modelos más pesados y entrenamiento previo; el resultado depende de la calidad del modelo.

Similitud del coseno

Métrica común en ambos enfoques.

Mide el ángulo entre dos vectores:

1 (o 100 %) → vectores idénticos → documentos muy similares.

0 → vectores ortogonales → documentos totalmente diferentes.

4. Usos prácticos

Detección de plagio o paráfrasis en informes o trabajos académicos.

Control de versiones: verificar si un documento nuevo es solo una copia parcial de otro.

Análisis de redundancia documental en proyectos de consultoría.

Auditoría de cumplimiento: identificar si varias memorias siguen la misma estructura y redacción.

5. Conclusión

El script combina la potencia de un método clásico (TF-IDF) con técnicas modernas de inteligencia artificial (SBERT), proporcionando un análisis completo de similitud entre documentos.
Gracias a esta doble perspectiva, es posible detectar tanto copias literales como plagio más sofisticado mediante paráfrasis, ofreciendo al analista una herramienta práctica para comparar documentos en entornos profesionales.