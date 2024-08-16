# Facial Recognition con LBPH_Face

Este proyecto es una aplicación de reconocimiento facial utilizando Python, OpenCV, y una interfaz gráfica basada en CustomTkinter.

## Requisitos

Asegúrate de tener instaladas las siguientes dependencias antes de ejecutar el proyecto:

1. OpenCV
OpenCV es una biblioteca esencial para el procesamiento de imágenes y video.
Librería principal:
pip nstall opencv-python`
Librería adicional:
pip opencv-contrib-python (necesaria para el módulo `cv2.face`)
Instalación:
pip install opencv-python opencv-contrib-python

2. NumPy
NumPy es utilizado para manejar arreglos y realizar operaciones matemáticas esenciales en el procesamiento de imágenes y entrenamiento de modelos.
Instalación:
pip install numpy

3. CustomTkinter
CustomTkinter se utiliza para crear una interfaz gráfica moderna y personalizable.
Instalación:
pip install customtkinter

4. Tkinter
Tkinter es la biblioteca estándar de Python para crear interfaces gráficas.
Tkinter viene incluido con Python, por lo que no es necesario instalarlo manualmente.

5. OS
OS es un módulo estándar de Python que proporciona funciones para interactuar con el sistema operativo.

OS también viene incluido con Python.
Instrucciones de Uso
Configuración del entorno:

Asegúrate de tener todas las dependencias instaladas.
Puedes utilizar un entorno virtual para mantener las dependencias aisladas.
Ejecución del programa:

Ubicación Principal: Selecciona la carpeta donde se guardarán los datos y modelos.
Capturar Rostros: Captura imágenes de rostros y guárdalas en una carpeta específica.
Entrenar Modelo: Entrena el modelo de reconocimiento facial utilizando las imágenes capturadas.
Reconocimiento Facial: Utiliza la cámara para reconocer rostros basados en el modelo entrenado.

Notas Adicionales
Modelo de Reconocimiento Facial: Este proyecto utiliza el algoritmo LBPH (Local Binary Patterns Histograms) para el reconocimiento facial.
Umbral de Confianza: Ajusta el umbral de confianza en el código según tus necesidades para mejorar la precisión del reconocimiento.
Problemas Comunes
Error al importar cv2.face: Asegúrate de tener instalada la biblioteca opencv-contrib-python ya que contiene módulos adicionales como cv2.face.
No se puede abrir la cámara: Verifica que la cámara esté conectada y no esté siendo utilizada por otra aplicación.
