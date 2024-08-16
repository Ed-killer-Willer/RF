
#Librerias o modulos de funciones (Biblioteca):
import cv2 #Librería que procesa imagenes y videos.
import os #Interacción con el SO, gestiona archivos y directorios.
import numpy as np #Realiza calculos numericos, operaciones matematicas, matrices, arrays, etc.
import customtkinter as ctk # Mejora el aspecto de tkinter
from tkinter import filedialog, messagebox, simpledialog #Permite crear interfaces con elementos (botones, cuadros de dialogo, texto, etc).

# Verifica si cv2.face está disponible
if not hasattr(cv2, 'face'): #hasattr funcion para verificar si el objeto tiene un atributo con un nombre especifico
    messagebox.showerror("Error", "El módulo cv2.face no está disponible. Asegúrate de tener instalada la biblioteca opencv-contrib-python.")
    exit()

# Variables globales
#None = no hay ruta clara, ruta por definir.

dataPath = None # Directorio principal
face_recognizer = cv2.face.LBPHFaceRecognizer_create() #cv2 = alias de opencv, face = caracteristicas para el RF y contiene el modelo de RF basado en histograma.
#marca rostros (rectangulo verde)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #especifica clasificador en cascada, detecta imagenes de rostro mediante modelo pre-entrenado.
personPath = None #Directorio para almacenar los rostros.
savePath = None #Ruta dibde se guardan los datos. (Caché)

#Cuadro de inf.
messagebox.showinfo("Recomendaciones", "El sujeto debe tener el rostro descubierto para lograr mayor precisión.\n\nSoftware hecho con amor <3")

#def = definir funciones, 

#Seleccionar directorios
#askdirectory Función predeterminada por py para indicar el directorio.
def select_directory():
    #global = variable definida fuera de la función.
    global dataPath
    dataPath = filedialog.askdirectory(title="Selecciona la ubicación del archivo principal")
    
    if dataPath:
        check_and_create_folders()
        messagebox.showinfo("Configuración", "Ubicación seleccionada y carpetas verificadas/creadas.")
    else:
        messagebox.showerror("Error", "No se seleccionó ninguna ubicación.")

#def sí no hay directorios se crean
def check_and_create_folders():
    #main_folder=crear Main

    #path = Submodulo dentro de OS operación de la ruta de archivo
    #join = une o más rutas ajustado a SO
    #makedirs = Crear la Carpeta

    main_folder = os.path.join(dataPath, 'Main')
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    #data_folder=crear Data dentro de Main
    data_folder = os.path.join(main_folder, 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    #training_folder=crear training dentro de Main
    training_folder = os.path.join(main_folder, 'training')
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)

#Capturar rostros
#pack_forget = Oculta ventanas inecesarias, mejorando UI. Volviendo a frame1.
def captura_rostros():
    frame1.pack_forget()
    frame2.pack()

#Crear nueva persona
def create_new_folder():
    #global = variable definida fuera de la función.
    global savePath #Almacenar la ruta donde se guardarán la información (Caché).
    #simpledialog = crea un cuadro de dialogo (TK)
    #askstring = Cuadro de dialogo ventana emergente, solicita cadena de texto.
    personName = simpledialog.askstring("Nombre de Persona", "Introduce el nombre de la persona:")
    if not personName:
        messagebox.showerror("Error", "No se introdujo el nombre de la persona.")
        return
    #Ruta de almacenamiento.
    #SavePath = caché
    #datapath = directorio default
    #PersonName = nombre de usuario captura rostros
    savePath = os.path.join(dataPath, 'Main', 'data', personName)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        messagebox.showinfo("Información", f"Se ha creado la carpeta para {personName}.")
    else:
        messagebox.showinfo("Información", f"Se ha seleccionado la carpeta existente para {personName}.")
    
    take_photos()

#Carpeta existente
def select_existing_folder():
    #global = variable definida fuera de la función.
    #SavePath = caché
    global savePath

    #datapath = directorio default
    #default_path = por default será data donde será ubicado
    default_path = os.path.join(dataPath,'Main', 'data')
    
    #Elige las carpetas disponibles en data.
    savePath = filedialog.askdirectory(
        title="Selecciona la carpeta para guardar las capturas", 
        initialdir=default_path  # Ruta por defecto
    )
    
    #Si no se elige una carpeta error
    if not savePath:
        messagebox.showerror("Error", "No se seleccionó ninguna carpeta para guardar las capturas.")
        return
    #Función tomar fotos.
    take_photos()
    
#Función tomar fotos.
def take_photos():
    #pack_forget = Oculta ventanas inecesarias, mejorando UI. Volviendo a frame1.
    frame2.pack_forget()
    frame3.pack()
    
    # Mensaje informativo al comenzar la captura de rostros
    messagebox.showinfo("Captura de Rostros", "Se recomienda mantener el rostro descubierto para lograr una mayor precisión y girar el rostro en diferentes ángulos. \n\n Comenzará ahora.")
    
    ##Captura de rostros en camara 2.
    vc = cv2.VideoCapture(2)
    if not vc.isOpened():
        #Mensaje de error si esque esto falla.
        messagebox.showerror("Error", "No se pudo abrir la captura de video.")
        return

    cv2.namedWindow("Camara")
    
    #crear archivo con el sufijo Rostro formato jpg
    existing_files = [f for f in os.listdir(savePath) if f.startswith('Rostro ') and f.endswith('.jpg')]
    max_number = max([int(f.split(' ')[1].split('.')[0]) for f in existing_files], default=0)
    
    #sumando 1, y capturando 60 imagenes
    count = max_number + 1
    num_capturas = 0
    limite_capturas = 60

    #mientras sea cierto se captura imagenes, y si es falso muestra el error.
    while True:
        ret, frame = vc.read()
        if not ret:
            messagebox.showerror("Error", "Error al capturar el fotograma.")
            break
        
        #Gray = Escala de grises para detección f.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faceClassif.detectMultiScale = Detecta rostros en escala de grises. (Detecta, redimensiona y guarda la imagen)
        faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        #rostro_color = Recorta la imagen que contiene el rorstro usando la coordenadas del rectangulo.
        for (x, y, w, h) in faces:
            rostro_color = frame[y:y+h, x:x+w]
            #Redimensiona la imagen a 200x200 px como estandar
            #cv2.INTER_CUBIC = metodo de interpolación para redimenzionar la imagen
            rostro_color = cv2.resize(rostro_color, (200, 200), interpolation=cv2.INTER_CUBIC)
            try:
                #Guarda los rostros en la ruta guardada con el sufijo Rostros, extensión jpg, de forma secuencial
                cv2.imwrite(os.path.join(savePath, f'Rostro {count}.jpg'), rostro_color)
                #contador para el archivo de imagen.
                count += 1
                #total de capturas realizadas.
                num_capturas += 1
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar la imagen: {str(e)}")
                continue
            
            #Dibuja un rectangulo en el rostro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        #Muestra el fotograma de la camara, para interrrumpirse se oprime esc. 
        cv2.imshow("Camara", frame)
        k = cv2.waitKey(1)
        if k == 27 or num_capturas >= limite_capturas:
            break

    #Libera la camara        
    vc.release()
    #Destruye la ventana de CV.
    cv2.destroyAllWindows()
    #Muestra mensaje de Total de imagenes tomadas
    messagebox.showinfo("Capturas Completadas", f"Total de capturas tomadas: {num_capturas}")
    #Oculta frame incesesario.
    frame3.pack_forget()
    frame1.pack()

#Fun entrenamiento.
def entrenamiento():
    #Sí no hay ruta previa deberá seleccionar una.
    if dataPath is None:
        messagebox.showerror("Error", "Primero selecciona la ubicación del archivo principal.")
        return
    #Crea el modelo en la Raíz Main - training con el nombre de "modeloLBPHFace.xml"
    modelo_path = os.path.join(dataPath, 'Main', 'training', 'modeloLBPHFace.xml')
    
    #Modelo existente = Reentrenar o no hacerlo.
    if os.path.isfile(modelo_path):
        respuesta = messagebox.askyesno("Modelo Existente", "El modelo ya existe. ¿Deseas reentrenar el modelo?")
        if not respuesta:
            messagebox.showinfo("Entrenamiento", "No se realizará el entrenamiento.")
            return
    
    #Lista de personas dentro de data
    peopleList = os.listdir(os.path.join(dataPath, 'Main', 'data'))
    #Iniciación de listas
    labels = [] # Etiquetas correspondientes a cada rostro
    facesData = [] #Imagenes en formato escala de grises
    label = 0  #Contador que asigna un número único a cada persona.

    #itera sonre cada carpeta
    for nameDir in peopleList:
        personPath = os.path.join(dataPath, 'Main', 'data', nameDir)
        #itera sobre los archivos de imagenes dentro de la carpeta de esa persona
        for fileName in os.listdir(personPath):
            #añade etiqueta actual
            labels.append(label)
            #lee imagen del archivo actual en formato de escala de grises y la añade a facesData
            facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))
        label += 1

    try:
        #Entrena el modelo 
        #np.array(labels) convierte la lista de etiquetas a un array de NumPy
        face_recognizer.train(facesData, np.array(labels))
        #guarda el modelo entrenado en el archivo especificado
        face_recognizer.write(modelo_path)
        messagebox.showinfo("Entrenamiento", f"Modelo almacenado en: {modelo_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Error durante el entrenamiento: {str(e)}")

#Reconocimiento
def reconocimiento():
    #Sí no hay ruta inicial error
    if dataPath is None:
        messagebox.showerror("Error", "Primero selecciona la ubicación del archivo principal.")
        return
    #buscará el modelo "modeloLBPHFace.xml" en training
    modelo_path = os.path.join(dataPath, 'Main', 'training', 'modeloLBPHFace.xml')
    if not os.path.isfile(modelo_path):
        #sí no hay debe entrenarse primero
        messagebox.showerror("Error", "El archivo del modelo no existe. Entrena el modelo primero.")
        return
    #sí hay error mostrará un mensaje
    try:
        face_recognizer.read(modelo_path)
    except cv2.error as e:
        messagebox.showerror("Error", f"Error al leer el modelo: {str(e)}")
        return
    #la captura de video se abre con la camará n2.
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        #en caso de error mostrar mensaje
        messagebox.showerror("Error", "No se pudo abrir la captura de video.")
        return

    #nombre de la ventana
    cv2.namedWindow("Reconocimiento_facial")
    #accede a la lista de personas disòibles en data
    peopleList = os.listdir(os.path.join(dataPath, 'Main', 'data'))  # Lista de personas
    
    #todo ok = captura se leerá
    while True:
        ret, frame = cap.read()
        #fallo mensaje de error
        if not ret:
            messagebox.showerror("Error", "Error al capturar el fotograma.")
            break
        #conversión a escala de grises
        #convierte el fotograma capturado de la camara a escala de grises 
        #copia de una imagen en escala de grises que se usa para extraer y procesar rostros detectados.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        #clasifica rostros en escala de grises, ajusta escala para la detección, ajusta la escala para la detección.
        faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        #faceClassif.detectMultiScale() utiliza el clasificador de rostros (faceClassif) para detectar rostros en la imagen en escala de grises (gray).
        #scaleFactor=1.1 ajusta la escala para la detección.
        #minNeighbors=5 define el número mínimo de vecinos necesarios para considerar una región como rostro.
        #minSize=(30, 30) especifica el tamaño mínimo de un rostro a detectar.
        #flags=cv2.CASCADE_SCALE_IMAGE es una bandera para la detección.

        #recorta la región corrspondiente
        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            #Redimenciona la imagen a 200x200
            rostro = cv2.resize(rostro, (200, 200), interpolation=cv2.INTER_CUBIC)

            #Reconocer el rostro y reconocer la confianza
            try:
                #predicción con el modelo y devuelve un resultado
                result = face_recognizer.predict(rostro)
                #seguirdad del modelo de su predicción
                confidence = result[1]
                #etiqueta predicha para el rostro
                predicted_label = result[0]
                #error = mensaje
            except cv2.error as e:
                messagebox.showerror("Error", f"Error en la predicción: {str(e)}")
                print(f"Detalles del error: {e}")  # Para más detalles en la consola

                continue
            #añade texto de confianza y el nombre del sujeto
            cv2.putText(frame, 'Confianza: {}'.format(confidence), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if confidence < 41:  # Ajustar el umbral según sea necesario
                if predicted_label < len(peopleList):
                    name = peopleList[predicted_label]
                else:
                    name = "Desconocido"
            else:
                name = "Desconocido"
            #texto del frame
            cv2.putText(frame, name, (x, y-25), 1, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
            #rectangulo del frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        #Nombre del frame 
        cv2.imshow("Reconocimiento_facial", frame)
        #esperando a salir con esc
        if cv2.waitKey(1) == 27:
            break
    #libera espacio 
    cap.release()
    #destruye ventanas inecesarias
    cv2.destroyAllWindows()

import customtkinter as ctk

def volver_al_menu():
    frame2.pack_forget()
    frame1.pack()    

def create_gui():
    # Inicializar la ventana principal usando customtkinter (ctk)
    root = ctk.CTk()
    root.title("Facial Recognition")  # Establece el título de la ventana principal
    root.geometry("300x250")  # Define el tamaño de la ventana
    ctk.set_default_color_theme("green")  # Establece el tema de color predeterminado para la ventana

    # Declarar variables globales para los marcos (frames) que se utilizarán en la interfaz
    global frame1, frame2, frame3
    
    # Crear y empaquetar el primer marco (frame1)
    frame1 = ctk.CTkFrame(root)  # Crea un marco dentro de la ventana principal
    frame1.pack(fill="both", expand=True)  # Empaqueta el marco para que llene el espacio disponible y se expanda

    # Crear el segundo y tercer marco, pero aún no se empaquetan
    frame2 = ctk.CTkFrame(root)
    frame3 = ctk.CTkFrame(root)
    
    # Crear y agregar botones al primer marco (frame1)
    btnSelectDirectory = ctk.CTkButton(frame1, text="Ubicación Principal", command=select_directory)  # Botón para seleccionar la ubicación principal
    btnCapturaRostros = ctk.CTkButton(frame1, text="Capturar Rostros", command=captura_rostros)  # Botón para capturar rostros
    btnEntrenarModelo = ctk.CTkButton(frame1, text="Entrenar Modelo", command=entrenamiento)  # Botón para entrenar el modelo
    btnReconocimientoFacial = ctk.CTkButton(frame1, text="Reconocimiento Facial", command=reconocimiento)  # Botón para reconocimiento facial
    btnSalir = ctk.CTkButton(frame1, text="Salir", command=root.quit)  # Botón para salir de la aplicación
    
    # Empaquetar los botones en frame1 con un margen vertical de 10 píxeles
    btnSelectDirectory.pack(pady=10)
    btnCapturaRostros.pack(pady=10)
    btnEntrenarModelo.pack(pady=10)
    btnReconocimientoFacial.pack(pady=10)
    btnSalir.pack(pady=10)
    
    # Crear y agregar botones al segundo marco (frame2)
    btnCreateNewFolder = ctk.CTkButton(frame2, text="Crear Nueva Carpeta", command=create_new_folder)  # Botón para crear una nueva carpeta
    btnSelectExistingFolder = ctk.CTkButton(frame2, text="Seleccionar Carpeta Existente", command=select_existing_folder)  # Botón para seleccionar una carpeta existente
    btnbackMenu = ctk.CTkButton(frame2, text="Volver", command=volver_al_menu)  # Botón para volver al menú principal
    
    # Empaquetar los botones en frame2 con un margen vertical de 10 píxeles
    btnCreateNewFolder.pack(pady=10)
    btnSelectExistingFolder.pack(pady=10)
    btnbackMenu.pack(pady=10)

    # Ejecutar el bucle principal de la interfaz gráfica
    root.mainloop()  # Mantiene la ventana principal abierta y en espera de eventos del usuario

# Llamar a create_gui() solo si este archivo es el principal ejecutado
if __name__ == "__main__":
    create_gui()
