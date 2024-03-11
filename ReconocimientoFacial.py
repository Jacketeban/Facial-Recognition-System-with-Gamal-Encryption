import cv2
import os
import random

p = 2083

#Codigo elgamal.py

def generar_claves(p):
    # Se elige aleatoriamente un generador g y un exponente secreto a
    g = random.randint(2, p - 1)
    a = random.randint(2, p - 2)
    
    # Se calcula la clave pública h
    h = pow(g, a, p)
    
    # Se devuelven las claves
    return p, g, h, a

def cifrar(mensaje, clave_publica):

    p, g, h, _ = clave_publica #Se obtienen los parámetros de la clave pública en p, g, h y un valor no utilizado
    k = random.randint(2, p - 2) 
    c1 = pow(g, k, p) # Calculamos c1 como g elevado a la potencia k módulo p
    s = pow(h, k, p)  # Calculamos s como h elevado a la potencia k módulo p
    c2 = [(ord(c) * s) % p for c in mensaje]  # Ciframos cada carácter del mensaje multiplicándolo por s y tomando el módulo p ((ord(c)es el valor numérico del carácter en el mensaje original)
    
    return c1, c2 # Devolvemos el par de valores cifrados: c1 y c2

def descifrar(cifrado, clave_privada):

    p, _, _, a = clave_privada # Se obtiene la clave privada en p, 2 valores vacios y a
    c1, c2 = cifrado  # Obtenemos el cifrado en c1 y c2
    s = pow(c1, a, p) # Calculamos s como c1 elevado a la potencia a módulo p
    s_inv = pow(s, -1, p) # Calculamos el inverso multiplicativo de s módulo p
    mensaje = ''.join([chr((c * s_inv) % p) for c in c2]) # Desciframos cada carácter del cifrado multiplicándolo por s_inv y tomando el módulo p
    
    return mensaje # Devolvemos el mensaje descifrado

#Reconocimiento Facial

# Ruta a la carpeta de imágenes
dataPath = 'C:/Users/jacke/OneDrive/Escritorio/Reconocimiento Facial/Data'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# Cargar el modelo de reconocimiento facial
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leer el modelo entrenado
face_recognizer.read('modeloLBPHFace.xml')

# Capturar video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('Video.mp4')

# Cargar el clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bucle principal
while True:
    # Leer un fotograma
    ret, frame = cap.read()
    if ret == False:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    # Detectar rostros
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    # Variable para almacenar si se ha encontrado una persona autorizada
    persona_autorizada = False

    # Procesar cada rostro detectado
    for (x, y, w, h) in faces:
        # Extraer el rostro
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Predecir la persona
        result = face_recognizer.predict(rostro)
        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        # Verificar si la persona está autorizada
        if result[1] < 70:
            nombre_persona = imagePaths[result[0]]
            print("Persona reconocida:", nombre_persona)

            def verificar_autorizacion(nombre_persona):
                personas_autorizadas = ["Jhon"]
                return nombre_persona in personas_autorizadas

            persona_autorizada = verificar_autorizacion(nombre_persona)

    # Descifrar el mensaje solo si se ha encontrado una persona autorizada
    if persona_autorizada:
        try:
            # Cifrar un mensaje de prueba
            mensaje_original = "Hola, como estas esto es una contraseña 123@JHON!$$$$$$$$$$"
            clave_publica = generar_claves(p)
            cifrado = cifrar(mensaje_original, clave_publica)

            # Descifrar el mensaje
            clave_privada = generar_claves(p)
            mensaje_descifrado = descifrar(cifrado, clave_publica)
            print("Mensaje descifrado:", mensaje_descifrado)

        except Exception as e:
            print("Error al descifrar:", e)

    else:
        # Mostrar mensaje si no se ha encontrado una persona autorizada
        cv2.putText(frame, 'No se ha encontrado una persona autorizada', (50, 50), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

    # Mostrar el fotograma
    cv2.imshow('frame', frame)

    # Salir con la tecla 'Esc'
    k = cv2.waitKey(1)
    if k == 27:
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
