import cv2 as cv
import os
def reconocimientofac():
    modelo = "nombre usuario"
    ##ruta1 es la ruta donde queremos almacenar la data o los rostros que vamos a recolectar
    ruta1 = 'data'
    rutacompleta = os.path.join(ruta1, modelo)
    if not os.path.exists(rutacompleta):
        os.makedirs(rutacompleta)

    camara = cv.VideoCapture(0)
    ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    id = 0
    while True:
        respuesta, captura = camara.read()
        if not respuesta:
            break

        grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
        idcaptura = captura.copy()

        caras = ruidos.detectMultiScale(grises, 1.2, 5)

        for (x, y, e1, e2) in caras:
            nuevo_ancho = int(e1 * 1.2)
            nuevo_alto = int(e2 * 1.2)
            
            nuevo_x = x - int((nuevo_ancho - e1) / 2)
            nuevo_y = y - int((nuevo_alto - e2) / 2)

            cv.rectangle(captura, (nuevo_x, nuevo_y), (nuevo_x + nuevo_ancho, nuevo_y + nuevo_alto), (0, 255, 0), 2)

            rostrocapturado = idcaptura[nuevo_y:nuevo_y + nuevo_alto, nuevo_x:nuevo_x + nuevo_ancho]

            rostrocapturado = cv.resize(rostrocapturado, (160, 160), interpolation=cv.INTER_CUBIC)

            cv.imwrite(os.path.join(rutacompleta, 'imagen_{}.jpg'.format(id)), rostrocapturado)
            id += 1

        cv.imshow("Resultado rostro", captura)

        if id == 100:  
            break

    camara.release()
    cv.destroyAllWindows()
reconocimientofac()