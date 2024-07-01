def reconocimiento():
    import cv2 as cv
    import os
    ##dataRuta es la ruta donde tenemos guardada la data
    dataRuta = 'data'
    listadata=os.listdir(dataRuta)
    entrenamientoMod1 = cv.face.EigenFaceRecognizer.create()
    entrenamientoMod1.read("entrenamientoEigenRecognizer.xml")
    ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    camara=cv.VideoCapture(0)
    while True:
        _,captura=camara.read()
        grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
        idcaptura = grises.copy()
        caras = ruidos.detectMultiScale(grises, 1.3, 5)
        for (x, y, e1, e2) in caras:
            nuevo_ancho = int(e1 * 1.2)
            nuevo_alto = int(e2 * 1.2)        
            nuevo_x = x - int((nuevo_ancho - e1) / 2)
            nuevo_y = y - int((nuevo_alto - e2) / 2)
            rostrocapturado = idcaptura[nuevo_y:nuevo_y + nuevo_alto, nuevo_x:nuevo_x + nuevo_ancho]
            rostrocapturado = cv.resize(rostrocapturado, (160, 160), interpolation=cv.INTER_CUBIC)
            resultado=entrenamientoMod1.predict(rostrocapturado)
            cv.putText(captura, '{}'.format(resultado), (nuevo_x, nuevo_y-5), 1,1.3, (52, 73, 94), 1, cv.LINE_AA)
            if resultado[1]<9000:
                cv.putText(captura, '{}'.format(listadata[resultado[0]]), (nuevo_x, nuevo_y-20), 2,1.3, (52, 73, 94), 1, cv.LINE_AA)
                cv.rectangle(captura, (nuevo_x, nuevo_y), (nuevo_x + nuevo_ancho, nuevo_y + nuevo_alto), (0, 255, 0), 2)
            else:
                cv.putText(captura, 'no encontrado', (nuevo_x, nuevo_y-20), 2,1.3, (52, 73, 94), 1, cv.LINE_AA)
                cv.rectangle(captura, (nuevo_x, nuevo_y), (nuevo_x + nuevo_ancho, nuevo_y + nuevo_alto), (0, 255, 0), 2)
        cv.imshow('resultados', captura)
        if cv.waitKey(1)==ord("q"):
            break
    camara.release()
    cv.destroyAllWindows()
reconocimiento()