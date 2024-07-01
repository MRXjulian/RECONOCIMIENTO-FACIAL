def entrenamiento():
    import cv2 
    import numpy as np
    import os
    from time import time
    ##dataRuta es la ruta donde guardamos nuestra data
    dataRuta = 'data'
    listadata=os.listdir(dataRuta)
    #print('data', listadata)
    ids=[]
    rostrosdata=[]
    id=0
    tiempoinicial=time()
    for fila in listadata:
        rutacompleta=dataRuta+'/'+fila
        print('iniciando lectura')
        for archivo in os.listdir(rutacompleta):
            print('imagenes: ', fila+'/'+archivo)
            ids.append(id)
            rostrosdata.append(cv2.imread(rutacompleta+'/'+archivo,0))
        id+=1
        tiempofinal=time()
        tiempototal=tiempofinal-tiempoinicial
        print('tiempo total lectura: '+str(tiempototal))

    entrenamientoMod1 = cv2.face.EigenFaceRecognizer.create()

    print('Iniciando el entrenamiento... Espere')
    entrenamientoMod1.train(rostrosdata, np.array(ids))

    tiempofinaldeentrenamiento = time()
    tiempototaldeentrenamiento = tiempofinaldeentrenamiento - tiempoinicial
    print('Tiempo de entrenamiento total: ' + str(tiempototaldeentrenamiento))

    entrenamientoMod1.write('entrenamientoEigenRecognizer.xml')
    print('Entrenamiento concluido')
entrenamiento()