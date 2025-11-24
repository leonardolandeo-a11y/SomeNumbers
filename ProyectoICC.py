
################ Importacion ################

from sklearn.datasets import load_digits
from PIL import Image
import numpy as np
import pandas as pd
################################################

################ Cargar Digits ################

Digits = load_digits()
CaracteristicasDigits = Digits.data
EtiquetasDigits = Digits.target
################################################


################ Procesamiento de Imagenes ################

def ProcesamientoImagenes(ruta):
    ImagenReal = Image.open(ruta) # se convierte a matrix (tipo pillow)
    ImagenGris = ImagenReal.convert("L") # [0-255] 0 es negro y blanco es 255
    ImagenGrisRedimensionada = ImagenGris.resize((8,8)) # Se redimensiona la matrix a 8x8
    
    MatrizImagenProcesada = np.array(ImagenGrisRedimensionada).astype(float)
    
    MatrizImagenProcesada = (-1*MatrizImagenProcesada)+ 255 # ahora 255 es negro y 0 es blanco 
    
    MatrizImagenProcesada = (MatrizImagenProcesada/255) *16  # Escalado entre [0- 16]
    Vector64 = MatrizImagenProcesada.flatten() # Se aplana la mat
    return Vector64
################################################


################ KNN Manual ################

################ Distancia Euclidiana ################

def DistanciaEuclidiana(Imagen1,Imagen2):
    return np.sqrt(np.sum((Imagen1 -Imagen2)**2))

################################################

################ Obtener Distancia ################

def ObtenerDistancia(Vector):
    distancia = []
    for i in range(len(CaracteristicasDigits)):
        AuxDistancia = DistanciaEuclidiana(Vector,CaracteristicasDigits[i])
        distancia.append((AuxDistancia,EtiquetasDigits[i]))
    return distancia
################################################


################ Obtener vecinos ################

def VecinosMasCercanos3(distancia):
    distancia.sort(key=lambda x: x[0])
    return distancia[:3]
################################################

################################################


################ Aplicacion de las funciones a cada imagen ################

DiccionarioNumerosClasificados = {}
i = 0
for NumeroImagen in range(1,31):
    ruta = f"Numbers/{i}_Img{NumeroImagen}.jpeg"
    print(f"RutaImagen: {ruta}")
    print(f"Imagen: {i}")
    VectorImagenProcesada = ProcesamientoImagenes(ruta)
    Distancias = ObtenerDistancia(VectorImagenProcesada)
    VecinosKNN = VecinosMasCercanos3(Distancias)
    ListaVecinosAux = []
    for distanciaAux,targetAux in VecinosKNN:
        ListaAuxVecinos2 = []
        print(f" Target: {targetAux}, Distancia: {round(distanciaAux,2)}")
        ListaAuxVecinos2.append(targetAux)
        ListaAuxVecinos2.append(distanciaAux)
        ListaVecinosAux.append(ListaAuxVecinos2)
    DiccionarioNumerosClasificados[NumeroImagen] = (i,ListaVecinosAux)
    if NumeroImagen %3 == 0:
        i +=1
    
    print("################")
    print("")

################################################

################ Creacion de DataFrame1 ################

DataDiccionarioNumerosClasificados = pd.DataFrame.from_dict(DiccionarioNumerosClasificados, orient="index")
DataDiccionarioNumerosClasificados[["Vecino 1", "Vecino 2", "Vecino 3"]] = pd.DataFrame(DataDiccionarioNumerosClasificados[1].tolist(), index=DataDiccionarioNumerosClasificados.index)
DataDiccionarioNumerosClasificados.drop([1],axis = 1, inplace= True)

DataDiccionarioNumerosClasificados = DataDiccionarioNumerosClasificados.rename(columns={0: "NumerosOriginales"})
DataDiccionarioNumerosClasificados.to_csv("NumerosClasificados.csv")
################################################


################ Deteccion de numeros ################

ListaNumeroDetectado = []
for indice, fila in DataDiccionarioNumerosClasificados.iterrows():
    NumeroOriginal = fila["NumerosOriginales"]
    
    Vecinos = [fila["Vecino 1"][0], fila["Vecino 2"][0], fila["Vecino 3"][0]]
    coincidencias = 0 
    for veci in Vecinos:
        if int(veci) == NumeroOriginal:
            coincidencias += 1
    
    if coincidencias == 3 or coincidencias == 2:
        print(f"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {NumeroOriginal}")
        ListaNumeroDetectado.append(NumeroOriginal)
    else:
        print("Pronto...")
        ListaNumeroDetectado.append(-1)

################################################

################ Creacion de DataFrame2 ################

SerieListaNumeroDetectado = pd.Series(ListaNumeroDetectado,name="NumeroDetectado",index=DataDiccionarioNumerosClasificados.index)
DataDiccionarioNumerosClasificados["NumeroDetectado"] = SerieListaNumeroDetectado
DataDiccionarioNumerosClasificados.to_csv("NumerosClasificados&NumerosDetectados.csv")
################################################


################ Creacion de Matriz de Confusion 10 clases ################

MatrizConfusion10 = np.zeros((10,10))
for indice, fila in DataDiccionarioNumerosClasificados.iterrows():
    NumeroOriginal = fila["NumerosOriginales"]
    NumeroDetectado = fila["NumeroDetectado"]
    MatrizConfusion10[int(NumeroOriginal),int(NumeroDetectado)] += 1

DataFrameMatrizConfusion10 = pd.DataFrame(MatrizConfusion10)
DataFrameMatrizConfusion10.to_csv("DataFrameMatrizConfusion10.csv",index = False)

################################################


################ Creacion de Matriz de confunsion 2 clases ################
Matrices2clasesLista = []
NumerosOriginalesM = DataDiccionarioNumerosClasificados["NumerosOriginales"]
for numero in NumerosOriginalesM:
    VerdaderoPositivo = 0
    VerdaderoNegativo = 0
    FalsoPositivo = 0
    FalsoNegativo = 0
    Matrices2clasesdiccionario = {}
    for indice, fila in DataDiccionarioNumerosClasificados.iterrows():
        NumeroReal = fila["NumerosOriginales"]
        NumeroDetectado = fila["NumeroDetectado"]
        
        if NumeroReal == numero and NumeroDetectado == numero:
            VerdaderoPositivo += 1
        elif NumeroReal == numero and NumeroDetectado != numero:
            FalsoNegativo += 1
        elif NumeroReal != numero and NumeroDetectado != numero:
            VerdaderoNegativo +=1
        elif NumeroReal != numero and NumeroDetectado == numero:
            FalsoPositivo += 1
    
    MatrizClases_2 = [
        [VerdaderoPositivo, FalsoNegativo],
        [FalsoPositivo, VerdaderoNegativo]
    ]
    Matrices2clasesdiccionario[numero] = MatrizClases_2
    Matrices2clasesLista.append(Matrices2clasesdiccionario)
    
for dictAux in Matrices2clasesLista:
    for key,valor in dictAux.items():
        print("")
        print(f"Matriz de confusion 2 clase: {key}")
        DataMatriz2Clases = pd.DataFrame(valor,index=["Si", "No"], columns=["Predijo Si", "Predijo No"])
        print(DataMatriz2Clases)
        print("################")

################################################
