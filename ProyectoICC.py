from sklearn.datasets import load_digits
from PIL import Image
import numpy as np
import pandas as pd

Digits = load_digits()
CaracteristicasDigits = Digits.data
EtiquetasDigits = Digits.target

def ProcesamientoImagenes(ruta):
    ImagenReal = Image.open(ruta) # se convierte a matrix (tipo pillow)
    ImagenGris = ImagenReal.convert("L") # [0-255] 0 es negro y blanco es 255
    ImagenGrisRedimensionada = ImagenGris.resize((8,8)) # 
    
    MatrizImagenProcesada = np.array(ImagenGrisRedimensionada).astype(float)
    
    MatrizImagenProcesada = (-1*MatrizImagenProcesada)+ 255 # ahora 255 es negro y 0 es blanco 
    
    MatrizImagenProcesada = (MatrizImagenProcesada/255) *16  # Escalado entre [0- 16]
    Vector64 = MatrizImagenProcesada.flatten()
    return Vector64


def DistanciaEuclidiana(Imagen1,Imagen2):
    return np.sqrt(np.sum((Imagen1 -Imagen2)**2))

def ObtenerDistancia(Vector):
    distancia = []
    for i in range(len(CaracteristicasDigits)):
        AuxDistancia = DistanciaEuclidiana(Vector,CaracteristicasDigits[i])
        distancia.append((AuxDistancia,EtiquetasDigits[i]))
    return distancia


def VecinosMasCercanos3(distancia):
    distancia.sort(key=lambda x: x[0])
    return distancia[:3]


DiccionarioNumerosClasificados = {}
i = 1
for NumeroImagen in range(1,28):
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

print(DiccionarioNumerosClasificados)
DataDiccionarioNumerosClasificados = pd.DataFrame.from_dict(DiccionarioNumerosClasificados, orient="index")
DataDiccionarioNumerosClasificados[["Vecino 1", "Vecino 2", "Vecino 3"]] = pd.DataFrame(DataDiccionarioNumerosClasificados[1].tolist(), index=DataDiccionarioNumerosClasificados.index)
DataDiccionarioNumerosClasificados.drop([1],axis = 1, inplace= True)

DataDiccionarioNumerosClasificados = DataDiccionarioNumerosClasificados.rename(columns={0: "NumerosOriginales"})
DataDiccionarioNumerosClasificados.to_csv("NumerosClasificados.csv")

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
        ListaNumeroDetectado.append(0)

SerieListaNumeroDetectado = pd.Series(ListaNumeroDetectado,name="NumeroDetectado",index=DataDiccionarioNumerosClasificados.index)
DataDiccionarioNumerosClasificados["NumeroDetectado"] = SerieListaNumeroDetectado
DataDiccionarioNumerosClasificados.to_csv("NumerosClasificados&NumerosDetectados.csv")
print(DataDiccionarioNumerosClasificados)


MatrizConfusion10 = np.zeros((10,10))
for indice, fila in DataDiccionarioNumerosClasificados.iterrows():
    NumeroOriginal = fila["NumerosOriginales"]
    NumeroDetectado = fila["NumeroDetectado"]
    MatrizConfusion10[int(NumeroOriginal),int(NumeroDetectado)] += 1

print(MatrizConfusion10)