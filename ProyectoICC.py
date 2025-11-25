"""
Objetivo: En este proyecto hemos creado un algoritmo que realiza
"""
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
    ImagenReal = Image.open(ruta) 
    ImagenGris = ImagenReal.convert("L")
    ImagenGrisRedimensionada = ImagenGris.resize((8,8))
    
    MatrizImagenProcesada = np.array(ImagenGrisRedimensionada).astype(float)
    
    MatrizImagenProcesada = (-1*MatrizImagenProcesada)+ 255
    
    MatrizImagenProcesada = (MatrizImagenProcesada/255) *16 
    Vector64 = MatrizImagenProcesada.flatten() 
    return Vector64
################################################


####################################### KNN Manual #######################################


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


#########################################################################################


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
DataDiccionarioNumerosClasificados.to_csv("NumerosClasificados/DataFrameNumerosClasificados.csv")
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
DataDiccionarioNumerosClasificados.to_csv("NumerosClasificados_&_NumerosDetectados/DataFrameNumerosClasificados_&_NumerosDetectados.csv")
################################################


################ Creacion de Matriz de Confusion 10 clases ################

MatrizConfusion10 = np.zeros((10,10))
for indice, fila in DataDiccionarioNumerosClasificados.iterrows():
    NumeroOriginal = fila["NumerosOriginales"]
    NumeroDetectado = fila["NumeroDetectado"]
    MatrizConfusion10[int(NumeroOriginal),int(NumeroDetectado)] += 1

DataFrameMatrizConfusion10 = pd.DataFrame(MatrizConfusion10)
DataFrameMatrizConfusion10.to_csv("MatrizConfusion_10_Clases/DataFrameMatrizConfusion10.csv",index = False)

################################################


################ Creacion de Matriz de confunsion 2 clases ################
Matrices2clasesdiccionario = {}
NumerosOriginalesM = DataDiccionarioNumerosClasificados["NumerosOriginales"]
for numero in NumerosOriginalesM:
    VerdaderoPositivo = 0
    VerdaderoNegativo = 0
    FalsoPositivo = 0
    FalsoNegativo = 0
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
    
i = 0
for key,valor in Matrices2clasesdiccionario.items():
    print("")
    print(f"Matriz de confusion 2 clase: {key}")
    DataMatriz2Clases = pd.DataFrame(valor,index=["Si", "No"], columns=["Predijo Si", "Predijo No"])
    DataMatriz2Clases.to_csv(f"MatricesConfusion_2/MatrizConfusion{i}.csv")
    i+=1
    print(DataMatriz2Clases)
    print("################")


################################################


################ Calculo de metricas ################
ColumnasMetricas = ["Accuracy","PrecisionVerdadero","RecallVerdadero","F1_Score_Verdadero","PrecisionFalso","RecallFalso","F1_Score_Verdadero"]
MetricasDataFrameFinal = pd.DataFrame(columns=ColumnasMetricas)


for i in range(10):
    Matriz2 = pd.read_csv(f"MatricesConfusion_2/MatrizConfusion{i}.csv")
    OperacionesLista = []
    Metricas = []
    for indice, fila in Matriz2.iterrows():
        VerdaderoPositivo = fila["Predijo Si"]
        VerdaderoNegativo = fila["Predijo No"]
        if indice == "No":
            FalsoPositivo = fila["Predijo No"]
            FalsoNegativo = fila["Predijo Si"]
            OperacionesLista.append(FalsoPositivo)
            OperacionesLista.append(FalsoNegativo)
        OperacionesLista.append(VerdaderoPositivo)
        OperacionesLista.append(VerdaderoNegativo)
    # Patron de Operaciones lista : [VP , FP, FN,VN]
    
    # Patron Metricas Lista: [Accuracy, PrecisionVerdadero, RecallVerdadero, F1_Score_Verdadero, PrecisionFalso, RecallFalso, F1_Score_Verdadero]
    
    ###### Accuracy ######
    Accuracy = (OperacionesLista[0] + OperacionesLista[3])/(sum(OperacionesLista))
    Metricas.append(Accuracy)
    ######################
    
    ###### Precision Verdadero ######
    PrecisionVerdadero = (OperacionesLista[0])/max(1,(OperacionesLista[0]+OperacionesLista[1]))
    Metricas.append(PrecisionVerdadero)
    ######################
    
    ###### Recall Verdadero ######
    RecallVerdadero = (OperacionesLista[0])/max(1,(OperacionesLista[0]+ OperacionesLista[2]))
    Metricas.append(RecallVerdadero)
    ######################
    
    ###### F1 Score Verdadero ######
    F1_Score_Verdadero = 2* ((PrecisionVerdadero*RecallVerdadero)/max(1,(PrecisionVerdadero + RecallVerdadero)))
    Metricas.append(F1_Score_Verdadero)
    #####################
    
    ###### Precision Falso ######
    PrecisionFalso = (OperacionesLista[3])/max(1,(OperacionesLista[3] + OperacionesLista[2]))
    Metricas.append(PrecisionFalso)
    #####################
    
    ###### Recall Falso ######
    RecallFalso = (OperacionesLista[3])/max(1,(OperacionesLista[3] + OperacionesLista[1]))
    Metricas.append(RecallFalso)
    #####################
    
    ###### F1 Score Falso ######
    F1_Score_Falso = 2* ((PrecisionFalso*RecallFalso)/max(1,(PrecisionFalso + RecallFalso)))
    Metricas.append(F1_Score_Falso)
    #####################
    
    MetricasDataFrameAux = pd.Series(Metricas,index = ["Accuracy","PrecisionVerdadero","RecallVerdadero","F1_Score_Verdadero","PrecisionFalso","RecallFalso","F1_Score_Verdadero"])
    MetricasDataFrameFinal = pd.concat([MetricasDataFrameFinal,MetricasDataFrameAux.to_frame().T],ignore_index=True)
    
    
MetricasDataFrameFinal.to_csv("Metricas/MetricasDataFrameFinal.csv")

################################################
