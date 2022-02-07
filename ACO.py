"""
0=Bogotá, 1=Palmira, 2=Medellín, 3=Pasto, 4=Tuluá, 5=Pereira, 6=Armenia, 7=Caldas, 8=Valledupar, 
9=Monteria, 10=Soledad, 11=Cartagena, 12=Barranquilla, 13=Bucaramanga, 14=Cúcuta

"""
#OPTIMIZACIÓN DE COLONIAS DE HORMIGAS
import numpy as np
import random as rd
import matplotlib.pyplot as plt

## SE CREA LA MATRIZ DE DISTANCIA ENTRE CIUDADES
m_ciudades=np.empty([15,15])
m_ciudades[0,:]=[0, 448, 415.6, 723, 369.7, 320, 280.9, 432, 864.6, 756.3, 995.7, 1038.6, 1001.5, 397.7, 555.8]
m_ciudades[1,:]=[448, 0, 407.7, 400.4, 78, 194.4, 164.9, 383.8, 1105.3, 806.8, 1095.8, 1033.4, 1242.1, 749.3, 942]
m_ciudades[2,:]=[415.6, 407.7, 0, 792.6, 325, 224, 268, 23.3, 748, 404, 696, 631, 706, 392, 584]
m_ciudades[3,:]=[723, 400.4, 792.6, 0, 469, 586, 556, 775, 1496, 1198, 1629, 1424, 1633, 1140, 133]
m_ciudades[4,:]=[369.7, 78, 325, 469, 0, 117, 87.8, 307, 1028, 730, 1022, 956, 1165, 672, 864]
m_ciudades[5,:]=[320, 194.4, 224, 586, 117, 0, 44.6, 191, 913, 614, 1045, 841, 1049, 557, 748]
m_ciudades[6,:]=[280.9, 164.9, 268, 556, 87.8, 44.6, 0, 236, 940, 659, 1073, 885, 1077, 584, 775]
m_ciudades[7,:]=[432, 383.8, 23.3, 775, 307, 191, 236, 0, 769, 425, 717, 651, 727, 413, 604]
m_ciudades[8,:]=[864.6, 1105.3, 748, 1496, 1028, 913, 940, 769, 0, 433, 298, 362, 301, 448, 539]
m_ciudades[9,:]=[756.3, 806.8, 404, 1198, 730, 614, 659, 425, 433, 0, 343, 246, 354, 613, 704]
m_ciudades[10,:]=[995.7, 1095.8, 696, 1629, 1022, 1045, 1073, 717, 298, 343, 0, 129, 12, 580, 670]
m_ciudades[11,:]=[1038.6, 1033.4, 631, 1424, 956, 841, 885, 651, 362, 246, 129, 0, 119, 622, 712]
m_ciudades[12,:]=[1001.5, 1242.1, 706, 1633, 1165, 1049, 584, 727, 301, 354, 12, 622, 0, 586, 676]
m_ciudades[13,:]=[397.7, 749.3, 392, 1140, 672, 557, 584, 413, 448, 613, 580, 622, 586, 0, 199]
m_ciudades[14,:]=[555.8, 942, 584, 133, 864, 748, 775, 604, 539, 704, 670, 712, 676, 199, 0 ]

## SE CREA LA MATRIZ DE PEAJES ENTRE CIUDADES
m_peajes=np.empty([15,15])
m_peajes[0,:]=[0, 99500, 68100, 128000, 99500, 66200, 51700, 40500, 96500, 80500, 132000, 122300, 134200, 41200, 39700]
m_peajes[1,:]=[99500, 0, 90000, 37500, 26300, 49600, 47900, 72700, 147000, 156500, 184700, 186000, 184700, 122000, 137100]
m_peajes[2,:]=[68100, 90000, 0, 118400, 90000, 40400, 54900, 28300, 83300, 79600, 105500, 96000, 105500, 58300, 73400]
m_peajes[3,:]=[128000, 37500, 118400, 0, 37500, 78000, 76300, 101100, 175400, 184900, 213100, 214400, 213100, 150400, 165500]
m_peajes[4,:]=[99500, 26300, 90000, 37500, 0, 49600, 47900, 72700, 147000, 156500, 184700, 186000, 184700, 122000, 137100]
m_peajes[5,:]=[66200, 49600, 40400, 78000, 49600, 0, 14500, 23100, 97400, 106900, 135100, 136400, 135100, 72400, 87500]
m_peajes[6,:]=[51700, 47900, 54900, 76300, 47900, 14500, 0, 37600, 94300, 121400, 132000, 150900, 132000, 69300, 84400]
m_peajes[7,:]=[40500, 72700, 28300, 101100, 72700, 23100, 37600, 0, 74300, 94800, 112000, 124300, 112000, 49300, 64400]
m_peajes[8,:]=[96500, 147000, 83300, 175400, 147000, 97400, 94300, 74300, 0, 45900, 47600, 35700, 47600, 47000, 28200]
m_peajes[9,:]=[80500, 156500, 79600, 184900, 156500, 106900, 121400, 94800, 45900, 0, 44100, 51700, 44100, 8300, 64200]
m_peajes[10,:]=[134200, 184700, 105500, 213100, 184700, 135100, 132000, 112000, 47600, 44100, 0, 34500, 6000, 84700, 65900]
m_peajes[11,:]=[122300, 186000, 96000, 214400, 186000, 136400, 150900, 124300, 35700, 51700, 34500, 0, 34500, 72800, 54000]
m_peajes[12,:]=[134200, 184700, 105500, 213100, 184700, 135100, 132000, 112000, 47600, 44100, 6000, 34500, 0, 84700, 65900]
m_peajes[13,:]=[41200, 122000, 58300, 150400, 122000, 72400, 69300, 49300, 47000, 8300, 84700, 72800, 84700, 0, 15100]
m_peajes[14,:]=[39700, 137100, 73400, 165500, 137100, 87500, 84400, 64400, 28200, 64200, 65900, 54000, 65900, 15100, 0 ]



## LA FUNCIÓN P_RUTA SE ENCARGA DE GENERAR UNA RUTA Y CREAR UNAS VARIABLES QUE GUARDEN LAS DISTANCIAS Y VALOR DE PEAJES DE DICHA RUTA
def P_ruta (a_ciudades, m_ciudades,m_peajes): # A la función le ingresan las dos matrices de valores
    ##Se inicializan las variables
    ruta=list(range(15))
    ## Se agrega la última distancia, es decir, de la última ciudad a la primera
    ruta[0:14]=a_ciudades
    ruta[15]=ruta[0]
    ruta_d=list(range(16))
    ruta_p=list(range(16))
    
    ##En este for se indexa los valores de distancia y peajes que hay entre ciudad y ciudad de la ruta
    ##Para ellos se utilizan dos variable "c y c_sig" que guardan la ciudad actual y la siguiente
    ## Para así poder colocar dichas posiciones en las matrices de distancias y peajes que hay desde una ciudad a la otra
    for j in range(0,15):
        c=ruta[j]
        c_sig=ruta[j+1]
        ruta_d[j]=m_ciudades[c,c_sig]
        ruta_p[j]=m_peajes[c,c_sig]
    
    ruta_d[j+1]=m_ciudades[c_sig,ruta[0]]
    ruta_p[j+1]=m_peajes[c_sig,ruta[0]]##Se busca tanto la distancia como el valor de peajes
                                           ## que hay entre la última ciudad y la primera
    return ruta,ruta_d,ruta_p

## LA FUNCIÓN RUTAS SE ENCARGA DE REALIZAR EL CÁLCULO DEL COSTE DE CADA RUTA
def Rutas (m_ciudades, poblacion,m_peajes,vh_v,v_c):
    ##Se inicializan las variables con sus respectivos tamaños
    pop_rutasd=np.empty([len(poblacion),16]) #Matriz que muestra la distancia de cada indviduo (ruta)
    pop_rutas=np.empty([len(poblacion),16],int) #Matriz de la población con las 16 ciudades
    pop_rutasp=np.empty([len(poblacion),16]) #Matriz que muestra el valor de los peajes de cada indviduo (ruta)
    mtriz_coste=np.empty([len(poblacion),3]) #Matriz que almacena los tres parametros que conforman el coste
                                        # costo peajes + costo horas + costo combustible
                                        
    ##Se utiliza un for para realizar el proceso de cálculo del coste de cada ruta
    for i in range(len(poblacion)):
        #Se utiliza la función P_ruta y se realiza el mismo proceso que en e algoritmo genético
        pop_rutas[i,:],pop_rutasd[i,:],pop_rutasp[i,:]=P_ruta(poblacion[i,:],m_ciudades,m_peajes)
        d=pop_rutasd[i,:]
        p=pop_rutasp[i,:]
        mtriz_coste[i,0]=(sum(d)/67)*v_c
        mtriz_coste[i,1]=sum(p)
        mtriz_coste[i,2]=(sum(d)/70)*vh_v
    return pop_rutas, mtriz_coste


#SE ESTABLECEN LOS PARÁMETROS INICIALES
vh_v=27874
v_c=8525
num_h = 15                   # Número de hormigas
alpha = 1                     # Factor de importancia de las feromonas
beta = 3                      # Factor de importancia de los datos de la función heurística
f_r = 0.3              # Tasa de evaporación de feromonas
num_c = 15                  # Cantidad de ciudades
feromonas = np.ones((num_c,num_c))   
m_c=(m_ciudades/67)*v_c
m_h=(m_ciudades//70)*vh_v
m_heu=m_c + m_h + m_peajes
#Se realizan los cálculos para la función heurística
heuristic = 1 / (np.eye(num_c) + m_heu) - np.eye(num_c)       # Matriz de información heurística, 1 / costo total
iter,itermax = 0,100     # Número de iteraciones a realizar

while iter < itermax:
    r_hs = np.zeros((num_h, num_c)).astype(int) - 1   # Se crea la matriz que contendra el camino de cada hormiga
    firstCity = [i for i in range(15)] 
    #rd.shuffle(firstCity)  # Se asigna aleatoriamente una ciudad de inicio para cada hormiga
    unvisted = []
    p = []
    pAccum = 0
    
    ##Este primer for se encarga de asignar la primera ciudad a la que irá cada hormiga
    for i in range(len(r_hs)):
        r_hs[i][0] = firstCity[i]
    ## Se actualiza gradualmente la próxima ciudad que cada hormiga debe visitar -->
    for i in range(len(r_hs[0]) - 1):       
        for j in range(len(r_hs)):
            for k in range(num_c):
            ##--> Para ésto, se debe tener en cuenta que la siuiente ciudad no haya sido 
            ## visitada anteriormente
                if k not in r_hs[j]:
                    unvisted.append(k)
            ##Posteriormente se calcula la probabilidad que hay de ir a esa siguiente ciudad
            for m in unvisted:
                pAccum += feromonas[r_hs[j][i]][m] ** alpha * heuristic[r_hs[j][i]][m] ** beta
            for n in unvisted:
                p.append(feromonas[r_hs[j][i]][n] ** alpha * heuristic[r_hs[j][i]][n] ** beta / pAccum)
            
            ##Después se crea una ruleta que ayudará a elegir la siguiente ciudad a visitar
            ruleta = np.array(p).cumsum()               
            r = rd.uniform(min(ruleta), max(ruleta))
            for x in range(len(ruleta)):
                #if ruleta[x] >= r:                      
                    r_hs[j][i + 1] = unvisted[x]
                    #break
            unvisted = []
            p = []
            pAccum = 0
    ## Se realiza la actualización de las feromonas volátiles
    feromonas = (1 - f_r) * feromonas            
    ## Se calcula el costo de la ruta creada
    pop_rutas,fitness = Rutas (m_ciudades, r_hs,m_peajes,vh_v,v_c)
    fitness=list(np.sum(fitness,axis=1))
    
    ##Para posteriormente, con base en el costo se realiza la actualización de las feromonas
    ## Que se define como la división entre la intensidad de la feromona y el costo de la ruta
    for i in range(len(r_hs)):
        for j in range(len(r_hs[i]) - 1):
            feromonas[r_hs[i][j]][r_hs[i][j + 1]] += 1 / fitness[i]     
        feromonas[r_hs[i][-1]][r_hs[i][0]] += 1 / fitness[i]
    iter += 1

print("Mejor ruta encontrada:")
print(pop_rutas[fitness.index(min(fitness))])
print("Valor de Coste (COP) menor:")
print(min(fitness))


plt.plot(sorted(fitness, reverse=True))
plt.xlabel("# Iteraciones")
plt.ylabel("Función de Coste")
plt.title('Variación Función de Coste con {} iteraciones'.format(itermax))
plt.show()
