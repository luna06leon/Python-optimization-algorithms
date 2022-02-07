## Genetic Algorithm ROSENBROCK Function

import numpy as np
import matplotlib.pyplot as plt

def Funcion_cost(pop): 
    fitness=(1 + pop[:,0])**2 + 100*(pop[:,1] - pop[:,0]**2)**2
    return fitness
    

def Funcion_padres(pop, fitness, num_padres):
    padres_f = np.empty((num_padres,pop.shape[1])) 
    best_fitness=sorted(fitness) 
    h=0 
    for i in range(num_padres):
        x=np.where(fitness==best_fitness[i])
        if (len(x)>1):
             x=x(h,1)
             h=1+h
        padres_f[i,:]=pop[x,:] 
    return padres_f

def crossover(padres, tam_des): 
    descendencia = np.empty(tam_des) 
    p_crossover = np.uint8(tam_des[1]/2) 

    for k in range(tam_des[0]): 
        p_1 = k%padres.shape[0] 
        p_2 = (k+1)%padres.shape[0]
        descendencia[k, 0:p_crossover] = padres[p_1, 0:p_crossover] 
        descendencia[k, p_crossover:] = padres[p_2, p_crossover:] 
    return descendencia

def mutacion(descendencia_cros, num_mutaciones=1):
    for idx in range(descendencia_cros.shape[0]):
        for mutacion_num in range(num_mutaciones):
            gene_idx = np.random.randint(descendencia_cros.shape[1]) 
            rand_val = np.random.uniform(-2.048, 2, 1)
            descendencia_cros[idx, gene_idx] = rand_val 
    return descendencia_cros

num_var = 2 
num_crom = 200 
num_padres = int(num_crom/2) 
tam_pop = (num_crom,num_var) 
new_pop = np.random.uniform(low=-2.048, high=2.048, size=tam_pop) 
num_gen = 1000 
for generacion in range(num_gen): 
    fitness =  Funcion_cost(new_pop) 
    padres = Funcion_padres(new_pop, fitness, num_padres) 
    descendencia_cros = crossover(padres, tam_des=(tam_pop[0]-padres.shape[0], num_var)) 
    descendencia_mut = mutacion(descendencia_cros, num_mutaciones=1)
    new_pop[0:padres.shape[0], :] = padres
    new_pop[padres.shape[0]:, :] = descendencia_mut


fitness =  Funcion_cost(new_pop)
num_padres_f=len(new_pop)
poblacion_f = Funcion_padres(new_pop, fitness, num_padres_f) 
fitness_f=sorted(fitness, reverse=True) 
print("Mejor Solución: ", poblacion_f[0, :])
print("Valor de Coste menor: ", fitness_f[len(fitness_f)-1])


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (16,8))
x = np.linspace(-2.048,2.048,250)
y = np.linspace(-1,2.048,250)
X, Y = np.meshgrid(x, y)
pop=np.empty((250,2))
pop[:,0],pop[:,1]=x,y
Z=(1 + X)**2 + 100*(Y - X**2)**2
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.plot(poblacion_f[0:101,0],poblacion_f[0:101,1],fitness_f[98:199],color = 'r', marker = '*', alpha = .4)
ax.set_title('Algoritmo Genético 3D con {} generaciones'.format(num_gen))
ax.view_init(40,235)
ax.set_xlabel('x')
ax.set_ylabel('y')

ax1 = fig.add_subplot(1, 2, 2)
ax1.plot(fitness_f)
ax1.set_xlabel("# Iteraciones")
ax1.set_ylabel("Función de Coste")
ax1.set_title('Vaiación Función de Coste con {} iteraciones'.format(num_gen))
plt.show()

#%%
##  Genetic Algorithm GOLDSTEIN PRICE Function
import numpy as np
import matplotlib.pyplot as plt

def Funcion_cost(x1,x2): 
    fact1a=(x1 + x2 + 1)**2
    fact1b=19 - (14*x1) + (3*(x1**2)) - (14*x2) + (6*x1*x2) + (3*(x2**2))
    fact1=1 + (fact1a*fact1b)
	
    fact2a=(2*x1 - 3*x2)**2
    fact2b=18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2=30 + fact2a*fact2b
    return fact1*fact2
    

def Funcion_padres(pop, fitness, num_padres):
    padres_f = np.empty((num_padres,pop.shape[1]))
    best_fitness=sorted(fitness)
    h=0
    for i in range(num_padres):
        x=np.where(fitness==best_fitness[i])
        if (len(x)>1):
             x=x(h,1)
             h=1+h
        padres_f[i,:]=pop[x,:]
    return padres_f

def crossover(padres, tam_des):
    descendencia = np.empty(tam_des)
    p_crossover = np.uint8(tam_des[1]/2)

    for k in range(tam_des[0]):
        p_1 = k%padres.shape[0]
        p_2 = (k+1)%padres.shape[0]
        descendencia[k, 0:p_crossover] = padres[p_1, 0:p_crossover]
        descendencia[k, p_crossover:] = padres[p_2, p_crossover:]
    return descendencia

def mutacion(descendencia_cros, num_mutaciones=1): 
    for idx in range(descendencia_cros.shape[0]):
        for mutacion_num in range(num_mutaciones):
            gene_idx = np.random.randint(descendencia_cros.shape[1]) 
            rand_val = np.random.uniform(-2, 2, 1)
            descendencia_cros[idx, gene_idx] = rand_val  
    return descendencia_cros

num_var = 2
num_crom = 200
num_padres = int(num_crom/2)
tam_pop = (num_crom,num_var) 
new_pop = np.random.uniform(low=-2, high=2, size=tam_pop)
num_gen = 1000
for generacion in range(num_gen):
    fitness =  Funcion_cost(new_pop[:,0],new_pop[:,1])
    padres = Funcion_padres(new_pop, fitness, 
                                      num_padres)
    descendencia_cros = crossover(padres,
                                       tam_des=(tam_pop[0]-padres.shape[0], num_var))

    descendencia_mut = mutacion(descendencia_cros, num_mutaciones=2)

    new_pop[0:padres.shape[0], :] = padres
    new_pop[padres.shape[0]:, :] = descendencia_mut
    
fitness =  Funcion_cost(new_pop[:,0],new_pop[:,1])
num_padres_f=len(new_pop)
poblacion_f = Funcion_padres(new_pop, fitness, num_padres_f)
fitness_f=sorted(fitness, reverse=True)
print("Mejor Solución: ", poblacion_f[0, :])
print("Valor de Coste menor: ", fitness_f[len(fitness_f)-1])

get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (16,8))
x = np.linspace(-2,2,250)
y = np.linspace(-1.5,2,250)
X, Y = np.meshgrid(x, y)
Z=Funcion_cost(X,Y)
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.plot(poblacion_f[0:101,0],poblacion_f[0:101,1],fitness_f[98:199],color = 'r', marker = '*', alpha = .4)
ax.set_title('Algoritmo Genético 3D con {} generaciones'.format(num_gen))
ax.view_init(40,235)
ax.set_xlabel('x')
ax.set_ylabel('y')

ax1 = fig.add_subplot(1, 2, 2)
ax1.plot(fitness_f)
ax1.set_xlabel("# Iteraciones")
ax1.set_ylabel("Función de Coste")
ax1.set_title('Vaiación Función de Coste con {} iteraciones'.format(num_gen))
plt.show()