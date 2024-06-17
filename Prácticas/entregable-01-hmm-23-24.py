#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================================
# Ampliación de Inteligencia Artificial. Tercer curso.
# Grado en Ingeniería Informática - Tecnologías Informáticas
# Curso 2023-24
# Primer entregable
# ===========================================================

# -----------------------------------------------------------
# NOMBRE: Adrián 
# APELLIDOS: González Lillo
# -----------------------------------------------------------



# Escribir el código Python de las funciones que se piden en el
# espacio que se indica en cada ejercicio.

# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS FUNCIONES QUE SE
# PIDEN (aquellas funciones con un nombre distinto al que se pide en el
# ejercicio NO se corregirán).

# ESTE ENTREGABLE SUPONE 1 PUNTO DE LA NOTA TOTAL

# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: la realización de los ejercicios es un
# trabajo personal, por lo que deben completarse por cada estudiante de manera
# individual.  La discusión con los compañeros y el intercambio de información
# DE CARÁCTER GENERAL con los compañeros se permite, pero NO AL NIVEL DE
# CÓDIGO. Igualmente el remitir código de terceros, obtenido a través
# de la red, mediante herramientas de genración de código, o cualquier otro medio
# SE CONSIDERARÁ PLAGIO.

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# estudiantes involucrados, independientemente de otras medidas de carácter
# DISCIPLINARIO que se pudieran tomar. Por tanto a estos estudiantes NO se les
# conservará, para futuras convocatorias, ninguna nota que hubiesen obtenido
# hasta el momento.
# *****************************************************************************



# Lo que sigue es la implementación de la clase HMM vista en la práctica de clase,
# que representa de manera genérica un modelo oculto de Markov, junto con los
# dos ejemplos de las diapositivas.

class HMM(object):
    """Clase para definir un modelo oculto de Markov"""

    def __init__(self,estados,mat_ini,mat_trans,observables,mat_obs):
        """El constructor de la clase recibe una lista con los estados, otra
        lista con los observables, un diccionario representado la matriz de
        probabilidades de transición, otro diccionario con la matriz de
        probabilidades de observación, y otro con las probabilidades de
        inicio. Supondremos (no lo comprobamos) que las matrices son 
        coherentes respecto de la  lista de estados y de observables."""
        
        self.estados=estados
        self.observables=observables
        self.a={(si,sj):ptrans
                for (si,l) in zip(estados,mat_trans)
                for (sj,ptrans) in zip(estados,l)}
        self.b={(si,vj):pobs
                for (si,l) in zip(estados,mat_obs)
                for (vj,pobs) in zip(observables,l)}
        self.pi=dict(zip(estados,mat_ini))


ej1_hmm=HMM(["c","f"],
            [0.8,0.2],
            [[0.7,0.3],[0.4,0.6]],
            [1,2,3],   
            [[0.2,0.4,0.4],[0.5,0.4,0.1]])
            

ej2_hmm=HMM(["l","no l"],
            [0.5,0.5],
            [[0.7, 0.3], [0.3,0.7]],
            ["u","no u"],   
            [[0.9,0.1],[0.2,0.8]])


# Lo que sigue son implementaciones de los algoritmos de avance y de viterbi:

def avance(hmm,observaciones):
    """Algoritmo de avance (forward). Dada una secuencia de observaciones
    hasta el instante t, devuelve la probabilidad de cada estado 
    en el instante t y la probabilidad de la secuencia de observaciones"""
    
    alpha_list=[hmm.b[(e,observaciones[0])]*hmm.pi[e] 
                for e in hmm.estados]
    for o in observaciones[1:]:
        alpha_list=[hmm.b[(e,o)]*sum(hmm.a[(e1,e)]*a 
                                       for (e1,a) in zip(hmm.estados,alpha_list)) 
                    for e in hmm.estados]
        
    prob_secuencia=sum(alpha_list)
    alpha_list_n=[(p/prob_secuencia) for p in alpha_list] 
    return dict(zip(hmm.estados,alpha_list_n)) 


def viterbi(hmm,observaciones):
        """Algoritmo de Viterbi. Dada una secuancia de observaciones, devuelve
        la secuencia de estados más probable"""

        def max_argmax(l):
            """Si l es una lista numérica, devuelve el índice del valor máximo
            y ese valor"""
            return max(enumerate(l),key=lambda x:x[1])

        def mult_seg(x,p):
            """Multiplica por x la segunda componente de un par p"""
            return (p[0],x*p[1])

        pr_nu_list=[(None,hmm.b[(e,observaciones[0])]*hmm.pi[e]) 
                              for e in hmm.estados]
        historia=dict()
        for (i,o) in enumerate(observaciones[1:]):

            historia[i]=pr_nu_list
            pr_nu_list=[mult_seg(hmm.b[(e,o)],max_argmax(hmm.a[(e1,e)]*alpha 
                                                for (e1,(_,alpha)) in zip(hmm.estados,pr_nu_list)))
                                            for e in hmm.estados]
        (m,(ptr,_))=max(enumerate(pr_nu_list),key=lambda x:x[1][1])
        secuencia=[hmm.estados[m]]
        for k in range(len(observaciones)-2,-1,-1):
            secuencia.append(hmm.estados[ptr])
            (ptr,_)=historia[k][ptr]
        return list(reversed(secuencia))    




# Un algoritmo de muestreo de secuencias de estados y observaciones en HMMs, 
# también visto en la práctica realizada en clase:
        

from collections import defaultdict
import random

def muestreo_hmm(hmm,n):
    """Genera una secuencia de estados junto con su correspondiente
    secuencia de observaciones, de longitud n"""

    def muestreo_ini():
        """Devuelve aleatoriamente (siguiendo las probabilidades del
        modelo) el primer estado en la secuencia"""

        aleatorio=random.random()
        acum=0
        for x in hmm.estados:
            acum+=hmm.pi[x]
            if acum > aleatorio:
                return x
            
    def muestreo_cond(estado,lista,mat_prob):
        """Esta función se usa para dos tareas muy similares:
           - Dado el estado actual, devolver aleatoriamente (pero
             siguiendo las probabilidades del modelo), el estado sigueinte
             en la secuencia. En ese caso, lista es la lista de estados y
             mat_prob la matriz de probabilidades de transición. 

           - Dado el estado actual, devolver aleatoriamento (pero
             siguiendo las probabilidades del modelo), la observación
             correspondiente a ese estado. En ese caso, lista es la lista
             de observables y mat_prob es la matriz de probabilidades de
             observación."""

        aleatorio=random.random()
        acum=0
        for x in lista:
            acum+=mat_prob[estado,x]
            if acum > aleatorio:
                return x

            
    secuencia_estados=[muestreo_ini()]
    secuencia_obs=[muestreo_cond(secuencia_estados[-1],hmm.observables,hmm.b)]

    for _ in range(n-1):
        secuencia_estados.append(muestreo_cond(secuencia_estados[-1],hmm.estados,hmm.a)) 
        secuencia_obs.append(muestreo_cond(secuencia_estados[-1],hmm.observables,hmm.b))

    return secuencia_estados, secuencia_obs



# Y una función que realiza una estimación de  P(X_t| o1,o2,...,o_n), 
# usando la función de muestreo anterior. 





# ===============================================================
# Parte 1: Probabilidad de filtrado usando muestreos ponderados
#=====================================================================


# Modificar las dos funciones anteriores para realizar la estimación de
# P(X_t| o1,o2,...,o_n), usando muestreo con ponderación por verosimilitud.  
# 
# Es decir, se trata de definir:
    
# * Una función muestreo_hmm_por_verosimilitud(hmm, obs) que recibe un modelo 
#   oculto de Markov hmm y una secuencia de observaciones obs, y genera una
#   secuencia de estados mediante muestreo, dando por supuesto que la secuencia de 
#   observaciones correspondiente es la dada por obs. Además de esa  secuencia 
#   de estados, devuelve una ponderación de la muestra generada. Esa ponderación
#   se obtiene multiplicando las probabilidades de que cada observación de la 
#   secuencia dada hubiera ocurrido si se hubiese muestreado. 


# * Una función estima_filtrado_por_verosimilitud(hmm,obs,n_muestras) que 
#   que recibiendo un modelo oculto de Markov hmm, una secuencia de observaciones
#   y un entero n_muestras indicando el número de muestras a generar, haga una
#   estimación de las probabilidades de cada estado dado que se ha observado la 
#   secuencia de observaciones dada, usando para ello la función de 
#   muestreo_hmm_por_verosimilitud anterior.

# Ejemplos (no necesariamente debe salir lo mismo):
    
# >>> muestreo_hmm_por_verosimilitud(ej1_hmm,[2, 1, 1, 1, 3])
# (['c', 'c', 'c', 'c', 'c'], 0.00128)    

# >>> muestreo_hmm_por_verosimilitud(ej2_hmm,['u', 'no u', 'u'])
# (['l', 'l', 'l'], 0.081)


# >>> estima_filtrado_por_verosimilitud(ej1_hmm,[3,1,3,2],100000)
# {'c': 0.6477223509962132, 'f': 0.35227764900378683}

# >>> estima_filtrado_por_verosimilitud(ej2_hmm,["u","u","no u"],100000)
# {'l': 0.19304969498121577, 'no l': 0.8069503050187843}
   
def muestreo_hmm_por_verosimilitud(hmm,obs):
    """Genera una secuencia de estados junto con su correspondiente
    secuencia de observaciones, de longitud n"""

    def muestreo_ini():
        """Devuelve aleatoriamente (siguiendo las probabilidades del
        modelo) el primer estado en la secuencia"""

        aleatorio=random.random()
        acum=0
        for x in hmm.estados:
            acum+=hmm.pi[x]
            if acum > aleatorio:
                return x
            
    def muestreo_cond(estado,lista,mat_prob):
        """Esta función se usa para dos tareas muy similares:
           - Dado el estado actual, devolver aleatoriamente (pero
             siguiendo las probabilidades del modelo), el estado sigueinte
             en la secuencia. En ese caso, lista es la lista de estados y
             mat_prob la matriz de probabilidades de transición. 

           - Dado el estado actual, devolver aleatoriamento (pero
             siguiendo las probabilidades del modelo), la observación
             correspondiente a ese estado. En ese caso, lista es la lista
             de observables y mat_prob es la matriz de probabilidades de
             observación."""

        aleatorio=random.random()
        res=0
        for x in lista:
            res+=mat_prob[estado,x]
            if res > aleatorio:
                return x
            
    secuencia_estados=[muestreo_ini()]

    for _ in range(len(obs)-1):
        secuencia_estados.append(muestreo_cond(secuencia_estados[-1],hmm.estados,hmm.a)) 

    res = 1
    for z1,z2 in zip(secuencia_estados, obs):
        res = res * hmm.b[(z1,z2)] 

    return secuencia_estados, res


def estima_filtrado_por_verosimilitud(hmm,obs,n_muestras):      
        ocurrencias_estados={e:0 for e in hmm.estados}
        for _ in range(n_muestras):
            secuencia_est,dato=muestreo_hmm_por_verosimilitud(hmm,obs)
            ocurrencias_estados[secuencia_est[-1]]+= dato
        norm = sum(ocurrencias_estados.values())
        return {e:m_e/norm for e,m_e in ocurrencias_estados.items()}


#=====================================================================
# Parte 2: Aplicación al movimiento de robots
#=====================================================================

# Vamos ahora a aplicar el algoritmo de viterbi para experimentar sobre
# un problema simple de localización de robots que se mueve en una cuadrícula.
# Esta aplicación es similar (aunque no igual) a la  descrita en la sección 
# 14.3.2 del libro "Artificial Intelligence: A Modern Approach (4th edition)" 
# de S. Russell y P. Norvig.

# Supongamos que tenemos la siguiente lista de strings, que representa una
# cuadrícula bidimensional, sobre la que se desplaza un robot:

#     ["ooooxoooooxoooxo",
#      "xxooxoxxoxoxExxo",
#      "xoEoxoxxoooooxxo",
#      "ooxooExooooxoooo"]

# Aquí la "x" representa una casilla bloquedada, y la "o" representa una
# casilla libre en la que puede estar el robot. La "E" representa también casilla 
# libre, pero con un tratamiento especial que más adelante se detalla. 

#   El robot puede iniciar su movimiento en cualquiera de las casillas libres,
# con igual probabilidad. En cada instante, si está en una casilla marcada 
# con "o", el robot se mueve a una casilla contigua no oblicua: al norte, al sur, 
# al este o al oeste, siempre que dicha casilla no esté bloqueda. Si la casilla en la 
# que se encuentra está marcada con "E", entonces además de poderse mover al 
# norte, sur, este y oeste, puede moverse a una casilla en oblicuo (NE,NO,SE,SO), 
# entendiendo siempre que la casilla a la que se mueva no puede estar bloqueada.     

# El movimiento del robot está sujeto a incertidumbre (no está determinada a qué 
# vecina se moverá), pero sabemos que se puede mover CON IGUAL PROBABILIDAD A CADA
# CASILLA VECINA NO BLOQUEADA, teniendo en cuenta que la noción de vecina accesible
# difiere si se trata de una casilla especial.     

#   Desgraciadamente, el robot no nos comunica en qué casilla se encuentra en
# cada instante de tiempo, ni nosotros podemos observarlo. Lo único que el
# robot puede observar en cada casilla son las direcciones hacia las que
# existen obstáculos (es decir, casillas bloqueadas o paredes). Por ejemplo, una
# observación "N-S" representa que el robot ha detectado que desde la casilla
# en la que está, al norte y al sur no pueda transitar, pero que sí puede
# hacerlo a las casillas que están al este y al oeste. Tanto en las casillas normales
# como en las especiales, los sensores solo detectan posibles bloqueos al N, S, E y O 
# (es decir, el sensor nunca detecta en oblicuo, tampoco en las casillas especiales)    

#   Para acabar de complicar la cosa, los sensores de obstáculos que tiene el
# robot no son perfectos, y están sujetos a una probabilidad de error.
# Supondremos que hay una probabilidad epsilon de que la detección de
# obstáculo en una dirección sea errónea (y por tanto, hay una probabilidad
# 1-epsilon de que sea correcta). Supondremos también que los errores en
# cada una de las cuatro direcciones son independientes entre sí. Esto nos
# permite calcular la probabilidad de las observaciones dados los estados, como
# ilustramos a continuación.

#   Por ejemplo, supongamos que X y E son, respectivamente, las variables
# aleatorias que indican la casilla en la que está el robot y la observación
# que realiza el robot. Supongamos también que c es una casilla que hacia el
# norte y el este tiene obstáculos, y que tiene casillas transitables al sur y
# al oeste. Si por ejemplo el robot informara que existen obstáculos al sur y
# al este, la probabilidad de esto sería 

#     P(E="S-E"|X=c) = (epsilon)^2 * (1-epsilon)^2 

# (ya que habría errado en dos direcciones, norte y sur, y acertado en otras
# dos, este y oeste). 

# Por el contrario, la probabilidad de que en ese mismo estado el robot
# informara de obstáculos al norte, sur y este, sería 

#     P(E="N-S-E"|X=c) = epsilon * (1-epsilon)^3 

# (ya que habría errado en una dirección y acertado en tres).

#robot0.b
#{((0, 0), (0, 0, 0, 0)): 0.008100000000000001,
# ((0, 0), (0, 0, 0, 1)): 0.07290000000000002,
# ((0, 0), (0, 0, 1, 0)): 0.0009000000000000002,
# ((0, 0), (0, 0, 1, 1)): 0.008100000000000001,

# Se pide:

# Definir una clase Robot, subclase de HMM, cuyo constructor reciba una lista
# de strings del estilo de la del ejemplo anterior, y un error epsilon, generando a
# partir de la misma un objeto de la clase HMM. Importante: se pide hacerlo de 
# manera genérica, no solo para la cuadrícula del ejemplo. 

# Aplicar el algoritmo de Viterbi a varias secuencias de observaciones del robot,
# para estimar las correspondientes secuencias de casillas más probables por
# las que ha pasado el robot, en la cuadrícula del ejemplo.

from collections import defaultdict
import math

def crea_estados(cuadricula):
        res = []
        for i in range(len(cuadricula)):
                for j in range(len(cuadricula[i])):    
                        if(cuadricula[i][j]!='x'):
                                res.append((i,j))
        return res
    
def matriz_pi_Robot(cuad):   
    estados = crea_estados(cuad)        
    tamaño = len(estados)
    proababilidad = 1/tamaño
    dic = defaultdict(int)
    for e in estados:
        dic[e]=proababilidad
    return dic 

def matriz_a_robot(cuad):
    estados = crea_estados(cuad)
    dic=defaultdict(float)
    for e1 in estados:
        listaVecinos=[]
        noVecinos = []
        for e2 in estados:
            if(cuad[e1[0]][e1[1]]=='E'): # Si la casilla es E
                if ((abs(e1[0]-e2[0])==1) and ((abs(e1[1]-e2[1])==1))): # Miro las diagonales disponibles
                    listaVecinos.append(e2) #añado las disponibles
                elif(((abs(e1[0]-e2[0])==1) and e2[1] == e1[1]) or 
                     ((abs(e1[1]-e2[1])==1) and e2[0] == e1[0])): # Miro las adyacentes disponibles
                    listaVecinos.append(e2) #añado las adyacentes disponibles
                else:
                    noVecinos.append(e2)    # Si no se cumple lo meto en no Vecino
            else:
                if (((abs(e1[0]-e2[0])==1) and e2[1] == e1[1]) or 
                     ((abs(e1[1]-e2[1])==1) and e2[0] == e1[0])): #Si es casilla o solo miro las adyacentes
                    listaVecinos.append(e2) #meto las adyacentes
                else:
                    noVecinos.append(e2)
        for e in listaVecinos:
            dic[(e1,e)]=1/len(listaVecinos)
        for e in noVecinos:
            dic[(e1,e)]=0      
    return dic         

def observables_robot():
    return [(0, 0, 0, 0),(0, 0, 0, 1),(0, 0, 1, 0),(0, 0, 1, 1),(0, 1, 0, 0),
    (0, 1, 0, 1),(0, 1, 1, 0),(0, 1, 1, 1),(1, 0, 0, 0),(1, 0, 0, 1),
    (1, 0, 1, 0),(1, 0, 1, 1),(1, 1, 0, 0),(1, 1, 0, 1),(1, 1, 1, 0),
    (1, 1, 1, 1)] 
    
def existe_estado(estado, estados):
    res = False
    for e in estados:
        if estado == e:
            res = True
    return res

def calcula_probabilidad(estado, obs, epsilon, estados):
    aciertos = 0
    fallos = 0
    #Norte libre
    if(existe_estado((estado[0] -1, estado[1]),estados)):
        if obs[0] == 0:
            aciertos += 1
        else:
            fallos += 1
    else:
        if(obs[0]==0):
            fallos += 1
        else:
            aciertos += 1
    #Sur libre
    if(existe_estado((estado[0] + 1, estado[1]),estados)):
        if obs[1] == 0:
            aciertos += 1
        else:
            fallos += 1
    else:
        if(obs[1]==0):
            fallos += 1
        else:
            aciertos += 1
    #Este libre
    if(existe_estado((estado[0], estado[1]+1),estados)):
        if obs[2] == 0:
            aciertos += 1
        else:
            fallos += 1
    else:
        if(obs[2]==0):
            fallos += 1
        else:
            aciertos += 1
    #Oeste libre
    if(existe_estado((estado[0], estado[1]-1), estados)):
        if obs[3] == 0:
            aciertos += 1
        else:
            fallos += 1
    else:
        if(obs[3]==0):
            fallos += 1
        else:
            aciertos += 1
    return (epsilon**(fallos)) * ((1-epsilon)**aciertos)
    

def matriz_b_robot(cuad,epsilon):
    dic = {}
    estados = crea_estados(cuad)
    for e in estados:
        for o in observables_robot():
            dic[(e,o)]=calcula_probabilidad(e,o,epsilon,estados)
    return dic
    
class HMM(object):
    """Clase para definir un modelo oculto de Markov"""

    def __init__(self,estados,mat_ini,mat_trans,observables,mat_obs):
        """El constructor de la clase recibe una lista con los estados, otra
        lista con los observables, un diccionario representado la matriz de
        probabilidades de transición, otro diccionario con la matriz de
        probabilidades de observación, y otro con las probabilidades de
        inicio. Supondremos (no lo comprobamos) que las matrices son 
        coherentes respecto de la  lista de estados y de observables."""
        
        self.estados=estados
        self.observables=observables
        self.a={(si,sj):ptrans
                for (si,l) in zip(estados,mat_trans)
                for (sj,ptrans) in zip(estados,l)}
        self.b={(si,vj):pobs
                for (si,l) in zip(estados,mat_obs)
                for (vj,pobs) in zip(observables,l)}
        self.pi=dict(zip(estados,mat_ini))


ej1_hmm=HMM(["c","f"],
            [0.8,0.2],
            [[0.7,0.3],[0.4,0.6]],
            [1,2,3],   
            [[0.2,0.4,0.4],[0.5,0.4,0.1]])
            

ej2_hmm=HMM(["l","no l"],
            [0.5,0.5],
            [[0.7, 0.3], [0.3,0.7]],
            ["u","no u"],   
            [[0.9,0.1],[0.2,0.8]])

class Robot(HMM):
    def __init__(self, cuadricula, epsilon):
        self.cuadricula = cuadricula
        self.epsilon = epsilon
        self.estados = crea_estados(cuadricula)
        self.pi = matriz_pi_Robot(cuadricula)
        self.a = matriz_a_robot(cuadricula)
        self.observables = observables_robot()
        self.b = matriz_b_robot(cuadricula, epsilon)




# NOTAS: 

# - Representar los estados por pares de coordenadas, en el que la (0,0) sería
#   la casilla de arriba a la izquierda. 
# - Nótese que en total son 16 posibles observaciones. Las observaciones las 
#   representamos por una tupla (i1,i2,i3,i4), en el que  sus elementos son 
#   0 ó 1, donde 0 indica que no se ha detectado obstáculo, y 1, indica que sí, 
#   respectivamente en  el N,S, E y O (en ese orden). 
#   Por ejemplo (1,1,0,0) indica que se detecta obstáculo en el N y en el S.
#   y (0,0,1,0) indica que se detecta obstáculo solo en el E.  
# - Supondremos que NO hay casillas bloqueadas (es decir, sin vecinos).    



# Ejemplo de HMM generado para una cuadrícula básica:
    
cuadr0=["ooo",
        "oxE",
        "ooo"]

# >>> robot0=Robot(cuadr0,0.1)

# >>> robot0.estados
# [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

# >>> robot0.observables

#[(0, 0, 0, 0),(0, 0, 0, 1),(0, 0, 1, 0),(0, 0, 1, 1),(0, 1, 0, 0),
# (0, 1, 0, 1),(0, 1, 1, 0),(0, 1, 1, 1),(1, 0, 0, 0),(1, 0, 0, 1),
# (1, 0, 1, 0),(1, 0, 1, 1),(1, 1, 0, 0),(1, 1, 0, 1),(1, 1, 1, 0),
# (1, 1, 1, 1)]

# >>> robot0.pi 
# {(0, 0): 0.125, (0, 1): 0.125, (0, 2): 0.125, (1, 0): 0.125,
#  (1, 2): 0.125, (2, 0): 0.125, (2, 1): 0.125, (2, 2): 0.125}

# >>> robot0.a
 
#{((0, 0), (0, 0)): 0, ((0, 0), (0, 1)): 0.5, ((0, 0), (0, 2)): 0,
# ((0, 0), (1, 0)): 0.5,((0, 0), (1, 2)): 0, ((0, 0), (2, 0)): 0,
# ((0, 0), (2, 1)): 0, ((0, 0), (2, 2)): 0,
# ((0, 1), (0, 0)): 0.5, ((0, 1), (0, 1)): 0, ((0, 1), (0, 2)): 0.5,
# ((0, 1), (1, 0)): 0, ((0, 1), (1, 2)): 0, ((0, 1), (2, 0)): 0,
# ((0, 1), (2, 1)): 0, ((0, 1), (2, 2)): 0,
# ((0, 2), (0, 0)): 0, ((0, 2), (0, 1)): 0.5,
# ...
#  ((1, 2), (0, 0)): 0, ((1, 2), (0, 1)): 0.25, # (1,2) es la casilla especial
# ((1, 2), (0, 2)): 0.25, ((1, 2), (1, 0)): 0,
# ((1, 2), (1, 2)): 0, ((1, 2), (2, 0)): 0,
# ((1, 2), (2, 1)): 0.25, ((1, 2), (2, 2)): 0.25,
# ....
# ... Continúa .....

# >>> robot0.b
#{((0, 0), (0, 0, 0, 0)): 0.008100000000000001,
# ((0, 0), (0, 0, 0, 1)): 0.07290000000000002,
# ((0, 0), (0, 0, 1, 0)): 0.0009000000000000002,
# ((0, 0), (0, 0, 1, 1)): 0.008100000000000001,
#  ... Continúa ....



# -----------

# Ejemplo de uso de Viterbi en la cuadrícula del ejemplo




cuadr_ej=     ["ooooxoooooxoooxo",
                "xxooxoxxoxoxExxo",
                "xoEoxoxxoooooxxo",
                "ooxooExooooxoooo"]




robot_ej_hmm=Robot(cuadr_ej,0.15)

# Secuencia de 7 observaciones:
seq_ej1=[(1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1),
        (1, 1, 0, 0),(0, 1, 1, 0),(1, 1, 0, 0)]

# Usando Viterbi, estimamos las casillas por las que ha pasado:

viterbi(robot_ej_hmm,seq_ej1)
# [(3, 14), (3, 13), (3, 12), (3, 13), (3, 14), (3, 15), (3, 14)]


#=====================================================================
# Parte 3: Experimentaciones
#=====================================================================


# Realizar experimentos para ver cómo de buenas son las secuencias que se
# obtienen con el algoritmo de Viterbi que se ha implementado. Para ello, una
# manera podría ser la siguiente: generar una secuencia de estados y la
# correspondiente secuencia de observaciones usando el algoritmo de
# muestreo. La secuencia de observaciones obtenida se puede usar como entrada
# al algoritmo de Viterbi y comparar la secuencia obtenida con la secuencia de
# estados real que ha generado las observaciones. Se pide ejecutar con varios
# ejemplos y comprobar cómo de ajustados son los resultados obtenidos. Para
# medir el grado de coincidencia entre las dos secuencias de estados, calcular
# la proporción de estados coincidentes, respecto del total de estados de la
# secuencia.


# Por ejemplo:

# Función que calcula el porcentaje de coincidencias:
def compara_secuencias(seq1,seq2):
    return sum(x==y for x,y in zip(seq1,seq2))/len(seq1)


# Generamos una secuencia de 20 estados y observaciones
# >>> seq_e,seq_o=muestreo_hmm(robot_ej_hmm,20)

# >>> seq_o 
# [(0, 0, 1, 1), (0, 1, 1, 0), (1, 1, 0, 0),....]

# >>> seq_e
# [(2, 5),(3, 5), (3, 4), (3, 3), (3, 4), ....]
 
# >>> seq_estimada=viterbi(robot_ej_hmm,seq_o)

# >>> seq_estimada
# [(2, 5),(3, 5),(3, 4),(3, 3),(3, 4),(3, 5),...]
 
# Vemos, cuántas coincidencias hay, proporcinalmente al total de estados de la 
# secuencia:
    
# >>> compara_secuencias(seq_e,seq_estimada)
# 0.95

# -----------------------------------

# Para mecanizar esta experimentación, definir una función

#     experimento_hmm_robot_viterbi(cuadricula,epsilon,n,m) 

def experimento_hmm_robot_viterbi(cuadricula,epsilon,n,m):
    lista = []
    rob = Robot(cuadricula, epsilon)
    for _ in range(m):
        seq_e,seq_o=muestreo_hmm(rob,n)
        seq_estimada=viterbi(rob,seq_o)
        b = compara_secuencias(seq_e,seq_estimada)
        lista.append(b)
    return sum(lista)/len(lista)
        

# que genera el HMM correspondiente a la cuadrícula y al epsilon, y realiza 
# m experimentos, como se ha descrito:
    
# - generar en cada uno de ellos una secuencia de n observaciones y estados 
#  (con muestreo_hmm)
# - con la secuencia de observaciones, llamar a viterbi para estimar la 
#   secuencia de estados más probable
# - calcular qué proporción de coincidencias hay entre la secuencia de estados real 
#   y la que ha estimado viterbi 
# Y devuelve la media de los m experimentos. 

# Experimentar al menos con la cuadrícula del ejemplo y con varios valores de
# n, con varios valores de epsilon y con un m suficientemente grande para que 
# la media devuelta sea significativa del rendimiento del algoritmo. 