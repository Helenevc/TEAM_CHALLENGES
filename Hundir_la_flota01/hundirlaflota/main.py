
from variables import stockbarcos, mensajeinicial, mensajebarco, mensajederrota, mensajefinal, mensajerrorbarco, mensajestart, mensajeturno, mensajevictoria, tableroCPU, tablerojp
from clases import Barco, Tablero, Acorazado, Destructor, Submarino, Portaaviones
import numpy as np
import random


def iniciarjuego():
    print(mensajeinicial)
    nombrejp = input("Nombre de jugador:")
    tableroobj=Tablero()

    tablerojp = tableroobj.creatablero(10)
    print(tablerojp)
    print("#"*40)
    tableroCPU= tableroobj.creatablero(10)
    tableromuestra= tableroCPU.copy()
    print(tableroCPU)

    print(f"Bienvenido {nombrejp} vamos a colocar tus barcos")
    print("#" *40)

    print(f"Tenemos 4 Destructores")
    print("#"*40)

    tableroobj.allbarcos( stockbarcos,tablerojp)
    tableroobj.allbarcos( stockbarcos,tableromuestra)

    print(tablerojp)
    print("#"*40)
    print(tableroCPU)
   

    
iniciarjuego()
