""" 
Vamos a establecer las diferentes clases
"""
import numpy as np
import random
from variables import stockbarcos
class Barco:

    orientación= ["N","S","E","O"]

    def __init__(self, nombre, tamaño, cantidad):
        self.nombre = nombre
        self.tamaño = tamaño
        self.cantidad = cantidad
        self.impactos = 0
        self.coordenadas=[]
        self.posicion = self.posicioninicial()
        
# tenemos los siguientes barcos
# 4 barcos de 1 posición de eslora = Destructor
# 3 barcos de 2 posiciones de eslora = Submarino
# 2 barcos de 3 posiciones de eslora =Acorazado
# 1 barco de 4 posiciones de eslora = Portaaviones
  
   
    def posicioninicial(self):
        
        for i in range(self.cantidad):
            fila = np.random.randint(0, 10)
            columna = np.random.randint(0, 10)
            coordenada = (fila, columna)
            if coordenada not in self.coordenadas:  # Asegura que no se repitan coordenadas
                self.coordenadas.append(coordenada)
        return self.coordenadas
    
    def coloca_barco(self, tablero):
        nummaxfilas = tablero.shape[0]
        nummaxcolum = tablero.shape[1]
        tablerotemp = tablero.copy()
        for x,y in self.posicion:
            fila = x
            columna = y
            if fila<0 or fila >= nummaxfilas:
                print(f"No puedo poner la pieza{self}, se sale del tablero")
                return False
            if columna<0 or columna >= nummaxcolum:
                print(f"No puedo poner la pieza{self} porque sale del tablero")
                return False
            if tablero[x,y] == "0" or tablero[self]=="X":
                print(f"Hay otro barco en este sitio")
                return False
            tablerotemp[x,y]="O"
        return tablerotemp
    
    def colocabarco(self, tablero):
        orientación= ["N","S","E","O"]
        fila, columna = self.posicion[0]

        newori=np.random.choice(orientación)

        if newori == "N" and fila - (self.cantidad - 1) >= 0:
            for i in range(self.cantidad):
                if tablero[fila - i, columna] != " " :
                    return "Espacio ocupado, intenta de nuevo."
            for i in range(self.cantidad):
                tablero[fila - i, columna] = "O"

        elif newori == "S" and fila + (self.cantidad - 1) < 10:
            for i in range(self.cantidad):
                if tablero[fila + i, columna] != " ":
                    return "Espacio ocupado, intenta de nuevo."
            for i in range(self.cantidad):
                tablero[fila + i, columna] = "O"

        elif newori == "E" and columna + (self.cantidad - 1) < 10:
            for i in range(self.cantidad):
                if tablero[fila, columna + i] != " ":
                    return "Espacio ocupado, intenta de nuevo."
            for i in range(self.cantidad):
                tablero[fila, columna + i] = "O"

        elif newori == "O" and columna - (self.cantidad - 1) >= 0:
            for i in range(self.cantidad):
                if tablero[fila, columna - i] != " ":
                    return "Espacio ocupado, intenta de nuevo."
            for i in range(self.cantidad):
                tablero[fila, columna - i] = "O"
        else:
            print(f"La CPU ha colocado sus barcos")
        return tablero
    
   

Destructor = Barco("Destructor", 1, 4)
Submarino = Barco("Submarino", 2, 3)
Acorazado = Barco("Acorazado", 3, 2)
Portaaviones = Barco("Portaaviones", 4, 1)      


class Tablero:

    def creatablero (self,lado=10):
        return np.full((lado, lado)," " )
    
    def allbarcos(self,stockbarcos,tablero):
        for nombre, cantidad in stockbarcos.items():
            for i in range(cantidad):
                barco= Barco(nombre, stockbarcos[nombre], cantidad)
                tablerotemp= barco.colocabarco(tablero)
                i-=1
        return tablerotemp
    
    def disparar(tablero):
        (x,y) = input ("¿Donde quieres disparar? Recuerda que tienes que facilitarnos dos coordenadas, la 'x'  y la 'y'.")
        if tablero[x][y]=="O":
            tablero[x][y]= "X"
        elif tablero[x][y]=="X":
            print(f"Nooooo, aquí ya habías disparado")
        else:
            tablero[x][y]="-"
            print("Has disparado al agua")

    def comprobartablero(tablero):
       
        if tablero!="O":
            return f"¡¡¡Has ganado!!!"
        
   
    

