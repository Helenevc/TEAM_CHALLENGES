import numpy as np
import random
# Las variables que creamos para nuestra versión de Hundir la flota son:
# Los mensajes de texto del juego

mensajeinicial = "Bienvenido a Hundir la Flota 2.0"
mensajefinal = "Hasta pronto Grumete"

mensajestart = "Comenzar la partida"

mensajevictoria = "Winner"
mensajederrota = "Tus barcos han sido hundidos"

mensajebarco = "Tu barco ha sido colocado con éxito"
mensajerrorbarco = "Tu barco no cabe en el tablero, prueba otra vez"



mensajeturno = "Tu turno, elige coordenadas de ataque"
disparoacertado = "Has acertado, sigue otro turno"
disparofallido = "Has fallado"

tablerojp = np.full((10,10)," ")
tableroCPU = np.full((10,10), " ")

stockbarcos = {" Destructores":4,"Submarino":3,"Acorazado":2,"Portaviones":1}
