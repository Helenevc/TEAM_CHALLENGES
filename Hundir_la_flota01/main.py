from clases import Tablero
from variables import stockbarcos, mensajeinicial, mensajefinal, mensajevictoria, mensajederrota, mensajeturno, disparoacertado, disparofallido
import numpy as np
import random
def pedir_coordenadas():
    """
    Pide al usuario que introduzca coordenadas válidas.
    """
    while True:
        try:
            x = int(input("Introduce la fila (0-9): "))
            y = int(input("Introduce la columna (0-9): "))
            if 0 <= x <= 9 and 0 <= y <= 9:
                return x, y
            else:
                print("Coordenadas fuera de rango. Intenta de nuevo.")
        except ValueError:
            print("Entrada no válida. Debes introducir números del 0 al 9.")


def disparo_cpu(tablero_jugador):
    """
    La CPU lanza un disparo aleatorio al tablero del jugador.
    """
    while True:
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        resultado = tablero_jugador.disparar(x, y)
        if resultado == "impacto":
            print(f"La CPU ha impactado en ({x}, {y})")
            return
        elif resultado == "agua":
            print(f"La CPU ha fallado en ({x}, {y})")
            return
        # Si es repetido, repite intento


def mostrar_tablero_cpu_visible(tablero_cpu):
    """
    Muestra el tablero de la CPU solo con los disparos recibidos (X y -).
    """
    tablero_visible = np.full((10, 10), " ")
    for i in range(10):
        for j in range(10):
            if tablero_cpu.tablero[i, j] == "X" or tablero_cpu.tablero[i, j] == "-":
                tablero_visible[i, j] = tablero_cpu.tablero[i, j]
    print("Tablero de la CPU (solo impactos visibles):")
    print(tablero_visible)


def iniciar_juego():
    print(mensajeinicial)
    nombre = input("Introduce tu nombre, jugador: ")
    print(f"\n¡Bienvenido, {nombre}!\n")

    # Crear tableros
    tablero_jugador = Tablero()
    tablero_cpu = Tablero()

    # Colocar barcos
    tablero_jugador.colocar_todos_los_barcos(stockbarcos)
    tablero_cpu.colocar_todos_los_barcos(stockbarcos)

    # Bucle principal del juego
    while True:
        print("\nTu tablero:")
        tablero_jugador.mostrar()
        print("\nTablero enemigo (solo impactos visibles):")
        mostrar_tablero_cpu_visible(tablero_cpu)

        print(f"\n{mensajeturno}")
        x, y = pedir_coordenadas()
        resultado = tablero_cpu.disparar(x, y)

        if resultado == "impacto":
            print(disparoacertado)
        elif resultado == "agua":
            print(disparofallido)
        elif resultado == "repetido":
            print("Ya has disparado ahí. Pierdes turno.")

        # Comprobar si el jugador ha ganado
        if tablero_cpu.comprobar_victoria():
            print(mensajevictoria)
            break

        # Turno de la CPU
        print("\nTurno de la CPU:")
        disparo_cpu(tablero_jugador)

        if tablero_jugador.comprobar_victoria():
            print(mensajederrota)
            break

    print(mensajefinal)


if __name__ == "__main__":
    iniciar_juego()