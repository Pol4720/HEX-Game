# HEX Game

## Requisitos

Instalar las dependencias con:

```
pip install -r requirements.txt
```

## Ejecución

Ejecutar el archivo principal:

```
python main.py
```

## Controles

- Seleccionar modalidad de juego en el menú inicial.
- Hacer clic en las celdas para realizar movimientos.

# Teoría y Demostración de la Heurística

## Heurística

La heurística utilizada es la distancia Manhattan: `h(node, goal) = |x1 - x2| + |y1 - y2|`.

### Admisibilidad

La heurística es admisible porque nunca sobreestima el costo real para llegar al objetivo. En un tablero Hex, la distancia Manhattan es siempre menor o igual al costo real.

### Consistencia

La heurística es consistente porque cumple con la desigualdad triangular: `h(node, goal) <= h(node, neighbor) + cost(neighbor, goal)`.

Por lo tanto, la heurística garantiza que el algoritmo A* encuentre el camino óptimo.
