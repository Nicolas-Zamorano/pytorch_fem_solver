import numpy as np
import matplotlib.pyplot as plt


def Triangulation(N):

    # Coordenadas de nodos principales y desplazados (para visualización)
    coords = np.linspace(0, 1, N)
    
    X, Y = np.meshgrid(coords, coords)
    
    # Lista de nodos: comenzamos con los nodos de la malla
    nodes = [tuple(p) for p in np.column_stack((X.ravel(), Y.ravel()))]
    
    # Diccionario para acceder rápidamente a índices
    node_to_index = {pt: i for i, pt in enumerate(nodes)}
    
    triangles = []
    
    for i in range(len(coords)-1):
        for j in range(len(coords)-1):
            # Definimos los vértices del cuadrado
            p0 = (coords[i], coords[j])
            p1 = (coords[i+1], coords[j])
            p2 = (coords[i+1], coords[j+1])
            p3 = (coords[i], coords[j+1])
            
            # Coordenada del centro
            pc = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)
            
            # Agregamos el centro si no está en los nodos
            if pc not in node_to_index:
                node_to_index[pc] = len(nodes)
                nodes.append(pc)
            
            # Índices
            i0 = node_to_index[p0]
            i1 = node_to_index[p1]
            i2 = node_to_index[p2]
            i3 = node_to_index[p3]
            ic = node_to_index[pc]
            
            # Dividimos en 4 triángulos
            triangles.append([i0, i1, ic])
            triangles.append([i1, i2, ic])
            triangles.append([i2, i3, ic])
            triangles.append([i3, i0, ic])
    
    return nodes, triangles
    