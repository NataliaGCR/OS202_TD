import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

def calculate_row(y):
    row = np.empty(width, dtype=np.double)
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        row[x] = mandelbrot_set.convergence(c, smooth=True)
    return row

'''
if __name__ == '__main__':
    times = []
    for num_processes in range(1, multiprocessing.cpu_count() + 1):
        deb = time()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(calculate_row, range(height))
        convergence = np.array(results)
        fin = time()
        execution_time = fin-deb
        times.append(execution_time)
        print(f"Temps du calcul de l'ensemble de Mandelbrot avec {num_processes} processus : {execution_time}")

    # Rassembler l'image
    #image = np.concatenate(results)
    
    # Rassembler l'image
    image = np.vstack(results)

    # Normaliser l'image
    image = (image / np.max(image)) * 255

    plt.imshow(image, cmap='hot', interpolation='nearest')
    plt.show()
    # Créer une image PIL et la sauvegarder
    Image.fromarray(image.astype(np.uint8)).save('mandelbrot.png')

    single_process_time = times[0]
    speedups = [single_process_time / time for time in times] 
    for num_processes, speedup in enumerate(speedups, start=1):
        print(f"Speedup avec {num_processes} processus : {speedup}")
    plt.plot(range(1, multiprocessing.cpu_count() + 1), speedups)
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Processes')
    plt.show() 
'''

    #Q1.3.2

def worker_process(args):
    queue, y = args
    row = calculate_row(y)
    queue.put((y, row))

if __name__ == '__main__':
    times = []
    for num_processes in range(1, cpu_count() + 1):
        deb = time()
        with Manager() as manager:
            queue = manager.Queue()
            with Pool(processes=num_processes) as pool:
                pool.map(worker_process, [(queue, y) for y in range(height)])
            results = [queue.get() for _ in range(height)]
            results.sort()  # ensure the results are sorted by y
            convergence = np.array([row for y, row in results])
        fin = time()
        execution_time = fin-deb
        times.append(execution_time)
        print(f"Temps du calcul de l'ensemble de Mandelbrot avec {num_processes} processus : {execution_time}")
        
    arrays = [row for y, row in results]
    image = np.vstack(arrays)
    # Normaliser l'image
    image = (image / np.max(image)) * 255
    plt.imshow(image, cmap='hot', interpolation='nearest')
    plt.show()

    single_process_time = times[0]
    speedups = [single_process_time / time for time in times]
    for num_processes, speedup in enumerate(speedups, start=1):
        print(f"Speedup avec {num_processes} processus : {speedup}")
    plt.plot(range(1, cpu_count() + 1), speedups)
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Processes')
    plt.show()

'''
    Temps du calcul de l'ensemble de Mandelbrot avec 1 processus : 5.381587266921997
    Temps du calcul de l'ensemble de Mandelbrot avec 2 processus : 3.9839632511138916
    Temps du calcul de l'ensemble de Mandelbrot avec 3 processus : 3.4963245391845703
    Temps du calcul de l'ensemble de Mandelbrot avec 4 processus : 3.544710159301758
    Temps du calcul de l'ensemble de Mandelbrot avec 5 processus : 3.7750158309936523
    Temps du calcul de l'ensemble de Mandelbrot avec 6 processus : 4.3469085693359375
    Temps du calcul de l'ensemble de Mandelbrot avec 7 processus : 4.210708379745483
    Temps du calcul de l'ensemble de Mandelbrot avec 8 processus : 4.486328125
    Speedup avec 1 processus : 1.0
    Speedup avec 2 processus : 1.3508124768513712
    Speedup avec 3 processus : 1.5392127380077585
    Speedup avec 4 processus : 1.518202342383353
    Speedup avec 5 processus : 1.4255800526021811
    Speedup avec 6 processus : 1.2380263309159327
    Speedup avec 7 processus : 1.2780717118308933
    Speedup avec 8 processus : 1.1995527560574935
    '''