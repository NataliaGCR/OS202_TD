from mpi4py import MPI
import numpy as np

#C:\Users\natyo>mpiexec -n 8 python "C:\Users\natyo\OneDrive - Universidad EIA\Escritorio\NATALIA G\ENSTA\OS202\Promotion2024\TravauxDiriges\TD_numero_3\bucket_sort.py"

def bucket_sort():
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank() # numéro du processus
    size = comm.Get_size() # nombre de processus

    # Chaque processus génère son propre tableau de données
    data = np.random.rand(10)
    print(f"Process {rank} [{rank/8} - {(rank+1)/8}], initial data: {np.round(data, 3)}")

    data_copy = data.copy()


    # Chaque processus vérifie si chaque nombre est dans son intervalle assigné
    for i in range(len(data_copy)):
        if data_copy[i] < rank/8 or data_copy[i] >= (rank+1)/8:
            # Si le nombre n'est pas dans l'intervalle, il est envoyé au processus correct
            target_rank = int(data_copy[i]*8)
            comm.send(data_copy[i], dest=target_rank)
            data = np.delete(data, np.where(data == data_copy[i]))


    # Chaque processus reçoit les nombres des autres processus
    while comm.Iprobe(source=MPI.ANY_SOURCE):
        received_data = comm.recv(source=MPI.ANY_SOURCE)
        data = np.append(data, received_data)

    # Chaque processus trie son tableau
    print(f"Process {rank} separate data: {np.round(data, 3)}")
    quickSort(data, 0, len(data) - 1)
    print(f"Process {rank} sorted data: {np.round(data, 3)}")
    print()

    # Gather all sorted sub-arrays to rank 0
    gathered_data = comm.gather(data, root=0)

    # If this is rank 0, concatenate all sub-arrays to create the final sorted array
    if rank == 0:
        sorted_data = np.concatenate(gathered_data)
        print(f"Final sorted data: {np.round(sorted_data, 3)}")

def partition(array, low, high):
    pivot = array[high]
    i = (low - 1)

    for j in range(low, high):
        if array[j] <= pivot:
            i = i + 1
            array[i], array[j] = array[j], array[i]

    array[i + 1], array[high] = array[high], array[i + 1]

    return (i + 1)

def quickSort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
        quickSort(array, low, pi - 1)
        quickSort(array, pi + 1, high)

if __name__ == "__main__":
    bucket_sort()



#Solución
'''
Process 3 [0.375 - 0.5], initial data: [0.559 0.923 0.462 0.152 0.134 0.863 0.637 0.823 0.615 0.043]
Process 3 separate data: [0.462 0.401]
Process 3 sorted data: [0.401 0.462]

Process 2 [0.25 - 0.375], initial data: [0.401 0.951 0.599 0.082 0.157 0.707 0.265 0.286 0.124 0.139]
Process 2 separate data: [0.265 0.286 0.285]
Process 2 sorted data: [0.265 0.285 0.286]

Process 5 [0.625 - 0.75], initial data: [0.269 0.494 0.71  0.1   0.926 0.608 0.284 0.274 0.656 0.924]
Process 5 separate data: [0.71  0.656 0.637 0.639 0.707]
Process 5 sorted data: [0.637 0.639 0.656 0.707 0.71 ]

Process 7 [0.875 - 1.0], initial data: [0.352 0.839 0.014 0.98  0.712 0.012 0.569 0.232 0.018 0.503]
Process 7 separate data: [0.98  0.923 0.951 0.903 0.926 0.924 0.887 0.957 0.91  0.973 0.924 0.975]
Process 7 sorted data: [0.887 0.903 0.91  0.923 0.924 0.924 0.926 0.951 0.957 0.973 0.975 0.98 ]

Process 6 [0.75 - 0.875], initial data: [0.521 0.144 0.099 0.868 0.797 0.11  0.285 0.903 0.639 0.106]
Process 6 separate data: [0.868 0.797 0.863 0.823]
Process 6 sorted data: [0.797 0.823 0.863 0.868]

Process 4 [0.5 - 0.625], initial data: [0.815 0.957 0.281 0.679 0.235 0.19  0.828 0.364 0.754 0.183]
Process 4 separate data: [0.521 0.559 0.615 0.599 0.608 0.58  0.619]
Process 4 sorted data: [0.521 0.559 0.58  0.599 0.608 0.615 0.619]

Process 1 [0.125 - 0.25], initial data: [0.91  0.868 0.973 0.716 0.582 0.365 0.455 0.924 0.975 0.824]
Process 1 separate data: [0.144 0.152 0.134 0.157 0.139 0.201 0.249 0.235 0.19  0.183]
Process 1 sorted data: [0.134 0.139 0.144 0.152 0.157 0.183 0.19  0.201 0.235 0.249]

Process 0 [0.0 - 0.125], initial data: [0.787 0.694 0.58  0.887 0.756 0.201 0.327 0.37  0.249 0.619]
Process 0 separate data: [0.099 0.11  0.106 0.082 0.124 0.043 0.1  ]
Process 0 sorted data: [0.043 0.082 0.099 0.1   0.106 0.11  0.124]

Final sorted data: [0.043 0.082 0.099 0.1   0.106 0.11  0.124 0.134 0.139 0.144 0.152 0.157
 0.183 0.19  0.201 0.235 0.249 0.265 0.285 0.286 0.401 0.462 0.521 0.559
 0.58  0.599 0.608 0.615 0.619 0.637 0.639 0.656 0.707 0.71  0.797 0.823
 0.863 0.868 0.887 0.903 0.91  0.923 0.924 0.924 0.926 0.951 0.957 0.973
 0.975 0.98 ]
'''