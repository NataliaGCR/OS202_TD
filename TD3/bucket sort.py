from mpi4py import MPI
import numpy as np

def bucket_sort():
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank() # numéro du processus
    size = comm.Get_size() # nombre de processus

    # Chaque processus génère son propre tableau de données
    data = np.random.rand(10)
    print(f"Process {rank} initial data: {data}")

    # Chaque processus vérifie si chaque nombre est dans son intervalle assigné
    for i in range(len(data)):
        if data[i] < rank/8 or data[i] >= (rank+1)/8:
            # Si le nombre n'est pas dans l'intervalle, il est envoyé au processus correct
            target_rank = int(data[i]*8)
            comm.send(data[i], dest=target_rank)
            data = np.delete(data, i)

    # Chaque processus reçoit les nombres des autres processus
    while comm.Iprobe(source=MPI.ANY_SOURCE):
        received_data = comm.recv(source=MPI.ANY_SOURCE)
        data = np.append(data, received_data)

    # Chaque processus trie son tableau
    quickSort(data, 0, len(data) - 1)
    print(f"Process {rank} sorted data: {data}")

    # Gather all sorted sub-arrays to rank 0
    gathered_data = comm.gather(data, root=0)

    # If this is rank 0, concatenate all sub-arrays to create the final sorted array
    if rank == 0:
        sorted_data = np.concatenate(gathered_data)
        print(f"Final sorted data: {sorted_data}")

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