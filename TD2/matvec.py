import numpy as np
from multiprocessing import Pool, cpu_count

# Dimension of the problem
dim = 120

# Initialization of the array
A = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(dim)])

# Initialization of vector u
u = np.array([i + 1. for i in range(dim)])

# Number of tasks
num_tasks = cpu_count()

# Number of columns per task
Nloc = dim // num_tasks

# Function to calculate the array-vector product for a subset of columns
'''
#Q 1.4.1
def matvec_subset(cols):
    return A[:, cols].dot(u[cols])
'''
#Q 1.4.2
def matvec_subset(rows):
    return A[rows, :].dot(u)


def main():
    # Create a process pool
    with Pool(num_tasks) as p:
        # Divide the array into subsets of columns
        subsets = [range(i * Nloc, (i + 1) * Nloc) for i in range(num_tasks)]
        # Apply the function to each subset of columns
        results = p.map(matvec_subset, subsets)


    #v = np.sum(results, axis=0)
        
    v = np.concatenate(results) 

    print(f"A = {A}")
    print(f"u = {u}")
    print(f"v = {v}")

if __name__ == '__main__':
    main()

#Q 1.4.1
'''
A = [[  1.   2.   3. ... 118. 119. 120.]
 [  2.   3.   4. ... 119. 120.   1.]
 [  3.   4.   5. ... 120.   1.   2.]
 ...
 [118. 119. 120. ... 115. 116. 117.]
 [119. 120.   1. ... 116. 117. 118.]
 [120.   1.   2. ... 117. 118. 119.]]

u = [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.
  15.  16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.  28.
  29.  30.  31.  32.  33.  34.  35.  36.  37.  38.  39.  40.  41.  42.
  43.  44.  45.  46.  47.  48.  49.  50.  51.  52.  53.  54.  55.  56.
  57.  58.  59.  60.  61.  62.  63.  64.  65.  66.  67.  68.  69.  70.
  71.  72.  73.  74.  75.  76.  77.  78.  79.  80.  81.  82.  83.  84.
  85.  86.  87.  88.  89.  90.  91.  92.  93.  94.  95.  96.  97.  98.
  99. 100. 101. 102. 103. 104. 105. 106. 107. 108. 109. 110. 111. 112.
 113. 114. 115. 116. 117. 118. 119. 120.]

v = [583220. 576080. 569060. 562160. 555380. 548720. 542180. 535760. 529460.
 523280. 517220. 511280. 505460. 499760. 494180. 488720. 483380. 478160.
 473060. 468080. 463220. 458480. 453860. 449360. 444980. 440720. 436580.
 432560. 428660. 424880. 421220. 417680. 414260. 410960. 407780. 404720.
 401780. 398960. 396260. 393680. 391220. 388880. 386660. 384560. 382580.
 380720. 378980. 377360. 375860. 374480. 373220. 372080. 371060. 370160.
 369380. 368720. 368180. 367760. 367460. 367280. 367220. 367280. 367460.
 367760. 368180. 368720. 369380. 370160. 371060. 372080. 373220. 374480.
 375860. 377360. 378980. 380720. 382580. 384560. 386660. 388880. 391220.
 393680. 396260. 398960. 401780. 404720. 407780. 410960. 414260. 417680.
 421220. 424880. 428660. 432560. 436580. 440720. 444980. 449360. 453860.
 458480. 463220. 468080. 473060. 478160. 483380. 488720. 494180. 499760.
 505460. 511280. 517220. 523280. 529460. 535760. 542180. 548720. 555380.
 562160. 569060. 576080.]
 '''

#Q 1.4.2
'''
np.concatenate(results) is joining the results of the matrix-vector multiplication from each subset of
 rows (or columns) that were processed in parallel.

v = [583220. 576080. 569060. 562160. 555380. 548720. 542180. 535760. 529460.
 523280. 517220. 511280. 505460. 499760. 494180. 488720. 483380. 478160.
 473060. 468080. 463220. 458480. 453860. 449360. 444980. 440720. 436580.
 432560. 428660. 424880. 421220. 417680. 414260. 410960. 407780. 404720.
 401780. 398960. 396260. 393680. 391220. 388880. 386660. 384560. 382580.
 380720. 378980. 377360. 375860. 374480. 373220. 372080. 371060. 370160.
 369380. 368720. 368180. 367760. 367460. 367280. 367220. 367280. 367460.
 367760. 368180. 368720. 369380. 370160. 371060. 372080. 373220. 374480.
 375860. 377360. 378980. 380720. 382580. 384560. 386660. 388880. 391220.
 393680. 396260. 398960. 401780. 404720. 407780. 410960. 414260. 417680.
 421220. 424880. 428660. 432560. 436580. 440720. 444980. 449360. 453860.
 458480. 463220. 468080. 473060. 478160. 483380. 488720. 494180. 499760.
 505460. 511280. 517220. 523280. 529460. 535760. 542180. 548720. 555380.
 562160. 569060. 576080.]
'''