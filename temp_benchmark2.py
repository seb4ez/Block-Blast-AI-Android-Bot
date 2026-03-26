import time
import numpy as np

masks = np.random.randint(0, 2**64-1, 49, dtype=np.uint64)
board = np.uint64(124124124)

starts = time.time()
for _ in range(10000):
    valid = (board & masks) == 0
    count = np.count_nonzero(valid)
ends = time.time()
print(f"Uint64 time: {(ends - starts) * 1000} ms for 10000 calls")
