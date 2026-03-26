import time
import numpy as np

u64 = np.uint64(4324234234)
starts = time.time()
for _ in range(10000):
    np.unpackbits(np.array([u64], dtype=np.uint64).view(np.uint8), bitorder='little').reshape((8, 8))
ends = time.time()
print(f"unpackbits time: {(ends - starts) * 1000} ms for 10000 calls")
