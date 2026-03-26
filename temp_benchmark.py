import time
import cv2
import numpy as np

board = np.random.randint(0, 2, (8, 8), dtype=np.uint8)

starts = time.time()
for _ in range(10000):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(board, connectivity=4)
    a = stats[1:, cv2.CC_STAT_AREA]
    
    ret2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(1 - board, connectivity=4)
    b = stats2[1:, cv2.CC_STAT_AREA]
ends = time.time()
print(f"CV2 CC time: {(ends - starts) * 1000} ms for 10000 calls")

starts = time.time()
figure = np.array([[1,1],[1,1]], dtype=np.int8)
for _ in range(10000):
    windows = np.lib.stride_tricks.sliding_window_view(board, figure.shape)
    valid = ~(windows & figure).any(axis=(2, 3))
    np.count_nonzero(valid)
ends = time.time()
print(f"Sliding window time: {(ends - starts) * 1000} ms for 10000 calls")
