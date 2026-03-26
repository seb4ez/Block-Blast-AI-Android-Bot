import numpy as np
import random
import time
import cv2

class BlockBlastSim:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.shapes = self._init_shapes()
        self.current_piece_indices = []
        self._init_specific_shapes()
        self._precompute_masks()
        self._precompute_rough_edges_lut()
        
    def _init_shapes(self):
        return [
            np.array([[1]], dtype=np.int8), # 1x1
            np.array([[1, 1]], dtype=np.int8), # 1x2
            np.array([[1], [1]], dtype=np.int8), # 2x1
            np.array([[1, 1, 1]], dtype=np.int8), # 1x3
            np.array([[1], [1], [1]], dtype=np.int8), # 3x1
            np.array([[1, 1, 1, 1]], dtype=np.int8), # 1x4
            np.array([[1], [1], [1], [1]], dtype=np.int8), # 4x1
            np.array([[1, 1, 1, 1, 1]], dtype=np.int8), # 1x5
            np.array([[1], [1], [1], [1], [1]], dtype=np.int8), # 5x1
            np.array([[1, 1], [1, 1]], dtype=np.int8), # 2x2
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int8), # 3x3
            # Small L-shapes
            np.array([[1, 0], [1, 1]], dtype=np.int8),
            np.array([[0, 1], [1, 1]], dtype=np.int8),
            np.array([[1, 1], [1, 0]], dtype=np.int8),
            np.array([[1, 1], [0, 1]], dtype=np.int8),
            # Large L-shapes
            np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int8), # BL
            np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=np.int8), # BR
            np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.int8), # TL
            np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=np.int8)  # TR
        ]
        
    def _init_specific_shapes(self):
        def _get_idx(s):
            for i, p in enumerate(self.shapes):
                if np.array_equal(p, s):
                    return i
            return -1
            
        self.shape_BR = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=np.int8)
        self.shape_BL = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int8)
        self.shape_TR = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=np.int8)
        self.shape_TL = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.int8)
        
        self.heuristic_indices = {
            '2x2': _get_idx(np.array([[1, 1], [1, 1]], dtype=np.int8)),
            '3x3': _get_idx(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int8)),
            '5x1': _get_idx(np.array([[1], [1], [1], [1], [1]], dtype=np.int8)),
            '4x1': _get_idx(np.array([[1], [1], [1], [1]], dtype=np.int8)),
            '3x1': _get_idx(np.array([[1], [1], [1]], dtype=np.int8)),
            '1x5': _get_idx(np.array([[1, 1, 1, 1, 1]], dtype=np.int8)),
            '1x4': _get_idx(np.array([[1, 1, 1, 1]], dtype=np.int8)),
            '1x3': _get_idx(np.array([[1, 1, 1]], dtype=np.int8)),
            'BR': _get_idx(self.shape_BR),
            'BL': _get_idx(self.shape_BL),
            'TR': _get_idx(self.shape_TR),
            'TL': _get_idx(self.shape_TL),
        }
        
    def _precompute_masks(self):
        self.shape_masks = []
        self.shape_fits_rc = []
        self.piece_xy_to_mask = {}
        for idx, shape in enumerate(self.shapes):
            h, w = shape.shape
            masks = []
            rcs = []
            for r in range(8 - h + 1):
                for c in range(8 - w + 1):
                    mask = np.uint64(0)
                    for i in range(h):
                        for j in range(w):
                            if shape[i, j] == 1:
                                bit = (r + i) * 8 + (c + j)
                                mask |= (np.uint64(1) << np.uint64(bit))
                    masks.append(mask)
                    rcs.append((r, c))
                    self.piece_xy_to_mask[(idx, r, c)] = mask
            self.shape_masks.append(np.array(masks, dtype=np.uint64))
            self.shape_fits_rc.append(rcs)
            
        self.ROW_MASKS = [np.uint64(0xFF) << np.uint64(r * 8) for r in range(8)]
        self.COL_MASKS = [np.uint64(0x0101010101010101) << np.uint64(c) for c in range(8)]

    def _precompute_rough_edges_lut(self):
        self.rough_edges_lut = np.zeros(512, dtype=np.float64)
        for i in range(512):
            area = np.unpackbits(np.array([i], dtype=np.uint16).view(np.uint8), bitorder='little')[:9].reshape((3, 3))
            if (area == self.shape_BR).all() or (area == self.shape_BL).all() or \
               (area == self.shape_TR).all() or (area == self.shape_TL).all():
                self.rough_edges_lut[i] = 0
            else:
                rows, cols = np.nonzero(area)
                if len(rows) == 0:
                    self.rough_edges_lut[i] = 0
                else:
                    self.rough_edges_lut[i] = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1) - len(rows)

    def get_new_pieces(self):
        self.current_piece_indices = random.choices(range(len(self.shapes)), k=3)
        return [self.shapes[i] for i in self.current_piece_indices]

    def clone(self):
        """Ultra-fast deep copy for MCTS node expansion."""
        new_sim = BlockBlastSim.__new__(BlockBlastSim)
        new_sim.shapes = self.shapes
        new_sim.shape_BR = self.shape_BR
        new_sim.shape_BL = self.shape_BL
        new_sim.shape_TR = self.shape_TR
        new_sim.shape_TL = self.shape_TL
        new_sim.heuristic_indices = self.heuristic_indices
        new_sim.shape_masks = self.shape_masks
        new_sim.shape_fits_rc = self.shape_fits_rc
        new_sim.piece_xy_to_mask = self.piece_xy_to_mask
        new_sim.ROW_MASKS = self.ROW_MASKS
        new_sim.COL_MASKS = self.COL_MASKS
        new_sim.rough_edges_lut = self.rough_edges_lut
        
        new_sim.board = self.board.copy()
        new_sim.current_piece_indices = self.current_piece_indices.copy()
        return new_sim
        
    def calculate_risk(self, board_u64):
        """Verifica si las piezas asesinas caben."""
        risk = 0
        if self.fast_waysToFit('3x3', board_u64) == 0: risk += 1
        if self.fast_waysToFit('5x1', board_u64) == 0: risk += 1
        if self.fast_waysToFit('1x5', board_u64) == 0: risk += 1
        return risk

    def waysToFit(self, figure, board, optimized=False):
        windows = np.lib.stride_tricks.sliding_window_view(board, figure.shape)
        valid = ~(windows & figure).any(axis=(2, 3))
        if optimized:
            return np.count_nonzero(valid)
        else:
            return list(zip(*np.nonzero(valid)))

    def waysToFitList(self, piece_idx, board_u64):
        masks = self.shape_masks[piece_idx]
        valid_indices = np.nonzero((board_u64 & masks) == 0)[0]
        rcs = self.shape_fits_rc[piece_idx]
        return [rcs[i] for i in valid_indices]

    def fast_waysToFit(self, name, board_u64):
        idx = self.heuristic_indices[name]
        masks = self.shape_masks[idx]
        return np.count_nonzero((board_u64 & masks) == 0)

    def is_game_over(self, board_u64, remaining_piece_indices):
        is_over = True
        for idx in remaining_piece_indices:
            masks = self.shape_masks[idx]
            if np.count_nonzero((board_u64 & masks) == 0) > 0:
                is_over = False
                break
        
        risk_score = self.calculate_risk(board_u64)
        return is_over, risk_score

    def count_component_sizes_cv2(self, board_int8, val):
        mask = 1 - board_int8 if val == 0 else board_int8
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
        return stats[1:, cv2.CC_STAT_AREA].tolist()

    def roughEdgesScore(self, board_int8):
        windows = np.lib.stride_tricks.as_strided(
            board_int8, shape=(6, 6, 3, 3), strides=board_int8.strides * 2
        )
        powers = np.array([[1, 2, 4], [8, 16, 32], [64, 128, 256]], dtype=np.int32)
        hashes = (windows * powers).sum(axis=(2, 3))
        return self.rough_edges_lut[hashes].sum()
        
    def get_state_score(self, board_int8, board_u64, cleared_lines):
        num2x2Fit = self.fast_waysToFit('2x2', board_u64)
        num3x3Fit = self.fast_waysToFit('3x3', board_u64)
        num5x1Fit = self.fast_waysToFit('5x1', board_u64)
        num4x1Fit = self.fast_waysToFit('4x1', board_u64)
        num3x1Fit = self.fast_waysToFit('3x1', board_u64)

        num1x5Fit = self.fast_waysToFit('1x5', board_u64)
        num1x4Fit = self.fast_waysToFit('1x4', board_u64)
        num1x3Fit = self.fast_waysToFit('1x3', board_u64)

        numBRFit = self.fast_waysToFit('BR', board_u64)
        numBLFit = self.fast_waysToFit('BL', board_u64)
        numTRFit = self.fast_waysToFit('TR', board_u64)
        numTLFit = self.fast_waysToFit('TL', board_u64)
        
        holes_0 = self.count_component_sizes_cv2(board_int8, 0)
        holes_1 = self.count_component_sizes_cv2(board_int8, 1)
        
        filled_count = np.count_nonzero(board_int8 == 1)
        
        score = (numBRFit*10 + numBLFit*10 + numTRFit*10 + numTLFit*10 + 
                 num2x2Fit*5 + num3x3Fit*20 + num5x1Fit*2 + num1x5Fit*2 + 
                 num4x1Fit*0.8 + num3x1Fit*0.5 + num1x4Fit*0.8 + num1x3Fit*0.5)
                 
        score += cleared_lines * 30
        
        score -= len(holes_0) * 5
        score -= filled_count * 0.5
        score -= len(holes_1) * 10
        
        banned_hole_sizes = {1, 2, 3}
        score -= sum(1 for i in holes_1 if i in banned_hole_sizes) * 20
        score -= sum(1 for i in holes_0 if i in banned_hole_sizes) * 20
        
        score -= self.roughEdgesScore(board_int8) * 0.5
        
        # Soft Game Over / Risk Metric
        max_empty_block = max(holes_0) if holes_0 else 0
        high_congestion = (max_empty_block < 9)
        risk_score = self.calculate_risk(board_u64)
        
        # Reward Shaping
        if high_congestion:
            score -= 500  # Penalización por área contigua ahogada
            
        if risk_score == 1:
            score -= 100
        elif risk_score == 2:
            score -= 400
        elif risk_score == 3:
            score -= 1000  # Casi letal matemáticamente
        
        return float(score), high_congestion

    def updateBoard(self, board, figure, x, y):
        # Fallback method if external logic passes weird arrays
        new_board = board.copy()
        h, w = figure.shape
        new_board[x:x+h, y:y+w] |= figure
        
        row_sums = new_board.sum(axis=1)
        col_sums = new_board.sum(axis=0)
        
        rows_to_clear = np.nonzero(row_sums == 8)[0]
        cols_to_clear = np.nonzero(col_sums == 8)[0]
        
        cleared = len(rows_to_clear) + len(cols_to_clear)
        if cleared > 0:
            new_board[rows_to_clear, :] = 0
            new_board[:, cols_to_clear] = 0
            
        return new_board, cleared

    def reset(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.current_piece_indices = []
        self.get_new_pieces()
        return self.board.copy(), [self.shapes[i] for i in self.current_piece_indices]

    def step(self, pieza_idx_or_array, x, y):
        list_idx = -1
        piece_idx = -1
        if isinstance(pieza_idx_or_array, int) or isinstance(pieza_idx_or_array, np.integer):
            list_idx = int(pieza_idx_or_array)
            piece_idx = self.current_piece_indices[list_idx]
            piece = self.shapes[piece_idx]
        else:
            piece = pieza_idx_or_array
            for i, p_idx in enumerate(self.current_piece_indices):
                if np.array_equal(self.shapes[p_idx], piece):
                    list_idx = i
                    piece_idx = p_idx
                    break
            if piece_idx == -1:
                for i, shape in enumerate(self.shapes):
                    if np.array_equal(shape, piece):
                        piece_idx = i
                        break

        if piece_idx != -1:
            mask = self.piece_xy_to_mask.get((piece_idx, x, y))
            if mask is None:
                raise ValueError(f"Pieza fuera del tablero o ubicacion invalida")
                
            board_u64 = np.packbits(self.board.flatten(), bitorder='little').view(np.uint64)[0]
            if (board_u64 & mask) != 0:
                raise ValueError(f"Movimiento invalido o colision en (x={x}, y={y})")
                
            new_board_u64 = board_u64 | mask
            
            rows_cleared = 0
            cols_cleared = 0
            clear_mask = np.uint64(0)
            
            for r_mask in self.ROW_MASKS:
                if (new_board_u64 & r_mask) == r_mask:
                    clear_mask |= r_mask
                    rows_cleared += 1
                    
            for c_mask in self.COL_MASKS:
                if (new_board_u64 & c_mask) == c_mask:
                    clear_mask |= c_mask
                    cols_cleared += 1
                    
            new_board_u64 &= ~clear_mask
            cleared_lines = rows_cleared + cols_cleared
            
            new_board_int8 = np.unpackbits(np.array([new_board_u64], dtype=np.uint64).view(np.uint8), bitorder='little').reshape((8, 8)).astype(np.int8)
            
        else:
            # Fallback para piezas desconocidas
            h, w = piece.shape
            if x < 0 or y < 0 or x + h > 8 or y + w > 8:
                raise ValueError(f"Pieza fuera del tablero (x={x}, y={y}, shape={piece.shape})")
                
            if np.any(self.board[x:x+h, y:y+w] & piece):
                raise ValueError(f"Movimiento invalido o colision en (x={x}, y={y})")
                
            new_board_int8, cleared_lines = self.updateBoard(self.board, piece, x, y)
            new_board_u64 = np.packbits(new_board_int8.flatten(), bitorder='little').view(np.uint64)[0]
            
        points, high_congestion = self.get_state_score(new_board_int8, new_board_u64, cleared_lines)
        
        self.board = new_board_int8
        
        if list_idx != -1:
            self.current_piece_indices.pop(list_idx)
        
        if len(self.current_piece_indices) == 0:
            self.get_new_pieces()
            
        game_over, risk_score = self.is_game_over(new_board_u64, self.current_piece_indices)
        
        info = {
            'risk_score': risk_score,
            'high_congestion': high_congestion
        }
        
        return self.board.copy(), points, game_over, info

def run_benchmark():
    sim = BlockBlastSim()
    sim.reset()
    
    start_time = time.time()
    turns = 0
    total_to_run = 150000
    
    while turns < total_to_run:
        if not sim.current_piece_indices:
            sim.get_new_pieces()
            
        piece_idx = sim.current_piece_indices[0]
        board_u64 = np.packbits(sim.board.flatten(), bitorder='little').view(np.uint64)[0]
        fits = sim.waysToFitList(piece_idx, board_u64)
        
        if not fits:
            sim.reset()
            continue
            
        x, y = fits[random.randint(0, len(fits)-1)]
        sim.step(0, x, y)
        turns += 1

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Ejecutados {turns} turnos en {elapsed:.4f} segundos.")
    print(f"Velocidad: {turns/elapsed:.2f} turnos/segundo.")

if __name__ == "__main__":
    run_benchmark()
