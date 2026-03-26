import time
import math
import random
import numpy as np

class MCTSNode:
    def __init__(self, state, parent=None, move=None, is_root=False):
        self.state = state
        self.parent = parent
        self.move = move
        self.is_root = is_root
        self.children = []
        self.visits = 0
        self.score_sum = 0.0
        
        self.game_over = False
        self.info = {}
        
        self.untried_moves = self._get_legal_moves()
        
    def _get_legal_moves(self):
        if not self.state.current_piece_indices:
            return []
            
        board_u64 = np.packbits(self.state.board.flatten(), bitorder='little').view(np.uint64)[0]
        all_moves = []
        for list_idx, piece_idx in enumerate(self.state.current_piece_indices):
            fits = self.state.waysToFitList(piece_idx, board_u64)
            for r, c in fits:
                all_moves.append((list_idx, piece_idx, r, c))
                
        if not all_moves:
            return []
            
        BEAM_WIDTH_ROOT = 15
        BEAM_WIDTH_CHILD = 8
        
        if self.is_root and len(all_moves) > BEAM_WIDTH_ROOT:
            # Evaluacion superficial rápida (Heuristica de política)
            scored_moves = []
            for move in all_moves:
                list_idx, piece_idx, r, c = move
                test_state = self.state.clone()
                _, points, game_over, info = test_state.step(list_idx, r, c)
                
                score = points
                if game_over: score -= 100000
                if info.get('high_congestion'): score -= 5000
                score -= info.get('risk_score', 0) * 1000
                
                scored_moves.append((score, move))
                
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            best_moves = [m[1] for m in scored_moves[:BEAM_WIDTH_ROOT]]
            return best_moves
        else:
            random.shuffle(all_moves)
            if len(all_moves) > BEAM_WIDTH_CHILD:
                return all_moves[:BEAM_WIDTH_CHILD]
            return all_moves

    def expand(self):
        move = self.untried_moves.pop()
        list_idx, piece_idx, r, c = move
        
        next_state = self.state.clone()
        _, points, game_over, info = next_state.step(list_idx, r, c)
        
        child = MCTSNode(next_state, parent=self, move=move, is_root=False)
        child.step_points = points
        child.game_over = game_over
        child.info = info
        
        self.children.append(child)
        return child
        
    def get_best_child(self, c_param=1.414):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                return child
            
            exploit = child.score_sum / child.visits
            explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
            
            if child.game_over:
                exploit -= 100000
            if child.info.get('high_congestion', False):
                exploit -= 5000
            exploit -= child.info.get('risk_score', 0) * 1000
            
            uct = exploit + explore
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child


def rollout(state):
    current_state = state.clone()
    total_points = 0.0
    
    while current_state.current_piece_indices:
        board_u64 = np.packbits(current_state.board.flatten(), bitorder='little').view(np.uint64)[0]
        legal_moves = []
        for list_idx, piece_idx in enumerate(current_state.current_piece_indices):
            fits = current_state.waysToFitList(piece_idx, board_u64)
            for r, c in fits:
                legal_moves.append((list_idx, r, c))
                
        if not legal_moves:
            total_points -= 50000
            break
            
        list_idx, r, c = random.choice(legal_moves)
        _, points, game_over, info = current_state.step(list_idx, r, c)
        total_points += points
        
        if game_over:
            total_points -= 50000
            break
            
        if info['high_congestion']:
            total_points -= 2000
        total_points -= info['risk_score'] * 500
            
    return total_points


def run_mcts(root_state, time_limit):
    start_time = time.time()
    root = MCTSNode(root_state, is_root=True)
    
    iterations = 0
    while time.time() - start_time < time_limit:
        node = root
        
        # 1. Select
        while not node.untried_moves and node.children:
            node = node.get_best_child()
            
        # 2. Expand
        if node.untried_moves and not node.game_over:
            node = node.expand()
            
        # 3. Rollout
        if not node.game_over:
            score = rollout(node.state)
        else:
            score = -50000
            
        # 4. Backpropagate
        temp = node
        accumulated_score = score
        while temp is not None:
            if hasattr(temp, 'step_points'):
                accumulated_score += temp.step_points
            temp.visits += 1
            temp.score_sum += accumulated_score
            temp = temp.parent
            
        iterations += 1
        
    if not root.children:
        return None, iterations
        
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move, iterations


class BlockBlastBrain:
    def __init__(self, sim):
        self.sim = sim
        
    def think_and_play(self, time_limit_ms=100):
        """
        Retorna la lista de movimientos para la mano actual dentro del tiempo.
        Devuelve formato: [(pieza_idx, x, y), ...] de inmediato.
        """
        time_limit_sec = time_limit_ms / 1000.0
        moves = []
        
        current_state = self.sim.clone()
        budgets = [0.50 * time_limit_sec, 0.30 * time_limit_sec, 0.20 * time_limit_sec]
        
        pieces_to_play = len(current_state.current_piece_indices)
        
        for i in range(pieces_to_play):
            budget = budgets[i] if i < len(budgets) else 0.01
            
            move, iters = run_mcts(current_state, budget)
            if not move:
                print("MCTS: No valid moves left. Game Over.")
                break
                
            list_idx, piece_idx, r, c = move
            moves.append((piece_idx, r, c))
            
            _, _, game_over, _ = current_state.step(list_idx, r, c)
            if game_over:
                break
                
        return moves

if __name__ == "__main__":
    from block_blast_simulator import BlockBlastSim
    
    sim = BlockBlastSim()
    sim.reset()
    brain = BlockBlastBrain(sim)
    
    start_t = time.time()
    best_moves = brain.think_and_play(time_limit_ms=100)
    end_t = time.time()
    
    print(f"Mejores 3 jugadas decididas en {(end_t - start_t)*1000:.2f} ms:")
    for m in best_moves:
        print(f" - Pieza {m[0]} -> (x: {m[1]}, y: {m[2]})")
