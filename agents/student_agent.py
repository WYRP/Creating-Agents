from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from agents.agent import Agent
from queue import Queue

@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(Agent, self).__init__()
        self.name = "StudentAgent"
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))  # Up, Right, Down, Left
        self.depth_limit = 3  # Depth limit for minimax search

    def check_boundary(self, pos, board_size):
        x, y = pos
        return 0 <= x < board_size and 0 <= y < board_size

    def bfs_search(self, chess_board, start_pos, max_step):
        queue = Queue()
        queue.put((start_pos, 0))
        visited = set([start_pos])

        while not queue.empty():
            current_spot, steps_taken = queue.get()
            if steps_taken < max_step:
                for move in self.moves:
                    next_spot = (current_spot[0] + move[0], current_spot[1] + move[1])
                    if self.check_boundary(next_spot, chess_board.shape[0]) and next_spot not in visited:
                        if not chess_board[next_spot[0], next_spot[1]].any():
                            return next_spot 
                        visited.add(next_spot)
                        queue.put((next_spot, steps_taken + 1))

        return start_pos

    def heuristic(self, chess_board, new_pos, adv_pos):
        barrier_dir = 1
        far = -1
        for way, step in enumerate(self.moves):
            place_to_check = (new_pos[0] + step[0], new_pos[1] + step[1])
            if not self.check_boundary(place_to_check, chess_board.shape[0]) or chess_board[place_to_check[0], place_to_check[1]].any():
                continue
            dis_from_adver = np.linalg.norm(np.array(place_to_check) - np.array(adv_pos))
            if dis_from_adver > far:
                far = dis_from_adver
                barrier_dir = way
        return far  # Return a numerical score as heuristic

    def minimax(self, chess_board, my_pos, adv_pos, depth, alpha, beta, is_maximizing):
        if depth == 0:
            return self.heuristic(chess_board, my_pos, adv_pos)

        best_score = float('-inf') if is_maximizing else float('inf')
        possible_moves = []
        for move in self.moves:
            next_pos = (my_pos[0] + move[0], my_pos[1] + move[1])
            if self.check_boundary(next_pos, chess_board.shape[0]) and not chess_board[next_pos[0], next_pos[1]].any():
                possible_moves.append(next_pos)
        
        for next_pos in possible_moves:
            new_board = deepcopy(chess_board)
            new_board[my_pos[0], my_pos[1]] = 1  # Mark my current position as visited
            if is_maximizing:
                score = self.minimax(new_board, next_pos, adv_pos, depth - 1, alpha, beta, False)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                score = self.minimax(new_board, adv_pos, next_pos, depth - 1, alpha, beta, True)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
            if beta <= alpha:
                break

        return best_score

    def step(self, chess_board, my_pos, adv_pos, max_step):
        best_move = my_pos
        best_score = float('-inf')
        for move in self.moves:
            next_pos = (my_pos[0] + move[0], my_pos[1] + move[1])
            if self.check_boundary(next_pos, chess_board.shape[0]) and not chess_board[next_pos[0], next_pos[1]].any():
                score = self.minimax(deepcopy(chess_board), next_pos, adv_pos, self.depth_limit, float('-inf'), float('inf'), False)
                if score > best_score:
                    best_score = score
                    best_move = next_pos

        barrier_dir = self.heuristic(chess_board, best_move, adv_pos)
        return best_move, barrier_dir
