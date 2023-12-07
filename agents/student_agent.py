# Student agent: Add your own agent here
import math

from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from agents.agent import Agent

from queue import Queue
import time



@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))  # Up, Right, Down, Left

    def check_boundary(self, pos, board_size):
        r, c = pos
        return 0 <= r < board_size and 0 <= c < board_size

    def bfs_search(self, chess_board, start_pos, max_step):
        queue = Queue()
        queue.put((start_pos, 0))  # (position, depth)
        visited = set([start_pos])

        while not queue.empty():
            current_pos, depth = queue.get()
            if depth >= max_step:
                continue

            for move in self.moves:
                new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

                # Check for boundary and avoid revisiting
                if new_pos in visited or not self.check_boundary(new_pos, chess_board.shape[0]):
                    continue

                # Check if the position is free of barriers
                if not chess_board[new_pos[0], new_pos[1]].any():  # Assuming chess_board is a numpy array
                    return new_pos  # Valid move found

                visited.add(new_pos)
                queue.put((new_pos, depth + 1))

        return start_pos  # If no move found, stay in place
    
    def heuristic(self, chess_board, new_pos, adv_pos):

        best_dir = 1  
        max_distance = -1

        for dir, move in enumerate(self.moves):
            adjacent_pos = (new_pos[0] + move[0], new_pos[1] + move[1])

            # Check if the adjacent position is within boundaries and not blocked
            if not self.check_boundary(adjacent_pos, chess_board.shape[0]) or chess_board[adjacent_pos[0], adjacent_pos[1]].any():
                continue

            # Calculate distance to adversary from this adjacent position
            distance_to_adv = np.linalg.norm(np.array(adjacent_pos) - np.array(adv_pos))

            # Select the direction that maximizes distance from the adversary
            if distance_to_adv > max_distance:
                max_distance = distance_to_adv
                best_dir = dir

        return best_dir
    
    def step(self, chess_board, my_pos, adv_pos, max_step):
        new_pos = self.bfs_search(chess_board, my_pos, max_step)
        barrier_dir = self.heuristic(chess_board, new_pos, adv_pos)
        return new_pos, barrier_dir


