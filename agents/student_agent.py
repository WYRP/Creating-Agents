# Student agent: Add your own agent here
import math

import world
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


from queue import Queue
import time
from world import World  # Import the World class


@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {"u": 0, "r": 1, "d": 2, "l": 3}


    def bfs_with_heuristic(self, chess_board, my_pos, adv_pos, max_step):
        queue = Queue()
        queue.put((my_pos, 0))  # (position, depth)
        best_score = float('-inf')
        best_move = None

        while not queue.empty():
            current_pos, depth = queue.get()
            if depth >= max_step:
                continue

            for move in get_legal_moves(chess_board, current_pos, max_step):
                new_pos = apply_move(current_pos, move)
                score = self.heuristic(chess_board, new_pos, adv_pos)
                
                if score > best_score:
                    best_score = score
                    best_move = move

                if not is_terminal(chess_board, new_pos):
                    queue.put((new_pos, depth + 1))

        return best_move
    
    def is_move_valid(self, chess_board, best_move, barrier_dir):
        # Adapt the logic from check_valid_step here
        # You may need to adjust the parameters and logic to match your agent's needs
        return valid  # Return True if the move is valid, False otherwise

    def step(self, chess_board, my_pos, adv_pos, max_step):
        start_time = time.time()
        # Call BFS with Heuristic
        best_move = self.bfs_with_heuristic(chess_board, my_pos, adv_pos, max_step)
        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")

        return best_move, self.dir_map["u"]

current_player_obj, current_player_pos, adversary_player_pos = world.get_current_positions()

