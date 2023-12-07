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
        super(Agent, self).__init__()
        self.name = "StudentAgent"
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))  # Up, Right, Down, Left

    def check_boundary(self, pos, board_size):
        x, y = pos
        return 0 <= x < board_size and 0 <= y < board_size

    def bfs_search(self, chess_board, start_pos, max_step):
        queue = Queue()
        queue.put((start_pos, 0))  # Starting position with depth 0
        visited = set([start_pos])  # We've seen the start already

        while not queue.empty():
            # Grab the next spot to check out
            current_spot, steps_taken = queue.get()
            if steps_taken < max_step:
                # Look in all directions
                for move in self.moves:
                    next_spot = (current_spot[0] + move[0], current_spot[1] + move[1])

                    # if this spot is within the board and we haven't visited yet...
                    if self.check_boundary(next_spot, chess_board.shape[0]) and next_spot not in visited:
                        # if there is no wall, we go to this position
                        if not chess_board[next_spot[0], next_spot[1]].any():
                            return next_spot 

                        # Mark as visited and add to the queue to check later
                        visited.add(next_spot)
                        queue.put((next_spot, steps_taken + 1))

        return start_pos  # Can't go anywhere, so stay put


    
    def heuristic(self, chess_board, new_pos, adv_pos):
        barrier_dir = 1
        far = -1

        # checking all the possible moves around the new position
        for way, step in enumerate(self.moves):
            place_to_check = (new_pos[0] + step[0], new_pos[1] + step[1])

            # If these place around me are out of bound or there is a wall, we can't go there
            if not self.check_boundary(place_to_check, chess_board.shape[0]) or chess_board[place_to_check[0], place_to_check[1]].any():
                continue

            # Calculate the distance between the new position and the adversary
            dis_from_adver = np.linalg.norm(np.array(place_to_check) - np.array(adv_pos))

            # I wanna stay away from the baddie!
            if dis_from_adver > far:
                far = dis_from_adver
                barrier_dir = way

        return barrier_dir

    
    def step(self, chess_board, my_pos, adv_pos, max_step):
        new_pos = self.bfs_search(chess_board, my_pos, max_step)
        barrier_dir = self.heuristic(chess_board, new_pos, adv_pos)
        return new_pos, barrier_dir