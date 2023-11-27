# Student agent: Add your own agent here
import math

import world
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]


'''
    wanted to use a list to represent the MCTS tree, but since it is not a binary tree
    That idea would not work. So making a class like this would make more sense, cuz each node
    now would be an object of class Node'''


import math
from world import World

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.num_visited = 0
        self.num_rollout = 0
        self.utility_sc = 0

    def add_child(self, move, state):
        child_node = Node(state, self, move)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return all(child.num_visited > 0 for child in self.children)

    def best_child(self, c_param=C):
        choices_weights = [
            (child.utility_sc / child.num_rollout) + c_param * math.sqrt((2 * math.log(self.num_visited) / child.num_rollout))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self):
        # Implementation of the rollout policy to evaluate utility score
        pass

    def rollout(self):
        # Implementation of a rollout (simulation)
        pass

    def backprop(self, result):
        # Update utility scores and visit counts
        self.num_visited += 1
        self.utility_sc += result
        if self.parent:
            self.parent.backprop(result)

# Usage
root = Node(World.get_current_state())
current_node = root
while not current_node.is_terminal_node():
    while not current_node.is_fully_expanded():
        current_node = current_node.expand()
    current_node = current_node.best_child()
    current_node.rollout()
