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
class Node():
    def __init__(self, current, parent, children = []):
        self.current = current
        self.parent = parent
        self.children = children
        # set the root as the position of the current player.
        self.root = world.get_current_player()
    def expand_children(self):
        child = Node(self.current, self.parent, self.children)
        return child




def selection(self, root, current):
    tree = []
    current = root
    tree.append(current)
    left_c = None
    right_c = None
    tree.append(left_c)
    tree.append(right_c)

def UCT(self, total_util, num_rollout,count,parent,C):
    exploitation = total_util/num_rollout
    exploration = math.sqrt(math.log(count,parent)/count)
    UCB1 = exploitation + C * exploration
    return UCB1



