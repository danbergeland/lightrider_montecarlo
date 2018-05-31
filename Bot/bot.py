import random
import sys
import pickle
import numpy as np
from collections import defaultdict
from os.path import isfile

class Bot:
    def __init__(self):
        self.game = None
        self.mybotid = -1
        self.q_table = defaultdict(lambda: [.25,.25,.25,.25])
        self.moves = ["up","down","left","right"]
        self.episode_buffer = []

    def setup(self, game):
        self.game = game
        self.mybotid = self.game.my_botid
        if self.mybotid == 0 and isfile("bot0qtable"):
            self.q_table = pickle.load("bot0qtable")
        if self.mybotid == 1 and isfile("bot1qtable"):
            self.q_table = pickle.load("bot1qtable")

    def do_turn(self):
        state_field = self.game.field
        action_values = self.q_table[state_field.hash]        
        order = np.choice(4,p=action_values)
        self.episode_buffer.append((state_field.hash,order,1.0))
        self.check_endgame()
        self.game.issue_order(self.moves[order])

    def check_endgame(self):
        for s,a,r in self.episode_buffer:
            #TODO, update q table values in case of win or loss
            pass


