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
        self.q_table = defaultdict(lambda: [0,0,0,0])
        self.moves = ["up","down","left","right"]
        self.episode_buffer = []
        #epsilon must be between 0 and 1.0
        #1.0 is completely stochastic, 0.0 is the greedy policy
        self.epsilon = .35
        #gamma is the discount rate, 1.0 props end reward through all states
        self.gamma = .99
        #setup rewards
        self.turn_reward = -1.0
        self.win_reward = 100.0
        self.loss_reward = -100.0
        #update rate
        self.update_rate = .5
        self.total_reward = 0

    def setup(self, game):
        self.game = game
        self.mybotid = self.game.my_botid
        if self.mybotid == 0 and isfile("bot0qtable"):
            sys.stderr.write('loading Bot0 table\n')
            sys.stderr.flush()
            temp_q_table = pickle.load( open( "bot0qtable", "rb" ))
            self.q_table = defaultdict(lambda: [0,0,0,0],temp_q_table)
        if self.mybotid == 1 and isfile("bot1qtable"):
            sys.stderr.write('loading Bot1 table\n')
            sys.stderr.flush()
            temp_q_table = pickle.load( open( "bot1qtable", "rb" ))
            self.q_table = defaultdict(lambda: [0,0,0,0],temp_q_table)

    def do_turn(self):
        state_field = self.game.field
        action_values = self.q_table[state_field.hash] 
        action_probs = self.make_move_probs(action_values)       
        order = np.random.choice(4,p=action_probs)
        reward, terminal = self.get_reward(self.moves[order])
        self.total_reward += reward
        self.episode_buffer.append((state_field.hash,order,reward))
        
        if terminal:
            self.update_q_table()
        
        self.game.issue_order(self.moves[order])
    
    def one_hot_possible_moves(self):
        #gives a list of tuples where 0 is coord change and 1 is word for move
        my_legal = self.game.field.legal_moves(self.game.my_botid, self.game.players)
        legal_move_names = [m[1] for m in my_legal]
        one_hot_moves = np.zeros((4,))
        for i, move in enumerate(self.moves):
            if move in legal_move_names:
                one_hot_moves[i] = 1.0
        return one_hot_moves
        
    def make_move_probs(self, action_values):
        possible_moves = self.one_hot_possible_moves()
        is_max = [int(x==np.max(action_values)) for x in action_values]
        proposed_probs = np.zeros((4,))
        #if multiple max values, choose randomly from max values
        if np.sum(is_max) > 1.0:
            proposed_probs = is_max/np.sum(is_max)
        #else use epsilon greedy stochastic policy
        else:
            action_probs = np.ones_like(action_values)*(self.epsilon/len(action_values))
            action_probs[np.argmax(action_values)] += (1.0-self.epsilon)
            proposed_probs = action_probs
        #remove invalid choices
        proposed_probs = np.multiply(proposed_probs,possible_moves)
        #normalize to sum 1
        if(np.sum(proposed_probs)>0):
            proposed_probs = proposed_probs/np.sum(proposed_probs)
            return proposed_probs
        return [.25,.25,.25,.25]

    
    def get_reward(self,move):
        my_legal = self.game.field.legal_moves(self.game.my_botid, self.game.players)
        if len(my_legal) == 0:
            sys.stderr.write('I lose\n')
            sys.stderr.flush()
            return self.loss_reward, True
        opponent_legal = self.game.field.legal_moves(self.game.other_botid, self.game.players)
        if len(opponent_legal) == 0:
            sys.stderr.write('I win\n')
            sys.stderr.flush()
            return self.win_reward, True
        return self.turn_reward, False

    def update_q_table(self):
        print(self.total_reward,file=sys.stderr)
        sys.stderr.flush()
        episode_length = len(self.episode_buffer)
        #apply discounts through states
        reward_discount_rates = [self.gamma**x for x in np.arange(episode_length)]
        reward_trace = [reward for state,action,reward in self.episode_buffer]

        #for each state, get the discounted value of the trace
        #calculate the error and do running average update 
        for i, episode_step in enumerate(self.episode_buffer):
            state_hash, action, reward = episode_step
            action_value = np.sum(np.multiply(reward_discount_rates[0:episode_length-i],reward_trace[i:]))
            value_error = action_value - self.q_table[state_hash][action]
            self.q_table[state_hash][action] += self.update_rate*value_error
        
        #pickle can't save defaultdict with lambda
        q_table = dict(self.q_table)
        if self.mybotid == 0:
            pickle.dump( q_table, open( "bot0qtable", "wb" ))
        if self.mybotid == 1:
            pickle.dump( q_table, open( "bot1qtable", "wb" ))
            