#!/usr/bin/dev python
#coding=utf-8
#author: spgoal
#date: Mar. 16th, 2018

import numpy as np

def produce_map(row=3, col=3):
  a = np.random.choice([0, 1], row * col)
  a = np.reshape(a, [row, col])
  print "a:"
  print a
  return a
  
class QLearningTable:
  def __init__(self, row, col, obs_map):
    self.name = "q-learning"
    self.row, self.col, self._map = row, col, obs_map
    self.state = [0, 1] # 0 for right move, and 1 for down move
    self.q_table = self.build_q_table(row, col)
    self.q_init = (0, 0)
    self.path_stack = []
    self.path_stack.append(self.q_init)
    self.direction_stack = []
    # fixed params
    self.epsilon = 0.9
    self.gamma = 0.9
    self.alpha = 0.1

  def build_q_table(self, row, col):
    qt = {}
    for i in range(row):
      for j in range(col):
        qt[(i, j)] = {}
        for s in self.state:
          qt[(i, j)][s] = 0
    return qt

  def get_argmax(self, dd):
    beg, res = -1000, 0
    for key, item in dd.items():
      if beg < item:
        beg, res = item, key
    return res
  
  def choose_action(self, state, definite=False):
    if definite:
      return self.get_argmax(self.q_table[state])
    if self.epsilon < np.random.uniform():
      action = np.random.choice([0, 1])
    else:
      action = self.get_argmax(self.q_table[state])
    return action

  def get_next_state(self, state, action):
    if action:
      if state[0] < self.row - 1:
        rr = state[0] + 1
        cc = state[1]
        obs = False
        return (rr, cc), obs
      else:
        return state, True
    else:
      if state[1] < self.col - 1:
        rr = state[0]
        cc = state[1] + 1
        obs = False
        return (rr, cc), obs
      else:
        return state, True

  def get_env_feedback(self, state, action):
    nst, obs = self.get_next_state(state, action)
    rr, cc = nst[0], nst[1]
    if obs:
      return -100
    elif rr == self.row - 1 and cc == self.col - 1:
      return 100
    else:
      return self._map[rr, cc].item()

  def is_terminated(self, state):
    return state[0] == self.row - 1 and state[1] == self.col - 1

  def learn(self, epoch):
    for epc in range(epoch):
      step_counter = 0
      st = self.q_init
      while not self.is_terminated(st):
        act = self.choose_action(st)
        next_st, _ = self.get_next_state(st, act)
        rwd = self.get_env_feedback(st, act)
        q_pre = self.q_table[st][act]
        if not self.is_terminated(next_st):
          q_tar = rwd + self.gamma * max(self.q_table[next_st])
        else:
          q_tar = rwd
        self.q_table[st][act] += self.alpha * (q_tar - q_pre)
        st = next_st
        step_counter += 1
    return self.q_table

  def check_res_path(self):
    st = self.q_init
    while not self.is_terminated(st):
      act = self.choose_action(st, True)
      self.direction_stack.append(act)
      st, obs = self.get_next_state(st, act)
      if obs:
        break
      self.path_stack.append(st)
    return self.path_stack, self.direction_stack

if __name__ == "__main__":
  a_map = produce_map(5, 5)
  ql = QLearningTable(5, 5, a_map)
  ql.learn(1000)
  res_path, res_dir = ql.check_res_path()
  print "res_path:", res_path
  print "res_dir:", res_dir
