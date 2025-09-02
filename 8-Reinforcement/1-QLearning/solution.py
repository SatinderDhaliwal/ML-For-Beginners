import matplotlib.pyplot as plt

from rlboard import *

width, height = 8, 8
m = Board(width, height)
m.randomize(seed=13)

actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }

class state:
    def __init__(self,board,energy=10,fatigue=0,init=True):
        self.board = board
        self.energy = energy
        self.fatigue = fatigue
        self.dead = False
        if init:
            self.board.random_start()
        self.update()

    def at(self):
        return self.board.at()

    def update(self):
        if self.at() == Board.Cell.water:
            self.dead = True
            return
        if self.at() == Board.Cell.tree:
            self.fatigue = 0
        if self.at() == Board.Cell.apple:
            self.energy = 10

    def move(self,a):
        self.board.move(a)
        self.energy -= 1
        self.fatigue += 1
        self.update()

    def is_winning(self):
        return self.energy > self.fatigue

    def info(self):
        return self.energy, self.fatigue

def reward(s_in):
    re = s_in.energy-s_in.fatigue
    if s_in.at()==Board.Cell.wolf:
        return 100 if s_in.is_winning() else -100
    if s_in.at()==Board.Cell.water:
        return -100
    return re

def probs(actions_q,eps=1e-4):
    actions_q = actions_q-actions_q.min()+eps
    actions_q = actions_q/actions_q.sum()
    return actions_q

Q = np.ones((width,height,len(actions)),dtype=float)*1.0/len(actions)

lpath = []
p = 0.000
for epoch in range(10):
    #clear_output(wait=True)
    print(f"Epoch = {epoch}",end='\n')

    # Pick initial point
    s = state(m)

    # Start travelling
    n=0
    cum_reward = 0
    while True:
        m.plot_energy(s.info())
        x,y = s.board.human
        v = probs(Q[x,y])
        while True:
            a = random.choices(list(actions),weights=v)[0]
            dpos = actions[a]
            m.plot_move(Q, (x, y), action_idx[a], epoch, p)
            if s.board.is_valid(s.board.move_pos(s.board.human,dpos)):
                break
        s.move(dpos)
        r = reward(s)
        m.plot_energy(s.info())
        if abs(r)==100: # end of game
            print(f" {n} steps",end='\n')
            m.plot_finished(Q, epoch, n, p)
            lpath.append(n)
            break

        alpha = np.exp(-n / 3000)
        gamma = 0.5
        ai = action_idx[a]
        new_value = (1 - alpha) * Q[x,y,ai] + alpha * (r + gamma * Q[x+dpos[0], y+dpos[1]].max())
        m.plot_moved(Q, (x, y), ai, new_value, r, epoch, alpha, p)
        Q[x,y,ai] = new_value
        n+=1

plt.ioff()
plt.plot(lpath)
plt.show()

