from abc import ABC, abstractmethod
import numpy as np
import itertools
class Discretization(ABC):
    def __init__(self):
        self.data = []
        self.discretization = []

    def obtainDiscretization(self):
        lis = list(itertools.product(*self.data))
        for i in range(len(lis)):
            self.discretization.append(np.array(list(lis[i])))

    @abstractmethod
    def getClosestState(self, state):
        pass

    def getDiscretization(self):
        return self.discretization


class CartpoleDiscretization(Discretization):
    def __init__(self):
        super().__init__()

        self.max_pos = 2.4
        self.max_angle = 0.2095
        self.pos = [-self.max_pos, -self.max_pos / 2, -self.max_pos / 4, 0, -self.max_pos / 4, self.max_pos / 2, self.max_pos]
       # self.pos = [ -self.max_pos / 2, 0,  self.max_pos / 2]
        self.vel = [-1, 1]
        self.angle = [-self.max_angle, -self.max_angle / 2, -3 * self.max_angle / 8, -self.max_angle / 4,
                      -self.max_angle / 6, -self.max_angle / 8, 0, self.max_angle / 8, self.max_angle / 6,
                      self.max_angle / 4, 3 * self.max_angle / 8, self.max_angle / 2, self.max_angle]
        #self.angle = [ -self.max_angle / 2,  0,  self.max_angle / 2]
        self.angle_vel = [-1, 1]

        self.data = [self.pos, self.vel, self.angle, self.angle_vel]
        self.obtainDiscretization()


    def getClosestState(self, state):
        pos = state[0]
        vel = state[1]
        angle = state[2]
        angle_vel = state[3]

        state_pos = 0
        state_vel = 0
        state_angle = 0
        state_angle_vel = 0

        min_dist = np.inf
        for i in range(len(self.pos)):
            dist = np.linalg.norm(pos - self.pos[i])
            if dist < min_dist:
                min_dist = dist
                state_pos = self.pos[i]

        min_dist = np.inf
        for j in range(len(self.angle)):
            dist = np.linalg.norm(angle - self.angle[j])
            if dist < min_dist:
                min_dist = dist
                state_angle = self.angle[j]

        min_dist = np.inf
        for k in range(len(self.vel)):
            dist = np.linalg.norm(vel - self.vel[k])
            if dist < min_dist:
                min_dist = dist
                state_vel = self.vel[k]

        min_dist = np.inf
        for l in range(len(self.angle_vel)):
            dist = np.linalg.norm(angle_vel - self.angle_vel[l])
            if dist < min_dist:
                min_dist = dist
                state_angle_vel = self.angle_vel[l]


        rtn = [state_pos, state_vel, state_angle, state_angle_vel]
        return np.array(rtn)

class AcrobotDiscretization(Discretization):
    def __init__(self):
        super().__init__()
        self.cos1 = [-0.9, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.9]
        self.sin1 = [-0.9, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.9]
        self.cos2 = [-0.9, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.9]
        self.sin2 = [-0.9, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.9]
        self.vel1 = [-1, 1]
        self.vel2 = [-1, 1]
        self.data = [self.cos1, self.sin1, self.cos2, self.sin2, self.vel1, self.vel2]
        self.obtainDiscretization()

    def getClosestState(self, state):
        cos1 = state[0]
        sin1 = state[1]
        cos2 = state[2]
        sin2 = state[3]
        vel1 = state[4]
        vel2 = state[5]

        state_cos1 = -1
        state_sin1 = -1
        state_cos2 = -1
        state_sin2 = -1
        state_vel1 = -1
        state_vel2 = -1

        min_dist = np.inf
        for j in range(len(self.cos1)):
            if (np.linalg.norm(cos1 - self.cos1[j]) < min_dist):
                min_dist = np.linalg.norm(cos1 - self.cos1[j])
                state_cos1 = self.cos1[j]

        min_dist = np.inf
        for j in range(len(self.sin1)):
            if (np.linalg.norm(sin1 - self.sin1[j]) < min_dist):
                min_dist = np.linalg.norm(sin1 - self.sin1[j])
                state_sin1 = self.sin1[j]

        min_dist = np.inf
        for j in range(len(self.cos2)):
            if (np.linalg.norm(cos2 - self.cos2[j]) < min_dist):
                min_dist = np.linalg.norm(cos2 - self.cos2[j])
                state_cos2 = self.cos2[j]

        min_dist = np.inf
        for j in range(len(self.sin2)):
            if (np.linalg.norm(sin2 - self.cos2[j]) < min_dist):
                min_dist = np.linalg.norm(sin2 - self.sin2[j])
                state_sin2 = self.sin2[j]

        if vel1 > 0:
            state_vel1 = 1
        if vel2 > 0:
            state_vel2 = 1

        rtn = [state_cos1, state_sin1, state_cos2, state_sin2, state_vel1, state_vel2]
        return np.array(rtn)

class NoDiscretization(Discretization):
    def __init__(self):
        super().__init__()

    def getClosestState(self, state):
        return state