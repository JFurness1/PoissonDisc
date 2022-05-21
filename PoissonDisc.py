import numpy as np
from numpy.random import random as rng
from numpy.random import choice
import matplotlib.pyplot as plt

class PoissonSamples:
    def __init__(self, N_SAMPLES: int = 100, R: float = 0.1):
        self.N_SAMPLES = N_SAMPLES
        self.R = R
    
        self.CELL_SIZE = R/np.sqrt(2)  # divided by sqrt[dimension]
        self.N_CELLS = int(1.0/self.CELL_SIZE)
        self.K = 30

        self.grid = np.full((self.N_CELLS, self.N_CELLS), -1, dtype=int)
        self.samples = []

        self.fig, self.ax = plt.subplots(1)
        plt.ion()
        plt.show()
        
    def generate(self):
        self.samples = [(rng(), rng())]
        sample_idx = 0
        active = [sample_idx]
        self.grid[self.get_grid_indices(self.samples[sample_idx])] = sample_idx
        
        while len(active) > 0:
            active_idx = int(rng()*len(active))
            sample_idx = active[active_idx]
            current = self.samples[sample_idx]
            success = False
            for i in range(self.K):
                dx = self._get_random_displacement()
                dy = self._get_random_displacement()
                sample = (current[0] + dx, current[1] + dy)

                if self.is_valid_sample(sample):
                    self.samples.append(sample)
                    active.append(len(self.samples) - 1)
                    sample_idx = self.get_grid_indices(sample)
                    self.grid[sample_idx] = len(self.samples) - 1
                    success = True
                    self.plotit(query=sample, valid=True, active=active)
                    break
            if not success:
                del active[active_idx]
                
            if len(self.samples) == self.N_SAMPLES:
                break
        
    def is_valid_sample(self, pt):
        if pt[0] < 0 or pt[0] > 1.0 or pt[1] < 0 or pt[1] > 1.0:
            return False
        grid_idx = self.get_grid_indices(pt)

        for dx in range(grid_idx[0] - 2, grid_idx[0] + 3):
            if dx < 0 or dx > self.N_CELLS - 1:
                continue
            for dy in range(grid_idx[1] - 2, grid_idx[1] + 3):
                if dy < 0 or dy > self.N_CELLS - 1:
                    continue
                other_idx = self.grid[dx, dy]
                if other_idx != -1 and self._distance(pt, self.samples[other_idx]) < self.R:
                    return False
        return True
    
    def get_grid_indices(self, pt):
        return (max(0, min(int(pt[0]//self.CELL_SIZE), self.grid.shape[0] - 1)), 
                max(0, min(int(pt[1]//self.CELL_SIZE), self.grid.shape[1] - 1)))
    
    def _distance(self, pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def _get_random_displacement(self):
        return self.R*(1 + rng())*np.sign(rng() - 0.5)

    def plotit(self, query=None, valid=False, active=[]):
        self.ax.cla()
        self.ax.vlines(np.arange(0, 1, self.CELL_SIZE), 0, 1, colors='gray')
        self.ax.hlines(np.arange(0, 1, self.CELL_SIZE), 0, 1, colors='gray')
        for s in self.samples:
            self.ax.add_patch(plt.Circle(s, self.R, facecolor=None, ec='k', fill=False))
            self.ax.scatter([s[0]], [s[1]], c='k', marker='.')
        if query is not None:
            if valid:
                c = 'g'
            else:
                c = 'r'
            self.ax.add_patch(plt.Circle(query, self.R, facecolor=None, ec=c, fill=False))
            self.ax.scatter([query[0]], [query[1]], c=c, marker='.')
        self.ax.set_title(f"Active set size: {len(active)}")
        self.ax.axis('equal')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        plt.draw()
        plt.pause(0.001)


ps = PoissonSamples()
ps.generate()

plt.ioff()
ps.plotit()
plt.show()
