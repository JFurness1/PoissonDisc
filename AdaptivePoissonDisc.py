import numpy as np
from numpy.random import random as rng
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from time import time


class Sample:
    def __init__(self, x, y, R):
        self.x = x
        self.y = y
        self.R = R

class AdaptivePoissonSamples:
    def __init__(self, height_map, N_SAMPLES: int = 10000, R_MIN: float = 0.005, R_MAX: float = 0.04):
        self.height_map = height_map
        self.N_SAMPLES = N_SAMPLES
        self.R_MIN = R_MIN
        self.R_MAX = R_MAX
    

        self.EPSILON = 1e-4
        self.CELL_SIZE = self.R_MAX/np.sqrt(2)  # divided by sqrt[dimension]
        self.N_CELLS = int(1.0/self.CELL_SIZE)
        self.K = 30
        self.IMPROVED_SAMPLING = False

        self.grid = [[[] for d in range(self.N_CELLS)] for d in range(self.N_CELLS)]
        self.samples = []

        self.fig, self.ax = plt.subplots(1)
        self.ax.axis('equal')
        plt.ion()
        plt.show()
        
    def generate(self):
        x = rng()
        y = rng()
        self.samples = [Sample(x, y, self.get_r_value((x, y)))]
        sample_idx = 0
        active = [sample_idx]
        idx = self.get_grid_indices(self.samples[sample_idx])
        self.grid[idx[0]][idx[1]].append(self.samples[0])
        
        while len(active) > 0:
            active_idx = int(rng()*len(active))
            sample_idx = active[active_idx]
            current = self.samples[sample_idx]
            success = False
            seed = rng()
            for i in range(self.K):
                modx, mody = self._get_random_displacement(current.R, i, seed)
                dx = current.x + modx
                dy = current.y + mody

                if dx < 0 or dx > 1 or dy < 0 or dy > 1:
                    continue
                
                sample = Sample(dx, dy, self.get_r_value((dx, dy)))

                if self.is_valid_sample(sample):
                    self.samples.append(sample)
                    active.append(len(self.samples) - 1)
                    sample_idx = self.get_grid_indices(sample)
                    self.grid[sample_idx[0]][sample_idx[1]].append(self.samples[-1])
                    success = True
                    # self.plotit(active=active)
                    break
            if not success:
                del active[active_idx]
                
            if len(self.samples) == self.N_SAMPLES:
                break
        
    def is_valid_sample(self, sample):
        if sample.x < 0 or sample.x > 1.0 or sample.y < 0 or sample.y > 1.0:
            return False
        grid_idx = self.get_grid_indices(sample)
        for dx in range(grid_idx[0] - 2, grid_idx[0] + 3):
            if dx < 0 or dx > self.N_CELLS - 1:
                continue
            for dy in range(grid_idx[1] - 2, grid_idx[1] + 3):
                if dy < 0 or dy > self.N_CELLS - 1:
                    continue
                for other in self.grid[dx][dy]:
                    if self._distance(sample, other) < max(sample.R, other.R):
                        return False
        return True
    
    def get_grid_indices(self, sample):
        return (max(0, min(int(sample.x//self.CELL_SIZE), len(self.grid) - 1)), 
                max(0, min(int(sample.y//self.CELL_SIZE), len(self.grid[0]) - 1)))
    
    def get_r_value(self, pt):
        return self.R_MIN + self._img_function(pt)*(self.R_MAX - self.R_MIN)

    def _distance(self, s1, s2):
        return np.sqrt((s1.x - s2.x)**2 + (s1.y - s2.y)**2)
    
    def _get_random_displacement(self, R, i, seed):
        if self.IMPROVED_SAMPLING:
            theta = 2.0*np.pi*(seed + i/self.K)
            r = R + self.EPSILON
        else:
            theta = 2.0*np.pi*rng()
            r = R + rng()*R
        return r*np.cos(theta), r*np.sin(theta)

    def plotit(self, query=None, valid=False, active=[]):
        self.ax.cla()
        # self.ax.vlines(np.arange(0, 1, self.CELL_SIZE), 0, 1, colors='gray')
        # self.ax.hlines(np.arange(0, 1, self.CELL_SIZE), 0, 1, colors='gray')
        # clist = [self._img_function((s.x, s.y)) for s in self.samples]
        # for s in self.samples:
            # self.ax.add_patch(plt.Circle((s.x, s.y), s.R, facecolor=None, ec='k', fill=False))
            # c = self._img_function((s.x, s.y))
            # c = 0
            # self.ax.scatter([s.x], [s.y], color=(0, c, 0), marker='.')
        self.ax.scatter([s.x for s in self.samples], [s.y for s in self.samples], c='k', marker='.')

        if query is not None:
            if valid:
                c = 'g'
            else:
                c = 'r'
            self.ax.add_patch(plt.Circle((query.x, query.y), query.R, facecolor=None, edgecolor=c, fill=False))
            self.ax.scatter([query.x], [query.y], c=c, marker='.')
        self.ax.set_title(f"Active set size: {len(active)}")
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.set_aspect(1.0)
        plt.draw()
        plt.pause(0.001)

    def _img_function(self, pt):
        return self.height_map[int(pt[0]*self.height_map.shape[0]), int(pt[1]*self.height_map.shape[1])]**(2)

img = Image.open("/home/jim/Documents/Poisson Disc/Jim.png", 'r')

ps = AdaptivePoissonSamples(np.array(ImageOps.grayscale(img)).T/255.0, R_MIN=0.001, R_MAX=0.05)
input('asdasd')
print("begin...")
stime = time()
ps.generate()
print(f"Generate in {time() - stime}")
print(f"Made {len(ps.samples)}")

plt.ioff()
ps.plotit()
plt.show()
