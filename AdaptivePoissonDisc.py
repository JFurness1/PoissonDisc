from email.utils import collapse_rfc2231_value
from random import sample
import matplotlib
import numpy as np
from numpy.random import random as rng
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from PIL import Image, ImageOps
from time import time


class Sample:
    def __init__(self, x, y, R):
        self.x = x
        self.y = y
        self.R = R

        self.color = matplotlib.colors.hsv_to_rgb((rng(),1.0,rng()))
    
        self.connections = set()

        self.triangle_list = []
        self.triangle_point_indices = []

    def set_index(self, index):
        self.index = index

class AdaptivePoissonSamples:
    def __init__(self, height_map, N_SAMPLES: int = 10000, R_MIN: float = 0.005, R_MAX: float = 0.04, ax = None, color='k'):
        self.height_map = height_map
        self.N_SAMPLES = N_SAMPLES
        self.R_MIN = R_MIN
        self.R_MAX = R_MAX
    

        self.EPSILON = 1e-4
        self.CELL_SIZE = self.R_MIN/np.sqrt(2)  # divided by sqrt[dimension]
        self.N_CELLS = int(1.0/self.CELL_SIZE) + 1
        self.K = 30
        self.IMPROVED_SAMPLING = False

        self.grid = [[[] for d in range(self.N_CELLS)] for d in range(self.N_CELLS)]
        self.samples = []

        self.voronoi_grid = np.full((self.N_CELLS, self.N_CELLS), np.nan, dtype=int)

        if ax is None:
            self.fig, self.ax = plt.subplots(1)
        else:
            self.fig = None
            self.ax = ax
        self.color = color
        self.ax.axis('equal')
        
    def generate(self):
        x = rng()
        y = rng()
        self.samples = [Sample(x, y, self.get_r_value((x, y)))]
        self.samples[0].set_index(0)
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
                    sample.set_index(len(self.samples) - 1)
                    active.append(len(self.samples) - 1)
                    self._add_to_all_cells(sample)
                    # sample_grid_idx = self.get_grid_indices(sample)
                    # self.grid[sample_grid_idx[0]][sample_grid_idx[1]].append(self.samples[-1])
                    success = True
                    # if len(self.samples)%7 == 0:
                    #     self.plotit(active=active)
                    break
            if not success:
                del active[active_idx]
                
            if len(self.samples) == self.N_SAMPLES:
                break
        
    def is_valid_sample(self, sample):
        if sample.x < 0 or sample.x > 1.0 or sample.y < 0 or sample.y > 1.0:
            return False
        grid_idx_min, grid_idx_max = self.get_grid_bounds(sample)
        for dx in range(grid_idx_min[0], grid_idx_max[0] + 1):
            if dx < 0 or dx > self.N_CELLS - 1:
                continue
            for dy in range(grid_idx_min[1], grid_idx_max[1] + 1):
                if dy < 0 or dy > self.N_CELLS - 1:
                    continue
                for other in self.grid[dx][dy]:
                    if self._distance(sample, other) < max(sample.R, other.R):
                        return False
        return True
    
    def get_grid_indices(self, sample):
        return (max(0, min(int(sample.x//self.CELL_SIZE), self.N_CELLS - 1)), 
                max(0, min(int(sample.y//self.CELL_SIZE), self.N_CELLS - 1)))
    
    def get_cell_center(self, x, y):
        return ((x + 0.5)*self.CELL_SIZE, (y + 0.5)*self.CELL_SIZE)

    def get_cell_bottom_left(self, x, y):
        return x*self.CELL_SIZE, y*self.CELL_SIZE

    def get_grid_bounds(self, sample):
        min_indices = self.get_grid_indices(Sample(sample.x - sample.R, sample.y - sample.R, sample.R))
        max_indices = self.get_grid_indices(Sample(sample.x + sample.R, sample.y + sample.R, sample.R))
        return min_indices, max_indices

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

    def _add_to_all_cells(self, sample):
        min_indices, max_indices = self.get_grid_bounds(sample)

        for i in range(min_indices[0], max_indices[0] + 1):
            for j in range(min_indices[1], max_indices[1] + 1):
                self.grid[i][j].append(sample)

    def plotit(self, query=None, valid=False, active=[]):
        indices = np.zeros((len(self.triangle_list), 3), dtype=int)

        for i, tri in enumerate(self.triangle_list):
            for j in range(3):
                indices[i, j] = tri[j].index
        x = [s.x for s in self.samples]
        y = [s.y for s in self.samples]
        tris = matplotlib.tri.Triangulation(x, y, triangles=indices)
        self.ax.tripcolor(
            tris, 
            facecolors=np.array([0.5 for g in range(len(self.triangle_list))]))
        self.ax.triplot(tris)

        self.ax.vlines([i*self.CELL_SIZE for i in range(self.N_CELLS)], 0, 1, colors='gray')
        self.ax.hlines([i*self.CELL_SIZE for i in range(self.N_CELLS)], 0, 1, colors='gray')
        if True:
            colors = [s.color for s in self.samples]
        else:
            colors = 'w'
        self.ax.scatter([s.x for s in self.samples], [s.y for s in self.samples], 
                c=colors, marker='o',s = 10)

        # for s in self.samples:
        #     self.ax.add_patch(plt.Circle((s.x, s.y), s.R, facecolor=None, edgecolor=s.color, fill=False))

        cells_x = []
        cells_y = []
        cells_c = []
        for x in range(self.N_CELLS):
            for y in range(self.N_CELLS):
                xy = self.get_cell_center(x, y)
                cells_x.append(xy[0])
                cells_y.append(xy[1])
                cells_c.append(self.samples[self.voronoi_grid[x][y]].color)
        self.ax.scatter(cells_x, cells_y, c=cells_c, marker='x')

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

    def _img_function(self, pt):
        return self.height_map[int(pt[0]*self.height_map.shape[0]), int(pt[1]*self.height_map.shape[1])]**(2)

    def _fill_approximate_voronoi(self):
        for i in range(self.N_CELLS):
            for j in range(self.N_CELLS):
                best = 1e30
                cell_center = self.get_cell_center(i, j)
                for entry in self.grid[i][j]:
                    distance = self._distance(entry, Sample(cell_center[0], cell_center[1], np.nan))
                    if distance < best:
                        best = distance
                        self.voronoi_grid[i, j] = entry.index


    def old_build_connections(self):
        for i in range(self.N_CELLS):
            for j in range(self.N_CELLS):
                if self.voronoi_grid[i][j] == -1:
                    print("No connections")
                    continue
                sample = self.samples[self.voronoi_grid[i][j]]
                diffs = set([sample])
                for di in range(max(0, i - 1), min(self.N_CELLS, i + 2)):
                    for dj in range(max(0, j - 1), min(self.N_CELLS, j + 2)):
                        diffs.add(self.samples[self.voronoi_grid[di][dj]])
                diffs.remove(sample)
                
                sample.connections = diffs
    
    def build_tris(self):
        self.triangle_list = []
        for i in range(1, self.N_CELLS):
            for j in range(1, self.N_CELLS):
                if np.isnan(self.voronoi_grid[i][j]):
                    print("No connections")
                    continue
                point = self.samples[self.voronoi_grid[i][j]]
                neighbours = set([
                    point,
                    self.samples[self.voronoi_grid[i - 1][j]],
                    self.samples[self.voronoi_grid[i][j - 1]],
                    self.samples[self.voronoi_grid[i - 1][j - 1]]
                    ])
                
                if len(neighbours) == 1:
                    pass
                elif len(neighbours) == 2:
                    pass
                elif len(neighbours) == 3:
                    self.triangle_list.append(list(neighbours))
                elif len(neighbours) == 4:
                    # Should Delaunay check this
                    plist = list(neighbours)
                    centroid = self.get_cell_bottom_left(i, j)
                    plist = sorted(plist, key = lambda p: np.arctan2(p.y - centroid[1], p.x - centroid[0]))
                    self.triangle_list.append(plist[1:])
                    self.triangle_list.append([plist[3]] + plist[:2])
                else:
                    raise Exception("panic! More than 4 neighbours")




                            
                


img = Image.open("/home/jim/Documents/Poisson Disc/Jim.png", 'r')

k = np.array(ImageOps.grayscale(img)).T/255.0

fig, ax = plt.subplots(1)

ksample = AdaptivePoissonSamples(k, R_MIN=0.15, R_MAX=0.15, ax=ax, color='w')

print("begin...")
stime = time()
ksample.generate()
print(f"Points generated in {time() - stime}")
stime = time()
ksample._fill_approximate_voronoi()
print(f"Voronoi built in {time() - stime}")
stime = time()
ksample.build_tris()
print(f"Connections made in {time() - stime}")
ksample.plotit()
print(f"Made {len(ksample.samples)}")

ksample.ax.set_xlim([-0.1, 1.1])
ksample.ax.set_ylim([-0.1, 1.1])
plt.show()

# 0.0221 0.497
# 0.053 0.5836
# 0.1328 0.615