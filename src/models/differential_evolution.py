import numpy as np
import random
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from tqdm import tqdm

class DifferentialEvolutionAlgorithm():
    def __init__(self,
                 num_epochs:int,
                 pop_size:int,
                 chrom_length:int,
                 value_ranges:list,
                 mutation_rate:float,
                 fitness_func, # Function Type,
                 crossover_rate = 0.8,
                 seed=42,
                 device="cpu"
                ):
        
        self.num_epochs = num_epochs
        self.pop_size = pop_size
        self.chrom_length = chrom_length
        self.device = device
        self.value_ranges = Tensor(value_ranges).to(self.device)
        self.mutation_rate = mutation_rate
        self.fitness_func = fitness_func
        self.seed = seed    
        self.crossover_rate = crossover_rate

        torch.manual_seed(seed=seed)

    def init_pop(self):
        """
        Initializes a matrix with random values from an uniform distribution
        """
        self.x_g = torch.rand(self.pop_size, self.chrom_length).to(self.device)
        # Denormalization process
        min_mat = self.value_ranges.T[0, :]
        max_mat = self.value_ranges.T[1,:]
        self.x_g = self.x_g * (max_mat - min_mat) + min_mat
        return

    def mutation(self):
        mutation_ind_indices_1 = torch.randint(low=0, high=self.pop_size, size=(1, self.pop_size)).to(self.device)
        mutation_ind_indices_2 = torch.randint(low=0, high=self.pop_size, size=(1, self.pop_size)).to(self.device)
        self.v_g = self.x_g + self.mutation_rate * \
              (self.x_g[mutation_ind_indices_1] - self.x_g[mutation_ind_indices_2])
        self.v_g = self.v_g.squeeze(dim=0)
        return
                
    def crossover(self):
        crossover_prob = torch.rand(self.pop_size, self.chrom_length).to(self.device)
        aleat_index = torch.randint(low=0, high=self.chrom_length, size=(1,self.pop_size))
        aleat_index_ohe = torch.zeros(aleat_index.size(1), aleat_index.max() + 1, dtype=torch.bool).to(self.device)
        self.u_g = self.x_g.clone()
        self.u_g[crossover_prob >= self.crossover_rate] = self.v_g[crossover_prob >= self.crossover_rate]
        self.u_g[aleat_index_ohe] = self.v_g[aleat_index_ohe]
        return

    def selection(self):
        self.fitness_x_g = self.fitness_func(self.x_g)
        self.fitness_u_g = self.fitness_func(self.u_g)
        replacement_indices = self.fitness_u_g > self.fitness_x_g
        self.x_g[replacement_indices] = self.u_g[replacement_indices]
        return
    
    def callback(self):
        ...
    
    
    def fit(self):
        self.init_pop()
        for epoch in tqdm(range(self.num_epochs)):
            self.mutation()
            self.crossover()
            self.selection()
        return self.x_g






if __name__ == "__main__":
    def schaffer_function(mat_x_y):
        x = mat_x_y[:, 0]
        y = mat_x_y[:, 1]
        g = 0.5 + (torch.pow((torch.sin( torch.sqrt( torch.pow(x, 2) + torch.pow(y, 2)))), 2) - 0.5)/ \
            (1 + 0.001 * (torch.pow(x, 2) + torch.pow(y, 2)))
        return g

    de_alg = DifferentialEvolutionAlgorithm(
                                            num_epochs=200,
                                            pop_size=100,
                                            chrom_length=2,
                                            value_ranges=[(-10,10), (-10,10)],
                                            mutation_rate=0.8,
                                            fitness_func=schaffer_function
                                            )
    best_solutions = de_alg.fit()
    def schaffer_function_plot(x,y):
        g = 0.5 + (np.power((np.sin( np.sqrt( np.power(x, 2) + np.power(y, 2)))), 2) - 0.5)/ \
            (1 + 0.001 * (np.power(x, 2) + np.power(y, 2)))
        return g

    x_data = best_solutions[:, 0]
    y_data = best_solutions[:, 1]
    z_data = schaffer_function_plot(x_data, y_data)

    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)

    X, Y = np.meshgrid(x, y)
    Z = schaffer_function_plot(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='jet', alpha=0.2)
    ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='binary', alpha=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
