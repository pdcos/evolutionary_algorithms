import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class CMA_ES():
    def __init__(self,
                 num_epochs:int,
                 lamb:int,
                 sigma,
                 mi:int,
                 chrom_length:int,
                 value_ranges:list,
                 fitness_func,
                 seed=1,
                 eval_every = 10,
                 verbose=0,
                 maintain_history=False,
                ):
        
        np.random.seed(seed=seed)
        self.num_epochs = num_epochs
        self.N = chrom_length
        self.x_mean = np.random.rand(self.N)
        self.lamb = lamb
        self.sigma = sigma # Step size
        self.mi = mi # Number of best candidates
        self.C = np.identity(self.N)
        self.fitness_func = fitness_func
        self.value_ranges = value_ranges
        #self.sigma = np.expand_dims(np.random.rand(self.N), axis=1).T
        self.C = np.identity(self.N)
        self.cov_mat = (self.sigma ** 2) * self.C
        self.weights = np.log(self.mi+1/2)-np.log(range(1, self.mi + 1))
        self.weights = self.weights/self.weights.sum()
        self.maintain_history = maintain_history
        self.x_i_history = []


        self.best_ind_list = np.zeros(self.num_epochs)
        self.avg_ind_list = np.zeros(self.num_epochs)
        self.eval_every = eval_every
        self.verbose = verbose

        self.min_mat = self.value_ranges.T[0, :]
        self.max_mat = self.value_ranges.T[1,:]


    def step(self):
        self.x_i = np.random.multivariate_normal(self.x_mean, self.cov_mat, size=self.lamb)
        self.f_x_i = self.fitness_func(self.x_i, self.value_ranges)
        mask = (-self.f_x_i).argsort()
        self.f_x_i = self.f_x_i[mask]
        self.x_i = self.x_i[mask]
        self.best_indvs = self.x_i[0:self.mi]
        self.cov_mat = np.cov(self.best_indvs.T)
        self.x_mean = np.dot(self.weights, self.best_indvs)
        #print(self.cov_mat)

        # Update step size (sigma) 
        #p_sigma = np.zeros(self.N)
        #print(p_sigma)
        #for i in range(self.mi):
        #    p_sigma += self.weights[i] * (self.best_indvs[i] - self.x_mean) / self.sigma
        #print(p_sigma)
        p_sigma= np.sum(self.weights[:, np.newaxis] * (self.best_indvs - self.x_mean) / self.sigma, axis=0)
        #print(p_sigma_2)
        p_sigma /= np.linalg.norm(p_sigma)
        self.sigma *= np.exp((np.linalg.norm(p_sigma) - 1) / self.N)

        

        if self.maintain_history:
            particle = self.x_i * (self.max_mat - self.min_mat) + self.min_mat
            self.x_i_history.append(particle)

    def callback(self):
        max_val = self.f_x_i.max()
        mean_val = np.mean(self.f_x_i)
        self.best_ind_list[self.curr_epoch] = max_val
        self.avg_ind_list[self.curr_epoch] = mean_val
        if (self.curr_epoch % self.eval_every == 0) and self.verbose != 0 :
            print(f"Epoch {self.curr_epoch}: Best: {max_val}, Average: {mean_val}")

    def fit(self):
        start_time = time.time()
        for epoch in tqdm(range(self.num_epochs)):
            self.curr_epoch = epoch
            self.step()
            self.callback()
        print("--- %s seconds ---" % (time.time() - start_time))
        return self.best_indvs

    def plot(self):
        plt.plot(self.best_ind_list, label="Best")
        plt.plot(self.avg_ind_list, label="Average")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Fitness Value")
        plt.legend()
        plt.show()
