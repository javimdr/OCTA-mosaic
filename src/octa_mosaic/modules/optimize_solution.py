import numpy as np


class OptimizeSolution:
    def __init__(self):
        self.x = None  # vector solución
        self.fitness = None
        self.fitness_record = dict(best=[], pop_mean=[], pop_std=[])

        self.nits = 0

        self.message = ""
        self.record = {}

    def add(self, key, value):
        self.record[key] = value

    def get(self, key):
        return self.record.get(key)

    def record_keys(self):
        return list(self.record.keys())

    def add_iteration_result(self, best_x, fitness_values):
        self.x = best_x

        best = float(np.max(fitness_values))
        mean = float(np.mean(fitness_values))
        std = float(np.std(fitness_values))
        self.fitness = best
        self.fitness_record["best"].append(best)
        self.fitness_record["pop_mean"].append(mean)
        self.fitness_record["pop_std"].append(std)
        self.nits += 1

    def set_message(self, message):
        self.message = str(message)

    def set_execution_time(self, exe_time):
        self.execution_time = exe_time

    def __str__(self):
        out = "Optimizer solution:\n"
        out += f" * Fitness: {self.fitness:.4f}\n"
        out += f" * x: {self.x}\n"
        return out
