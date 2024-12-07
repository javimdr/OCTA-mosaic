from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class IterationState:
    it: int
    population: np.ndarray
    population_fitness: np.ndarray
    idx_best_solution: int

    @property
    def x(self) -> np.ndarray:
        return self.population[self.idx_best_solution]

    @property
    def fitness(self) -> float:
        return self.population_fitness[self.idx_best_solution]

    @classmethod
    def from_dict(cls, it_data: Dict[str, Any]) -> "IterationState":
        iteration_state = IterationState(
            it=it_data["it"],
            population=np.array(it_data["population"], "float32"),
            population_fitness=np.array(it_data["population_fitness"], "float32"),
            idx_best_solution=it_data["idx_best_solution"],
        )
        return iteration_state

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            print("Not same class")
            return False

        same_it = int(self.it) == int(other.it)
        same_population = np.allclose(self.population, other.population)
        same_population_fitness = np.allclose(
            self.population_fitness, other.population_fitness
        )
        same_idx_best_solution = int(self.idx_best_solution) == int(
            other.idx_best_solution
        )

        same = (
            same_it
            and same_population
            and same_population_fitness
            and same_idx_best_solution
        )
        return same
