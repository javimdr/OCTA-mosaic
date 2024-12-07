from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

from octa_mosaic.modules import optimization_utils
from octa_mosaic.modules.experiments.procedure import Procedure, Report
from octa_mosaic.modules.mosaico import Mosaico
from octa_mosaic.modules.optimization.differential_evolution import (
    DifferentialEvolutionParams,
    differential_evolution_from_params,
)
from octa_mosaic.modules.optimization.optimize_result import OptimizeResult


class DEProcess(Procedure):
    def _execution(
        self,
        mosaic: Mosaico,
        fobj: Callable,
        fobj_args: Sequence[Any],
        bounds: np.ndarray,
        de_params: DifferentialEvolutionParams,
        initial_population: Optional[np.ndarray] = None,
    ) -> Tuple[Mosaico, Report]:

        solution = differential_evolution_from_params(
            de_params, bounds, fobj, fobj_args, initial_population
        )

        mosaic_solution = optimization_utils.individual_to_mosaico(solution.x, mosaic)
        report = self.generate_report(solution)
        return mosaic_solution, report

    def generate_report(
        self,
        de_solution: OptimizeResult,
    ) -> Report:
        report = {
            "result": {
                "x": [float(str(v)) for v in de_solution.x],
                "fitness": float(str(de_solution.fitness)),
                "message": de_solution.message,
                "nits": de_solution.nits,
            }
        }

        return report


def minimization_fobj(x: np.ndarray, maximization_fobj: Callable, *args: Tuple) -> float:
    fitness = maximization_fobj(x, *args)
    return -fitness


class ScipyOptimizeProcedure(Procedure):
    def _execution(
        self,
        method: str,
        x0: np.ndarray,
        fobj: Callable,
        fobj_args: Sequence[Any] = (),
        **optimize_kwargs,
    ) -> Tuple[Mosaico, Report]:

        scipy_result = scipy.optimize.minimize(
            fun=minimization_fobj,
            x0=x0,
            args=(fobj, *fobj_args),
            method=method,
            **optimize_kwargs,
        )

        mosaic_solution = optimization_utils.individual_to_mosaico(
            scipy_result.x, fobj_args[0]
        )

        optimize_result = OptimizeResult(
            x=scipy_result.x,
            fitness=-scipy_result.fun,
            message=scipy_result.get("message", "Unknown."),
            nits=scipy_result.get("nit", -1),
        )

        report = self.generate_report(optimize_result, method)
        return mosaic_solution, report

    def generate_report(self, de_solution: OptimizeResult, method: str) -> Report:
        report = {
            "optimize_method": method,
            "result": {
                "x": [float(str(v)) for v in de_solution.x],
                "fitness": float(str(de_solution.fitness)),
                "message": de_solution.message,
                "nits": de_solution.nits,
            },
        }

        return report
