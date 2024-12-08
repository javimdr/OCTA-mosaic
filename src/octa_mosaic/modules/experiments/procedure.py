import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from octa_mosaic.mosaic.mosaic import Mosaic

Report = Dict[str, Any]
ProcedureReport = Dict[str, Report]


class Procedure(ABC):
    def __init__(self, name_id: str):
        self.name_id = name_id

    @abstractmethod
    def _execution(self, x: Any, *args, **kargs) -> Tuple[Mosaic, Report]:
        """Execute the procedure and return the result and a report."""

    def run(self, x: Any, *args, **kargs) -> Tuple[Any, ProcedureReport]:
        start_time = time.perf_counter()
        y, process_report = self._execution(x, *args, **kargs)
        end_time = time.perf_counter()

        process_report["execution_time"] = end_time - start_time
        report = {self.name_id: process_report}
        return y, report
