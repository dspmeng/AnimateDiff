from codetiming import Timer
import torch
import math


class FancyTimer(Timer):
    start_prof = False

    def __init__(self, name, logger=None):
        super().__init__(name, logger=logger)
        self.ticking = False
        self.cuda = torch.cuda.is_available()

    def tick(self):
        if FancyTimer.start_prof:
            self.ticking = True
            if self.cuda: torch.cuda.nvtx.range_push(self.name)
            self.start()

    def tock(self):
        if FancyTimer.start_prof and self.ticking:
            if self.cuda: torch.cuda.synchronize()
            self.stop()
            if self.cuda: torch.cuda.nvtx.range_pop()
            self.ticking = False

    def start_profiling():
        FancyTimer.start_prof = True
        if torch.cuda.is_available(): torch.cuda.cudart().cudaProfilerStart()

    def stop_profiling():
        FancyTimer.start_prof = False
        if torch.cuda.is_available(): torch.cuda.cudart().cudaProfilerStop()

