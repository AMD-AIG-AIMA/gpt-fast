import time
from functools import wraps
from contextlib import ContextDecorator
from collections import defaultdict
import atexit
import torch

from tp import _get_rank


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    # size_all_mb = (param_size + buffer_size) / 1024**2
    return param_size + buffer_size

class TimeProfiler(ContextDecorator):
    _timings = defaultdict(list)
    _bw_utilizations = defaultdict(list)
    _num_warmup = 0
    _rank = _get_rank()
    
    def __init__(self, name, model_size=None, peak_bandwidth=None):
        self.name = name
        self.start_time = None
        self.peak_bandwidth = peak_bandwidth
        self.tokens_processed = None
        self.model_size = model_size
        

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        elapsed_time = time.time() - self.start_time
        self.__class__._timings[self.name].append(elapsed_time)
        
        if self.tokens_processed is not None and self.model_size is not None and self.peak_bandwidth is not None:
            tokens_per_second = self.tokens_processed / elapsed_time
            bw_utilization = (tokens_per_second * self.model_size) / self.peak_bandwidth
            self.__class__._bw_utilizations[self.name].append(bw_utilization)
        
        self.tokens_processed = None  # Reset for next use
        return False

    def set_tokens_processed(self, tokens):
        self.tokens_processed = tokens

    @classmethod
    def set_warm_up(cls, num_warmup):
        cls._num_warmup = num_warmup

    @classmethod
    def print_report(cls):
        if cls._rank == 0 or cls._rank is None:
            print("\nTime Profiling Report:")
            print("--------------------------")
            for name in set(list(cls._timings.keys()) + list(cls._bw_utilizations.keys())):
                times = cls._timings[name]
                bw_utils = cls._bw_utilizations[name]
                
                if len(times) <= cls._num_warmup:
                    continue

                valid_times = times[cls._num_warmup:]
                avg_time = sum(valid_times) / len(valid_times)
                total_time = sum(valid_times)
                calls = len(valid_times)
                
                print(f"{name}:")
                print(f"  Total time: {total_time:.6f} seconds")
                print(f"  Average time: {avg_time:.6f} seconds")
                print(f"  Number of calls: {calls}")
                print(f"  Warm-up calls: {cls._num_warmup}")
                
                if bw_utils:
                    valid_bw_utils = bw_utils[cls._num_warmup:]
                    avg_bw_util = sum(valid_bw_utils) / len(valid_bw_utils)
                    print(f"  Average Bandwidth Utilization: {avg_bw_util:.6f}")
                
                print()

def timer_profile(func=None, *, name=None, model_size=None, peak_bandwidth=None):
    if func is None:
        return lambda f: timer_profile(f, name=name, model_size=model_size, peak_bandwidth=peak_bandwidth)

    @wraps(func)
    def wrapper(*args, **kwargs):
        with TimeProfiler(name or func.__name__, model_size, peak_bandwidth) as profiler:
            result = func(*args, **kwargs)
            if hasattr(result, 'num_tokens'):
                profiler.set_tokens_processed(result.num_tokens)
            return result

    return wrapper

# Register the print_report function to be called at program exit
atexit.register(TimeProfiler.print_report)