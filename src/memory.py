# pylint: disable = (missing-module-docstring)

from nvitop import Device, ResourceMetricCollector

initial_memory_used = None
global_max_memory_used = 0.0

# Record metrics to the logger in background every 5 seconds.
# It will collect 5-second mean/min/max for each metric.
def vram_monitor_factory(interval: float, device: str):
    def on_collect(metrics):  # will be called periodically
        global global_max_memory_used
        global initial_memory_used
        for k, v in metrics.items():
            if "memory_used" in k and "max" in k and device in k:
                max_memory_used = v
                if initial_memory_used is None:
                    initial_memory_used = max_memory_used
                elif max_memory_used > global_max_memory_used:
                    global_max_memory_used = max_memory_used
        return True

    ResourceMetricCollector(Device.cuda.all()).daemonize(
        on_collect=on_collect,
        interval=interval,
    )


def get_memory_info() -> float:
    value = global_max_memory_used
    if initial_memory_used is not None:
        value -= initial_memory_used

    # print(f"max_memory_used: \t {value:.3f} MB")
    return value
