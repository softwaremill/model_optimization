# pylint: disable = (missing-module-docstring)

from nvitop import ResourceMetricCollector, Device


global_max_memory_used = 0.0

# Record metrics to the logger in background every 5 seconds.
# It will collect 5-second mean/min/max for each metric.
def vram_monitor_factory(interval: float, device: str):
    def on_collect(metrics):  # will be called periodically
        global global_max_memory_used
        for k, v in metrics.items():
            if "memory_used" in k and "max" in k and device in k:
                max_memory_used = v
                if max_memory_used > global_max_memory_used:
                    global_max_memory_used = max_memory_used
        return True

    ResourceMetricCollector(Device.cuda.all()).daemonize(
        on_collect=on_collect,
        interval=interval,
    )

def print_memory_info() -> None:
    print(f"max_memory_used: \t {global_max_memory_used:.3f} MB")
    print("")
