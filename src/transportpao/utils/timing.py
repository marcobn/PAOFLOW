from time import perf_counter
from collections import OrderedDict
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.rank


class Clock:
    """
    A high-resolution timing utility for measuring execution durations.

    Parameters
    ----------
    name : str
        Descriptive name of the timed routine or block.

    Attributes
    ----------
    name : str
        Name of the routine being timed.
    call_count : int
        Number of times the timer was started and stopped.
    total_time : float
        Total time accumulated over all calls in seconds.
    _start_time : float or None
        Internal timestamp when the timer was last started.

    Notes
    -----
    Uses `perf_counter` for high-precision timing.
    """

    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self._start_time = None

    def start(self) -> None:
        """
        Start the timer. Raises an error if already started.
        """
        if self._start_time is not None:
            raise RuntimeError(f"Clock {self.name} is already running.")
        self._start_time = perf_counter()
        self.call_count += 1

    def stop(self) -> None:
        """
        Stop the timer and update total accumulated time.
        Raises an error if the timer was never started.
        """
        if self._start_time is None:
            raise RuntimeError(f"Clock {self.name} was not started.")
        elapsed = perf_counter() - self._start_time
        self.total_time += elapsed
        self._start_time = None

    def avg_time(self) -> float:
        """
        Returns
        -------
        float
            Average time per call in seconds. Returns 0.0 if never called.
        """
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    def time_upto_now(self) -> float:
        """
        Returns
        -------
        float
            Time since last start (not included in total_time yet).
            Returns 0.0 if timer is not running.
        """
        if self._start_time is None:
            return 0.0
        return perf_counter() - self._start_time


class TimingManager:
    """
    A timing manager that tracks multiple named clocks.

    Attributes
    ----------
    clocks : OrderedDict
        Dictionary of active clocks indexed by name.

    Notes
    -----
    Only rank 0 outputs timing reports to avoid duplicate MPI prints.
    """

    def __init__(self):
        self.clocks = OrderedDict()

    def start(self, name: str) -> None:
        """
        Start a named timer.

        Parameters
        ----------
        name : str
            Identifier for the routine being timed.
        """
        clock = self.clocks.setdefault(name, Clock(name))
        clock.start()

    def stop(self, name: str) -> None:
        """
        Stop a named timer.

        Parameters
        ----------
        name : str
            Identifier for the routine being timed.

        Raises
        ------
        ValueError
            If the clock was never started.
        """
        if name not in self.clocks:
            raise ValueError(f"No clock with name '{name}' was started.")
        self.clocks[name].stop()

    def report(self, header: str = "<global routines>") -> None:
        """
        Print a report of all clocked timings.

        Parameters
        ----------
        header : str
            Optional header for the printed report.
        """
        if rank != 0:
            return

        print()
        print(f"{header:>10}")
        print(f"{'':13}clock number : {len(self.clocks):5}")
        print()
        for clock in self.clocks.values():
            time_s = clock.total_time
            calls = clock.call_count
            avg = clock.avg_time()
            if calls == 1:
                print(f"{clock.name:>20} : {time_s:8.2f}s CPU")
            else:
                print(
                    f"{clock.name:>20} : {time_s:8.2f}s CPU ({calls:8d} calls,{avg:8.3f} s avg)"
                )
        print()

    def timing_upto_now(self, name: str, label: str = None) -> None:
        """
        Print time elapsed since a given clock was started (not yet stopped).

        Parameters
        ----------
        name : str
            Clock name to inspect.
        label : str, optional
            Custom label to print. Defaults to clock name.

        Notes
        -----
        This does not stop the clock; it's purely diagnostic.
        """
        if rank != 0 or name not in self.clocks:
            return
        clock = self.clocks[name]
        elapsed = clock.time_upto_now()
        if elapsed > 0.0:
            print(f"    {label or name:>20} : {elapsed:8.2f}s")
            print()


global_timing = TimingManager()


def timed_function(name: str = None):
    """
    Decorator to automatically time a function using global_timing.

    Parameters
    ----------
    name : str, optional
        Custom name for the timer. If None, the function's __name__ is used.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            global_timing.start(timer_name)
            try:
                return func(*args, **kwargs)
            finally:
                global_timing.stop(timer_name)

        return wrapper

    return decorator
