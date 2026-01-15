from mpi4py import MPI


def write_header(msg: str) -> None:
    """
    Print out the given header message `msg` from rank 0 only.

    Parameters
    ----------
    `msg` : str
        Header message to be printed. Must be shorter than 66 characters.

    Raises
    ------
    ValueError
        If the message exceeds 66 characters.
    """
    if len(msg) >= 66:
        raise ValueError(f"Message longer than 66 characters: {msg}")

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        return

    separator = "=" * 70
    print(f"  {separator}")
    print(f"  =  {msg:^66s}=")
    print(f"  {separator}")


def headered_function(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            write_header(name)
            return func(*args, **kwargs)

        return wrapper

    return decorator
