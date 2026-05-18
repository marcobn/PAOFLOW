def report_exception():
    """Print the current exception type, message, and traceback to standard output.

    Returns
    -------
    None

    Notes
    -----
    This function is intended to be called inside an ``except`` block.  It
    retrieves the active exception via :func:`sys.exc_info` and prints the
    exception class, value, and full traceback.  It is used throughout PAOFLOW
    as a lightweight diagnostic alternative to ``logging`` when an exception
    must be reported without suppressing the calling code flow.
    """
    import sys
    import traceback

    etype, evalue, etb = sys.exc_info()
    print('Exception: ', etype)
    print(evalue, flush=True)
    traceback.print_tb(etb)
