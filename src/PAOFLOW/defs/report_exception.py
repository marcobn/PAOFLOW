def report_exception():
    import sys
    import traceback

    etype, evalue, etb = sys.exc_info()
    print('Exception: ', etype)
    print(evalue, flush=True)
    traceback.print_tb(etb)
