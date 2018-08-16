import sys
import traceback

def report_exception ( ):
    etype, evalue, etb = sys.exc_info()
    print('Exception: ', etype)
    print(evalue)
    traceback.print_tb(etb)
 
