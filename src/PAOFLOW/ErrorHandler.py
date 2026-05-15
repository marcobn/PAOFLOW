class ErrorHandler:
    def __init__(self):
        pass

    def report_exception(self, mname='UNKNOWN'):
        """
        Print the exception and traceback. Print suggestion if the error is recognized as a user mistake.

        Arguments:
            mname (str): Module name (name of function in PAOFLOW)

        Returns:
            None
        """
        import sys
        import traceback

        from .defs.module_prerequisites import key_error_strings, module_pre_reqs, report_pre_reqs

        etype, evalue, etb = sys.exc_info()
        print('Exception: ', etype, evalue, flush=True)
        traceback.print_tb(etb)

        if etype is KeyError:
            if mname in module_pre_reqs:
                print('HHH', mname, type(mname))
                pre_reqs = module_pre_reqs[mname]
                if len(pre_reqs) > 1:
                    pr_str = ', '.join(pre_reqs[:-1]) + ' and %s' % pre_reqs[-1]
                else:
                    pr_str = pre_reqs[0]

                print('')
                print(report_pre_reqs % (pr_str, mname), flush=True)

            if evalue.args[0] in key_error_strings:
                print('')
                print('SUGGESTION: %s\n' % key_error_strings[evalue.args[0]], flush=True)
