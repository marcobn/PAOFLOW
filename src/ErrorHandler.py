#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
#F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli, Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .


class ErrorHandler:

  def __init__ ( self ):
    pass

  def report_exception ( self, mname='UNKNOWN' ):
    '''
    Print the exception and traceback. Print suggestion if the error is recognized as a user mistake.

    Arguments:
        mname (str): Module name (name of function in PAOFLOW)

    Returns:
        None
    '''
    import sys
    import traceback
    from .defs.module_prerequisites import key_error_strings,report_pre_reqs,module_pre_reqs

    etype, evalue, etb = sys.exc_info()
    print('Exception: ', etype, evalue, flush=True)
    traceback.print_tb(etb)

    if etype is KeyError:

      if mname in module_pre_reqs:
        print('HHH',mname, type(mname))
        pre_reqs = module_pre_reqs[mname]
        if len(pre_reqs) > 1:
          pr_str = ', '.join(pre_reqs[:-1]) + ' and %s'%pre_reqs[-1]
        else:
          pr_str = pre_reqs[0]

        print('')
        print(report_pre_reqs%(pr_str,mname), flush=True)

      if evalue.args[0] in key_error_strings:
        print('')
        print('SUGGESTION: %s\n'%key_error_strings[evalue.args[0]], flush=True)
