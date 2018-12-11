#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#


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
    #from .defs.module_prerequisites import key_error_strings
    from .defs.module_prerequisites import report_pre_reqs,module_pre_reqs

    etype, evalue, etb = sys.exc_info()
    print('Exception: ', etype)
    print(evalue)
    traceback.print_tb(etb)

    if etype is KeyError:
      pre_reqs = module_pre_reqs[mname] if mname in module_pre_reqs else '<%s>'%mname

      if len(pre_reqs) > 1:
        pr_str = ', '.join(pre_reqs[:-1]) + ' and %s'%pre_reqs[-1]
      else:
        pr_str = pre_reqs[0]

      print('')
      print(report_pre_reqs%(pr_str,mname))
