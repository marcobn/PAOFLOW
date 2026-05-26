# def header():
#     """Print the PAOFLOW ASCII banner to standard output.

#     Returns
#     -------
#     None
#     """
#     print('')
#     print(
#         '#############################################################################################'
#     )
#     print(
#         '#                                                                                           #'
#     )
#     print(
#         '#                                          PAOFLOW                                          #'
#     )
#     print(
#         '#                                                                                           #'
#     )
#     print(
#         '#                  Utility to construct and operate on Hamiltonians from                    #'
#     )
#     print(
#         '#                 the Projections of DFT wfc on Atomic Orbital bases (PAO)                  #'
#     )
#     print(
#         '#                                                                                           #'
#     )
#     print(
#         '#############################################################################################\n'
#     )
def header(style='large'):
    """Print the PAOFLOW logo.

    Parameters
    ----------
    style : str
        'large' for the full letter-art logo (default), 'small' for the
        compact self-filled logo.
    """
    large = (
        'PPPPPP\\   AA\\AAAAA\\   OOOOOO\\  FFFFFFFF\\ LLL\\        OOOOOO\\  WW\\   WW\\   WW\\ \n'
        'PP  __PP\\ AA  __AA\\  OO  __OO\\ FF  ____/ L  |       OO  __OO\\ W  |  W  |  W  |\n'
        'PP /  P  |AA /  AA | OO /  OO |FF /      L  |       OO /  OO |W  |  W  |  W  |\n'
        'PPPPPPP  |AAAAAAAA | OO |  OO |FFFFFF    L  |       OO |  OO |W  |  W  |  W  |\n'
        'PP  ____/ AA  __AA | OO |  OO |FF  _/    L  |       OO |  OO |W |  W |  W | \n'
        'PP /      AA /  AA | OO \\__OO |FF /      L  |       OO \\__OO |W |  W |  W | \n'
        'PP/       AA/   AA |  \\OOOOO   |FF/       LLLLLLLL\\  \\OOOOO   |\\WWWWWWWWWW /  \n'
        '\\__|      \\__|  \\__|   \\______/ \\__|      \\________|  \\______/  \\_________/   \n'
    )
    small = (
        'ppppp     aa     oooooo  ffff  l     oooooo  ww        ww\n'
        'p  pp    aaaa    oo  oo  ffff  l     oo  oo   ww  ww  ww\n'
        'ppppp   aaaaaa   oo  oo  f     l     oo  oo    wwwwwwww\n'
        'p      aa    aa  oooooo  f     llll  oooooo     ww  ww\n'
    )
    logo = large if style == 'large' else small
    print('\n' + logo)
