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
        'PP  __PP\\ AA  __$$\\  OO  __OO\\ FF  ____/ L  |       OO  __OO\\ W  |  W  |  W  |\n'
        'PP /  P  |AA /  @  | OO /  O  |FF /      L  |       OO /  O  |W  |  W  |  W  |\n'
        'PPPPPPP  |AAAAAAAA | OO |  O  |FFFFFF    L  |       OO |  O  |W  |  W  |  W  |\n'
        'PP  ____/ AA  __AA | OO |  O  |FF  _/    L  |       OO |  O  |W |  W |  W | \n'
        'PP /      AA /  A  | OO \\__O  |FF /      L  |       OO \\__O  |W |  W |  W | \n'
        'PP/       AA/   A  |  \\OOOOOO  |FF/       LLLLLLLL\\  \\OOOOOO  |\\WWWWWWWWWW /  \n'
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
