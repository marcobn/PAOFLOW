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
def header(style='color'):
    """Print the PAOFLOW logo.

    Parameters
    ----------
    style : str
        'large'    – full letter-art logo in plain text (default).
        'small'    – compact self-filled logo in plain text.
        'color'    – large logo in UNT green with decorative separators and subtitle (requires a 24-bit colour terminal).
        'markdown' – large logo wrapped in a fenced code block (README / Sphinx).
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

    if style == 'markdown':
        print('\n```\n' + large + '```\n')

    elif style == 'color':
        _GREEN = '\033[38;2;0;133;62m'  # UNT green  #00853E  (separators)
        _LIME = '\033[38;2;74;222;128m'  # bright lime #4ade80 (logo text)
        _MINT = '\033[38;2;134;239;172m'  # subtitle   #86efac
        _RESET = '\033[0m'
        _WIDTH = 84
        _SEP = _GREEN + '\u2500' * _WIDTH + _RESET
        _SEP2 = _LIME + '\u2500' * _WIDTH + _RESET
        _SUBTITLE = 'From DFT wavefunctions to materials properties via atomic-orbital Hamiltonians'

        lines = large.rstrip('\n').split('\n')
        colored_lines = [_LIME + '   ' + line + _RESET for line in lines]

        print()
        print(_SEP)
        print('\n'.join(colored_lines))
        print(_SEP2)
        print(_MINT + _SUBTITLE.center(_WIDTH) + _RESET)
        print(_SEP)
        print()

    else:
        logo = large if style == 'large' else small
        print('\n' + logo)
