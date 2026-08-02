def header(style):
    """Print the PAOFLOW logo.

    Parameters
    ----------
    style : str
        'color'   full letter-art logo in plain text (default).
        'small'   minimal PAOFLOW title
        if style == None, no title is printed
    """
    large = """
        ██████╗  █████╗  ██████╗ ███████╗██╗      ██████╗ ██╗    ██╗
        ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║     ██╔═══██╗██║    ██║
        ██████╔╝███████║██║   ██║█████╗  ██║     ██║   ██║██║ █╗ ██║
        ██╔═══╝ ██╔══██║██║   ██║██╔══╝  ██║     ██║   ██║██║███╗██║
        ██║     ██║  ██║╚██████╔╝██║     ███████╗╚██████╔╝╚███╔███╔╝
        ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
        """

    if style == 'color':
        _GREEN = '\033[38;2;0;133;62m'  # UNT green  #00853E  (separators)
        _LIME = '\033[38;2;74;222;128m'  # bright lime #4ade80 (logo text)
        _MINT = '\033[38;2;134;239;172m'  # subtitle   #86efac
        _RESET = '\033[0m'
        _WIDTH = 84
        _SEP = _GREEN + '\u2500' * _WIDTH + _RESET
        _SEP2 = _GREEN + '\u2500' * _WIDTH + _RESET
        _SUBTITLE = 'From DFT wavefunctions to materials properties via atomic-orbital Hamiltonians'

        lines = large.rstrip('\n').split('\n')
        colored_lines = [_GREEN + '   ' + line + _RESET for line in lines]

        print()
        print(_SEP)
        print('\n'.join(colored_lines))
        print(_SEP2)
        print(_GREEN + _SUBTITLE.center(_WIDTH) + _RESET)
        print(_SEP)
        print()

    elif style == 'small':
        print("""
----------------------------------------------------------------------------------
                                      PAOFLOW
----------------------------------------------------------------------------------
        """)

    else:
        pass
