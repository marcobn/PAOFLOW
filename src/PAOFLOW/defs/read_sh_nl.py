def read_sh_nl(data_controller):
    # reads in shelks from pseudo files
    from os.path import join

    import numpy as np

    #    arry,attr = data_controller.data_dicts()
    arry = data_controller.data_arrays
    attr = data_controller.data_attributes

    # Get Shells for each species
    sdict = {}
    jchid = {}
    jchia = None

    for s in arry['species']:
        sdict[s[0]], jchid[s[0]] = read_pseudopotential(
            join(attr['workpath'], attr['savedir'], s[1])
        )

    for s, p in sdict.items():
        tmp_list = []
        tmp_list_chi = []
        for o in range(len(p)):
            tmp_list_chi.append(jchid[s][o])
            tmp_list.append(p[o])
            # if l=0 include it twice
            if p[o] == 0 or len(jchid[s]) == 0:
                tmp_list_chi.append(jchid[s][o])
                tmp_list.append(p[o])

            # when using scalar rel pseido with spin orb..
            if len(jchid[s]) == 0:
                tmp = []
                for o in sdict[s][::2]:
                    if o == 0:
                        tmp.extend([0.5])
                    if o == 1:
                        tmp.extend([0.5, 1.5])
                    if o == 2:
                        tmp.extend([1.5, 2.5])
                    if o == 3:
                        tmp.extend([2.5, 3.5])
                jchid[s] = np.array(tmp)

        sdict[s] = np.array(tmp_list)
        jchid[s] = np.array(tmp_list_chi)

    # value of l
    shell = np.hstack([sdict[a] for a in arry['atoms']])
    jchia = np.hstack([jchid[a] for a in arry['atoms']])

    return shell[::2], jchia[::2]


############################################################################################
############################################################################################
############################################################################################


def read_pseudopotential(fpp):
    """
    Reads a psuedopotential file to determine the included shells and occupations.

    Arguments:
        fnscf (string) - Filename of the pseudopotential, copied to the .save directory

    Returns:
        sh, nl (lists) - sh is a list of orbitals (s-0, p-1, d-2, etc)
                         nl is a list of occupations at each site
        sh and nl are representative of one atom only
    """

    import re
    import xml.etree.cElementTree as ET
    from tempfile import NamedTemporaryFile

    import numpy as np

    sh = []
    jchi = []

    # clean xnl before reading
    with open(fpp) as ifo:
        temp_str = ifo.read()

    temp_str = re.sub('&', ' ', temp_str)
    f = NamedTemporaryFile(mode='w', delete=True)
    f.write(temp_str)

    try:
        iterator_obj = ET.iterparse(f.name, events=('start', 'end'))
        iterator = iter(iterator_obj)
        event, root = next(iterator)

        for event, elem in iterator:
            try:
                for i in elem.findall('PP_PSWFC/'):
                    sh.append(int(i.attrib['l']))
            except Exception:
                pass

        sh = np.array(sh)

    except Exception:
        with open(fpp) as ifo:
            ifs = ifo.read()
        res = re.findall('(.*)\s*Wavefunction', ifs)[1:]
        sh = np.array(list(map(int, list([x.split()[1] for x in res]))))

    for i in elem.findall('PP_SPIN_ORB/'):
        try:
            jchi.append(float(i.attrib['jchi']))
        except:
            pass

    jchi = np.array(jchi)

    return sh, jchi
