def installed():
    import os
    from ase.test import NotAvailable
    vcmd = os.getenv('VASP_COMMAND')
    vscr = os.getenv('VASP_SCRIPT')
    if vcmd == None and vscr == None:
        raise NotAvailable('Neither VASP_COMMAND nor VASP_SCRIPT defined')
    return True


def installed2():
    # Check if env variables exist for Vasp2
    import os
    from ase.test import NotAvailable
    vcmd = os.getenv('VASP_COMMAND')
    vscr = os.getenv('VASP_SCRIPT')
    vase = os.getenv('ASE_VASP_COMMAND')
    if vcmd is None and vscr is None and vase is None:
        raise NotAvailable('Neither ASE_VASP_COMMAND, VASP_COMMAND nor VASP_SCRIPT defined')
    return True
