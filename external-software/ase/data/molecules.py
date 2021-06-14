def latex(name):
    """Convert name to LaTeX"""
    s = '$'
    last = False
    for i in name:
        if i.isalpha():
            if not last:
                s = s + r'\rm{'
                last = True
        elif last:
            s = s + '}'
            last = False
        s = s + i
    if i.isalpha():
        s = s + '}'
    s = s.replace(' ', r'\ ') + '$'
    return s


def rest(name):
    """Convert name to reStructuredText."""
    s = ''
    while name:
        c = name[0]
        if c == '_':
            s += r'\ :sub:`%s`\ ' % name[1]
            name = name[2:]
        elif c == '^':
            s += r'\ :sup:`%s`\ ' % name[1]
            name = name[2:]
        else:
            s += c
            name = name[1:]
    return s
