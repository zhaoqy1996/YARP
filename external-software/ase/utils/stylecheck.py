# Check coding style compliance.
#
# For a description of error codes see:
#
#     http://pep8.readthedocs.org/en/latest/intro.html#error-codes

import argparse
import os
import smtplib
import subprocess
from email.mime.text import MIMEText


def mail(to, subject, txt):
    msg = MIMEText(txt)
    msg['Subject'] = subject
    msg['From'] = 'pep8@fysik.dtu.dk'
    msg['To'] = to
    s = smtplib.SMTP('mail.fysik.dtu.dk')
    s.sendmail(msg['From'], [to], msg.as_string())
    s.quit()


p8 = 'pep8 --ignore W293,E129'


def pep8(name):
    if not os.path.isfile(name):
        return []
    try:
        output = subprocess.check_output(p8 + ' ' + name, shell=True)
    except subprocess.CalledProcessError as ex:
        output = ex.output
    lines = []
    for line in output.decode().splitlines():
        name, l, c, error = line.split(':', 3)
        # Allow 'a**b' while still disallowing 'a+b':
        if error.startswith(' E225') or error.startswith(' E226'):
            ln = open(name).readlines()[int(l) - 1]
            c = int(c) - 1
            if ln[c:c + 2] == '**':
                continue
        lines.append(line)
    return lines
        

def pyflakes(name):
    try:
        output = subprocess.check_output('pyflakes ' + name, shell=True)
    except subprocess.CalledProcessError as ex:
        output = ex.output
    return [line for line in output.decode().splitlines()
            if 'list comprehension redefines' not in line]


def check_file(name):
    for line in pep8(name):
        print(line)
    for line in pyflakes(name):
        print(line)
    
        
grrr = """Please always run this check on Python source-code before committing:

    $ alias check="python -m ase.utils.stylecheck"
    $ check foo.py bar.py ...
    
This will run pep8 and pyflakes on you source.  Install pep8 and pyflakes
like this:
    
    $ pip install pep8 pyflakes
    
"""


def check_repository(to):
    output = subprocess.check_output('svn merge --dry-run -r BASE:HEAD .',
                                     shell=True)

    lines = output.decode().splitlines()
    names = []
    for line in lines:
        st, name = line.split()[:2]
        if st in ['U', 'A'] and name.endswith('.py'):
            names.append(name)
    
    warnings = {}
    for name in names:
        w = pep8(name)
        warnings[name] = len(w)
    
    if names:
        subprocess.call('svn up > up.out', shell=True)
    
    n81 = 0
    n82 = 0
    nf = 0
    txt = []
    for name in names:
        w = pep8(name)
        n1 = warnings[name]
        n2 = len(w)
        if n2 > n1:
            n81 += n1
            n82 += n2
            txt.append('Number of PEP-8 errors increased from {} to {}:'
                       .format(n1, n2))
            for x in w:
                txt.append(x)
            txt.append('')
            
        q = pyflakes(name)
        if q:
            nf += len(q)
            txt.append('Warnings from PyFlakes:')
            txt += q
            txt.append('')
    
    if txt:
        subject = []
        if n82 > n81:
            subject.append(
                'PEP8 warnings increased from {} to {}'.format(n81, n82))
        if nf:
            subject.append('PyFlakes warnings: {}'.format(nf))
        txt = '\n'.join(txt)
        if to:
            mail(to, ' - '.join(subject), grrr + txt)
        else:
            print(txt)
        

parser = argparse.ArgumentParser(description='Run both pep8 and pyflakes '
                                 'on file(s).')
parser.add_argument('--check-repository', action='store_true')
parser.add_argument('--mail')
parser.add_argument('filenames', nargs='*', metavar='filename')
args = parser.parse_args()
if args.check_repository:
    check_repository(args.mail)
else:
    for name in args.filenames:
        check_file(name)
