"""Build ASE's web-page.

Initial setup::

    cd ~
    python3 -m venv ase-web-page
    cd ase-web-page
    . bin/activate
    pip install sphinx-rtd-theme
    pip install Sphinx
    pip install matplotlib scipy flask
    git clone http://gitlab.com/ase/ase.git
    cd ase
    pip install -U .

Crontab::

    WEB_PAGE_FOLDER=...
    CMD="python -m ase.utils.build_web_page"
    10 19 * * * cd ~/ase-web-page; . bin/activate; cd ase; $CMD > ../ase.log

"""

from __future__ import print_function
import os
import subprocess
import sys

from ase import __version__


cmds = """\
touch ../ase-web-page.lock
pip install --upgrade pip
git clean -fdx
git checkout web-page -q
git pull -q > /dev/null 2>&1
pip install .
cd doc; sphinx-build -b html -d build/doctrees . build/html
mv doc/build/html ase-web-page
git clean -fdx doc
git checkout master -q
git pull -q > /dev/null 2>&1
pip install .
cd doc; sphinx-build -b html -d build/doctrees . build/html
mv doc/build/html ase-web-page/dev
python setup.py sdist
cp dist/ase-*.tar.gz ase-web-page/
cp dist/ase-*.tar.gz ase-web-page/dev/
find ase-web-page -name install.html | xargs sed -i s/snapshot.tar.gz/{0}/g
tar -czf ase-web-page.tar.gz ase-web-page
cp ase-web-page.tar.gz {1}/tmp-ase-web-page.tar.gz
mv {1}/tmp-ase-web-page.tar.gz {1}/ase-web-page.tar.gz"""

cmds = cmds.format('ase-' + __version__ + '.tar.gz',
                   os.environ['WEB_PAGE_FOLDER'])


def build():
    if os.path.isfile('../ase-web-page.lock'):
        print('Locked', file=sys.stderr)
        return
    try:
        for cmd in cmds.splitlines():
            subprocess.check_call(cmd, shell=True)
    finally:
        os.remove('../ase-web-page.lock')


if __name__ == '__main__':
    build()
