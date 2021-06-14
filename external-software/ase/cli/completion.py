from __future__ import print_function
import os
import sys

# Path of the complete.py script:
my_dir, _ = os.path.split(os.path.realpath(__file__))
filename = os.path.join(my_dir, 'complete.py')


class CLICommand:
    """Add tab-completion for Bash.

    Will show the command that needs to be added to your '~/.bashrc file.
    """
    cmd = ('complete -o default -C "{py} {filename}" ase'
           .format(py=sys.executable, filename=filename))

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        cmd = CLICommand.cmd
        print(cmd)


def update(filename, commands):
    """Update commands dict.

    Run this when ever options are changed::

        python3 -m ase.cli.complete

    """

    import textwrap
    from ase.utils import import_module

    dct = {}  # type: Dict[str, List[str]]

    class Subparser:
        def __init__(self, command):
            self.command = command
            dct[command] = []

        def add_argument(self, *args, **kwargs):
            dct[command].extend(arg for arg in args
                                if arg.startswith('-'))

        def add_mutually_exclusive_group(self, required=False):
            return self

    for command, module_name in commands:
        module = import_module(module_name)
        module.CLICommand.add_arguments(Subparser(command))

    txt = 'commands = {'
    for command, opts in sorted(dct.items()):
        txt += "\n    '" + command + "':\n        ["
        if opts:
            txt += '\n'.join(textwrap.wrap("'" + "', '".join(opts) + "'],",
                                           width=65,
                                           break_on_hyphens=False,
                                           subsequent_indent='         '))
        else:
            txt += '],'
    txt = txt[:-1] + '}\n'
    with open(filename) as fd:
        lines = fd.readlines()
        a = lines.index('# Beginning of computer generated data:\n')
        b = lines.index('# End of computer generated data\n')
    lines[a + 1:b] = [txt]
    with open(filename + '.new', 'w') as fd:
        print(''.join(lines), end='', file=fd)
    os.rename(filename + '.new', filename)
    os.chmod(filename, 0o775)


if __name__ == '__main__':
    from ase.cli.main import commands
    update(filename, commands)
