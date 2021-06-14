from __future__ import print_function
import os
import sys
import subprocess
from multiprocessing import Process, cpu_count, Queue
import tempfile
import unittest
from glob import glob
from distutils.version import LooseVersion
import time
import traceback
import warnings

import numpy as np

from ase.calculators.calculator import names as calc_names, get_calculator
from ase.utils import devnull
from ase.cli.info import print_info

NotAvailable = unittest.SkipTest

test_calculator_names = []

if sys.version_info[0] == 2:
    class ResourceWarning(UserWarning):
        pass  # Placeholder - this warning does not exist in Py2 at all.


def require(calcname):
    if calcname not in test_calculator_names:
        raise NotAvailable('use --calculators={0} to enable'.format(calcname))


def get_tests(files=None):
    dirname, _ = os.path.split(__file__)
    if files:
        fnames = [os.path.join(dirname, f) for f in files]

        files = set()
        for fname in fnames:
            files.update(glob(fname))
        files = list(files)
    else:
        files = glob(os.path.join(dirname, '*'))
        files.remove(os.path.join(dirname, 'testsuite.py'))

    sdirtests = []  # tests from subdirectories: only one level assumed
    tests = []
    for f in files:
        if os.path.isdir(f):
            # add test subdirectories (like calculators)
            sdirtests.extend(glob(os.path.join(f, '*.py')))
        else:
            # add py files in testdir
            if f.endswith('.py'):
                tests.append(f)
    tests.sort()
    sdirtests.sort()
    tests.extend(sdirtests)  # run test subdirectories at the end
    tests = [os.path.relpath(test, dirname)
             for test in tests if not test.endswith('__.py')]
    return tests


def runtest_almost_no_magic(test):
    dirname, _ = os.path.split(__file__)
    path = os.path.join(dirname, test)
    # exclude some test for windows, not done automatic
    if os.name == 'nt':
        skip = [name for name in calc_names]
        skip += ['db_web', 'h2.py', 'bandgap.py', 'al.py',
                 'runpy.py', 'oi.py']
        if any(s in test for s in skip):
            raise NotAvailable('not on windows')
    try:
        with open(path) as fd:
            exec(compile(fd.read(), path, 'exec'), {})
    except ImportError as ex:
        module = ex.args[0].split()[-1].replace("'", '').split('.')[0]
        if module in ['scipy', 'matplotlib', 'Scientific', 'lxml', 'Tkinter',
                      'flask', 'gpaw', 'GPAW', 'netCDF4', 'psycopg2']:
            raise unittest.SkipTest('no {} module'.format(module))
        else:
            raise


def run_single_test(filename, verbose, strict):
    """Execute single test and return results as dictionary."""
    result = Result(name=filename)

    # Some tests may write to files with the same name as other tests.
    # Hence, create new subdir for each test:
    cwd = os.getcwd()
    testsubdir = filename.replace(os.sep, '_').replace('.', '_')
    os.mkdir(testsubdir)
    os.chdir(testsubdir)
    t1 = time.time()

    if not verbose:
        sys.stdout = devnull
    try:
        with warnings.catch_warnings():
            if strict:
                # We want all warnings to be errors.  Except some that are
                # normally entirely ignored by Python, and which we don't want
                # to bother about.
                warnings.filterwarnings('error')
                for warntype in [PendingDeprecationWarning, ImportWarning,
                                 ResourceWarning]:
                    warnings.filterwarnings('ignore', category=warntype)

            # This happens from matplotlib sometimes.
            # How can we allow matplotlib to import badly and yet keep
            # a higher standard for modules within our own codebase?
            warnings.filterwarnings('ignore',
                                    'Using or importing the ABCs from',
                                    category=DeprecationWarning)
            runtest_almost_no_magic(filename)
    except KeyboardInterrupt:
        raise
    except unittest.SkipTest as ex:
        result.status = 'SKIPPED'
        result.whyskipped = str(ex)
        result.exception = ex
    except AssertionError as ex:
        result.status = 'FAIL'
        result.exception = ex
        result.traceback = traceback.format_exc()
    except BaseException as ex:
        result.status = 'ERROR'
        result.exception = ex
        result.traceback = traceback.format_exc()
    else:
        result.status = 'OK'
    finally:
        sys.stdout = sys.__stdout__
        t2 = time.time()
        os.chdir(cwd)

    result.time = t2 - t1
    return result


class Result:
    """Represents the result of a test; for communicating between processes."""
    attributes = ['name', 'pid', 'exception', 'traceback', 'time', 'status',
                  'whyskipped']

    def __init__(self, **kwargs):
        d = {key: None for key in self.attributes}
        d['pid'] = os.getpid()
        for key in kwargs:
            assert key in d
            d[key] = kwargs[key]
        self.__dict__ = d


def runtests_subprocess(task_queue, result_queue, verbose, strict):
    """Main test loop to be called within subprocess."""

    try:
        while True:
            result = test = None

            test = task_queue.get()
            if test == 'no more tests':
                return

            # We need to run some tests on master:
            #  * doctest exceptions appear to be unpicklable.
            #    Probably they contain a reference to a module or something.
            #  * gui/run may deadlock for unknown reasons in subprocess

            t = test.replace('\\', '/')
            if t in ['bandstructure.py', 'doctests.py', 'gui/run.py',
                     'matplotlib_plot.py', 'fio/oi.py', 'fio/v_sim.py',
                     'fio/animate.py', 'db/db_web.py', 'x3d.py']:
                result = Result(name=test, status='please run on master')
                result_queue.put(result)
                continue

            result = run_single_test(test, verbose, strict)

            # Any subprocess that uses multithreading is unsafe in
            # subprocesses due to a fork() issue:
            #   https://gitlab.com/ase/ase/issues/244
            # Matplotlib uses multithreading and we must therefore make sure
            # that any test which imports matplotlib runs on master.
            # Hence check whether matplotlib was somehow imported:
            assert 'matplotlib' not in sys.modules, test
            result_queue.put(result)

    except KeyboardInterrupt:
        print('Worker pid={} interrupted by keyboard while {}'
              .format(os.getpid(),
                      'running ' + test if test else 'not running'))
    except BaseException as err:
        # Failure outside actual test -- i.e. internal test suite error.
        result = Result(pid=os.getpid(), name=test, exception=err,
                        traceback=traceback.format_exc(),
                        time=0.0, status='ABORT')
        result_queue.put(result)


def print_test_result(result):
    msg = result.status
    if msg == 'SKIPPED':
        msg = 'SKIPPED: {}'.format(result.whyskipped)
    print('{name:36} {time:6.2f}s {msg}'
          .format(name=result.name, time=result.time, msg=msg))
    if result.traceback:
        print('=' * 78)
        print('Error in {} on pid {}:'.format(result.name, result.pid))
        print(result.traceback.rstrip())
        print('=' * 78)


def runtests_parallel(nprocs, tests, verbose, strict):
    # Test names will be sent, and results received, into synchronized queues:
    task_queue = Queue()
    result_queue = Queue()

    for test in tests:
        task_queue.put(test)

    for i in range(nprocs):  # Each process needs to receive this
        task_queue.put('no more tests')

    procs = []
    try:
        # Start tasks:
        for i in range(nprocs):
            p = Process(target=runtests_subprocess,
                        name='ASE-test-worker-{}'.format(i),
                        args=[task_queue, result_queue, verbose, strict])
            procs.append(p)
            p.start()

        # Collect results:
        for i in range(len(tests)):
            if nprocs == 0:
                # No external workers so we do everything.
                task = task_queue.get()
                result = run_single_test(task, verbose, strict)
            else:
                result = result_queue.get()  # blocking call
                if result.status == 'please run on master':
                    result = run_single_test(result.name, verbose, strict)
            print_test_result(result)
            yield result

            if result.status == 'ABORT':
                raise RuntimeError('ABORT: Internal error in test suite')
    except KeyboardInterrupt:
        raise
    except BaseException:
        for proc in procs:
            proc.terminate()
        raise
    finally:
        for proc in procs:
            proc.join()


def summary(results):
    ntests = len(results)
    err = [r for r in results if r.status == 'ERROR']
    fail = [r for r in results if r.status == 'FAIL']
    skip = [r for r in results if r.status == 'SKIPPED']
    ok = [r for r in results if r.status == 'OK']

    if fail or err:
        print()
        print('Failures and errors:')
        for r in err + fail:
            print('{}: {}: {}'.format(r.name, r.exception.__class__.__name__,
                                      r.exception))

    print('========== Summary ==========')
    print('Number of tests   {:3d}'.format(ntests))
    print('Passes:           {:3d}'.format(len(ok)))
    print('Failures:         {:3d}'.format(len(fail)))
    print('Errors:           {:3d}'.format(len(err)))
    print('Skipped:          {:3d}'.format(len(skip)))
    print('=============================')

    if fail or err:
        print('Test suite failed!')
    else:
        print('Test suite passed!')


def test(calculators=[], jobs=0,
         stream=sys.stdout, files=None, verbose=False, strict=False):
    """Main test-runner for ASE."""

    if LooseVersion(np.__version__) >= '1.14':
        # Our doctests need this (spacegroup.py)
        np.set_printoptions(legacy='1.13')

    test_calculator_names.extend(calculators)
    disable_calculators([name for name in calc_names
                         if name not in calculators])

    tests = get_tests(files)
    if len(set(tests)) != len(tests):
        # Since testsubdirs are based on test name, we will get race
        # conditions on IO if the same test runs more than once.
        print('Error: One or more tests specified multiple times',
              file=sys.stderr)
        sys.exit(1)

    if jobs == -1:  # -1 == auto
        jobs = min(cpu_count(), len(tests), 32)

    print_info()

    origcwd = os.getcwd()
    testdir = tempfile.mkdtemp(prefix='ase-test-')
    os.chdir(testdir)

    # Note: :25 corresponds to ase.cli indentation
    print('{:25}{}'.format('test directory', testdir))
    if test_calculator_names:
        print('{:25}{}'.format('Enabled calculators:',
                               ' '.join(test_calculator_names)))
    print('{:25}{}'.format('number of processes',
                           jobs or '1 (multiprocessing disabled)'))
    print('{:25}{}'.format('time', time.strftime('%c')))
    if strict:
        print('Strict mode: Convert most warnings to errors')
    print()

    t1 = time.time()
    results = []
    try:
        for result in runtests_parallel(jobs, tests, verbose, strict):
            results.append(result)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')
        return 1
    else:
        summary(results)
        ntrouble = len([r for r in results if r.status in ['FAIL', 'ERROR']])
        return ntrouble
    finally:
        t2 = time.time()
        print('Time elapsed: {:.1f} s'.format(t2 - t1))
        os.chdir(origcwd)


def disable_calculators(names):
    for name in names:
        if name in ['emt', 'lj', 'eam', 'morse', 'tip3p']:
            continue
        try:
            cls = get_calculator(name)
        except ImportError:
            pass
        else:
            def get_mock_init(name):
                def mock_init(obj, *args, **kwargs):
                    raise NotAvailable('use --calculators={0} to enable'
                                       .format(name))
                return mock_init

            def mock_del(obj):
                pass
            cls.__init__ = get_mock_init(name)
            cls.__del__ = mock_del


def cli(command, calculator_name=None):
    if (calculator_name is not None and
        calculator_name not in test_calculator_names):
        return
    proc = subprocess.Popen(' '.join(command.split('\n')),
                            shell=True,
                            stdout=subprocess.PIPE)
    print(proc.stdout.read().decode())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError('Failed running a shell command.  '
                           'Please set you $PATH environment variable!')


class must_raise:
    """Context manager for checking raising of exceptions."""
    def __init__(self, exception):
        self.exception = exception

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise RuntimeError('Failed to fail: ' + str(self.exception))
        return issubclass(exc_type, self.exception)


class CLICommand:
    """Run ASE's test-suite.

    By default, tests for external calculators are skipped.  Enable with
    "-c name".
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            '-c', '--calculators',
            help='Comma-separated list of calculators to test')
        parser.add_argument('--list', action='store_true',
                            help='print all tests and exit')
        parser.add_argument('--list-calculators', action='store_true',
                            help='print all calculator names and exit')
        parser.add_argument('-j', '--jobs', type=int, default=-1,
                            metavar='N',
                            help='number of worker processes.  '
                            'By default use all available processors '
                            'up to a maximum of 32.  '
                            '0 disables multiprocessing')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Write test outputs to stdout.  '
                            'Mostly useful when inspecting a single test')
        parser.add_argument('--strict', action='store_true',
                            help='convert warnings to errors')
        parser.add_argument('tests', nargs='*',
                            help='Specify particular test files.  '
                            'Glob patterns are accepted.')

    @staticmethod
    def run(args):
        if args.calculators:
            calculators = args.calculators.split(',')
        else:
            calculators = []

        if args.list:
            dirname, _ = os.path.split(__file__)
            for testfile in get_tests(args.tests):
                print(os.path.join(dirname, testfile))
            sys.exit(0)

        if args.list_calculators:
            for name in calc_names:
                print(name)
            sys.exit(0)

        for calculator in calculators:
            if calculator not in calc_names:
                sys.stderr.write('No calculator named "{}".\n'
                                 'Possible CALCULATORS are: '
                                 '{}.\n'.format(calculator,
                                                ', '.join(calc_names)))
                sys.exit(1)

        ntrouble = test(calculators=calculators, jobs=args.jobs,
                        strict=args.strict,
                        files=args.tests, verbose=args.verbose)
        sys.exit(ntrouble)
