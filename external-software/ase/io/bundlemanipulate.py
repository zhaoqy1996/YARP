"""Functions for in-place manipulation of bundletrajectories.

This module defines a number of functions that can be used to
extract and delete data from BundleTrajectories directly on
disk.  The functions are intended for large-scale MD output,
so they avoid copying the potentially large amounts of data.
In stead, data is either directly deleted in-place; or copies
are made by creating a new directory structure, but hardlinking
the data files.  Hard links makes it possible to delete the
original data without invalidating the copy.

Usage from command line:

python -m ase.io.bundlemanipulate inbundle outbundle [start [end [step]]]
"""

from __future__ import print_function
from ase.io.bundletrajectory import PickleBundleBackend, UlmBundleBackend
import os
import pickle
import json
import numpy as np


def copy_frames(inbundle, outbundle, start=0, end=None, step=1,
                verbose=False):
    """Copies selected frame from one bundle to the next."""
    if not (isinstance(start, int) and
            (isinstance(end, int) or end is None) and
            isinstance(step, int)):
        raise TypeError("copy_frames: start, end and step must be integers.")
    metadata, nframes = read_bundle_info(inbundle)
    if metadata['backend'] == 'ulm':
        ulm = True
        backend = UlmBundleBackend(True, metadata['ulm.singleprecision'])
    else:
        ulm = False
        assert(metadata['backend'] == 'pickle')
        backend = PickleBundleBackend(True)
        backend.readpy2 = False
        
    if start < 0:
        start += nframes
    if end is None:
        end = nframes
    if end < 0:
        end += nframes
    if start < 0 or (start > nframes - 1 and end > 0):
        raise ValueError("copy_frames: Invalid start value.")
    if end < 0 or (end > nframes - 1 and end < 0):
        raise ValueError("copy_frames: Invalid end value.")
    if step == 0:
        raise ValueError("copy_frames: Invalid step value (zero)")
    frames = list(range(start, end, step))
    if verbose:
        print("Copying the frames", frames)
    
    # Make the new bundle directory
    os.mkdir(outbundle)
    with open(os.path.join(outbundle, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    for nout, nin in enumerate(frames):
        if verbose:
            print("F%i -> F%i" % (nin, nout))
        indir = os.path.join(inbundle, "F" + str(nin))
        outdir = os.path.join(outbundle, "F" + str(nout))
        os.mkdir(outdir)
        names = os.listdir(indir)
        for name in names:
            fromfile = os.path.join(indir, name)
            tofile = os.path.join(outdir, name)
            os.link(fromfile, tofile)
        if nout == 0 and nin != 0:
            if verbose:
                print("F0 -> F0 (supplemental)")
            # The smalldata.pickle stuff must be updated.
            # At the same time, check if the number of fragments
            # has not changed.
            data0 = backend.read_small(os.path.join(inbundle, "F0"))
            data1 = backend.read_small(indir)
            split_data = (metadata['subtype'] == 'split')
            if split_data:
                fragments0 = data0['fragments']
                fragments1 = data1['fragments']

            data0.update(data1)  # Data in frame overrides data from frame 0.
            backend.write_small(outdir, data0)

            # If data is written in split mode, it must be reordered
            firstnames = os.listdir(os.path.join(inbundle, "F0"))
            if not split_data:
                # Simple linking
                for name in firstnames:
                    if name not in names:
                        if verbose:
                            print("   ", name, "  (linking)")
                        fromfile = os.path.join(inbundle, "F0", name)
                        tofile = os.path.join(outdir, name)
                        os.link(fromfile, tofile)
            else:
                # Must read and rewrite data
                # First we read the ID's from frame 0 and N
                if ulm:
                    assert 'ID_0.ulm' in firstnames and 'ID_0.ulm' in names
                else:
                    assert 'ID_0.pickle' in firstnames and 'ID_0.pickle' in names
                backend.nfrag = fragments0
                f0_id, dummy = backend.read_split(os.path.join(inbundle, "F0"), "ID")
                backend.nfrag = fragments1
                fn_id, fn_sizes = backend.read_split(indir, "ID")
                for name in firstnames:
                    # Only look at each array, not each file
                    if '_0.' not in name:
                        continue
                    if name not in names:
                        # We need to load this array
                        arrayname = name.split('_')[0]
                        print("    Reading", arrayname)
                        backend.nfrag = fragments0
                        f0_data, dummy = backend.read_split(os.path.join(inbundle, "F0"), arrayname)
                        # Sort data
                        f0_data[f0_id] = np.array(f0_data)
                        # Unsort with new ordering
                        f0_data = f0_data[fn_id]
                        # Write it
                        print("    Writing reshuffled", arrayname)
                        pointer = 0
                        backend.nfrag = fragments1
                        for i, s in enumerate(fn_sizes):
                            segment = f0_data[pointer:pointer+s]
                            pointer += s
                            backend.write(outdir, arrayname+"_{0}".format(i), segment)
    # Finally, write the number of frames
    f = open(os.path.join(outbundle, 'frames'), 'w')
    f.write(str(len(frames)) + '\n')
    f.close()
    

# Helper functions
def read_bundle_info(name):
    """Read global info about a bundle.
    
    Returns (metadata, nframes)
    """
    if not os.path.isdir(name):
        raise IOError("No directory (bundle) named '%s' found." % (name,))
    metaname = bestmetaname = os.path.join(name, 'metadata.json')
    if os.path.isfile(metaname):
        with open(metaname) as f:
            mdata = json.load(f)
    else:
        metaname = os.path.join(name, 'metadata')
        if os.path.isfile(metaname):
            with open(metaname, "rb") as f:
                mdata = pickle.load(f)
        else:
            raise IOError("'%s' does not appear to be a BundleTrajectory (no %s)"
                        % (name, bestmetaname))
    if 'format' not in mdata or mdata['format'] != 'BundleTrajectory':
        raise IOError("'%s' does not appear to be a BundleTrajectory" %
                      (name,))
    if mdata['version'] != 1:
        raise IOError("Cannot manipulate BundleTrajectories with version "
                      "number %s" % (mdata['version'],))
    f = open(os.path.join(name, "frames"))
    nframes = int(f.read())
    if nframes == 0:
        raise IOError("'%s' is an empty BundleTrajectory" % (name,))
    return mdata, nframes


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit()
    inname, outname = sys.argv[1:3]
    if len(sys.argv) > 3:
        start = int(sys.argv[3])
    else:
        start = 0
    if len(sys.argv) > 4:
        end = int(sys.argv[4])
    else:
        end = None
    if len(sys.argv) > 5:
        step = int(sys.argv[5])
    else:
        step = 1
    copy_frames(inname, outname, start, end, step, verbose=1)
