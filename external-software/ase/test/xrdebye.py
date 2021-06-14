"""Tests for XrDebye class"""
from __future__ import print_function

from ase.utils.xrdebye import XrDebye, wavelengths
from ase.cluster.cubic import FaceCenteredCubic
import numpy as np

tolerance = 1E-5
# previously calculated values
expected_get = 116850.37344
expected_xrd = np.array([18549.274677, 52303.116995, 38502.372027])
expected_saxs = np.array([372650934.006398, 280252013.563702,
                          488123.103628])

# test system -- cluster of 587 silver atoms
atoms = FaceCenteredCubic('Ag', [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
                          [6, 8, 8], 4.09)
xrd = XrDebye(atoms=atoms, wavelength=wavelengths['CuKa1'], damping=0.04,
              method='Iwasa', alpha=1.01, warn=True)
# test get()
obtained_get = xrd.get(s=0.09)
assert np.abs((obtained_get - expected_get) / expected_get) < tolerance

# test XRD
obtained_xrd = xrd.calc_pattern(x=np.array([15, 30, 50]), mode='XRD')
assert np.allclose(obtained_xrd, expected_xrd, rtol=tolerance)

# test SAXS
obtained_saxs = xrd.calc_pattern(x=np.array([0.021, 0.09, 0.53]),
                                 mode='SAXS')
assert np.allclose(obtained_xrd, expected_xrd, rtol=tolerance)
