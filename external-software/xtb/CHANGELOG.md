## xtb version 6.2.3

Maintenance and bugfix release.

- Bugfix: JSON dump did not return correct version number
- Bugfix: Sign error in printout for GFN1-xTB Mulliken charges
- Bugfix: PDB charges were not written correctly in PDB output
- Bugfix: SRB strain derivatives were wrong
- Bugfix: EEQ returns correct energy for one atom systems
- The error handling has been completely reworked
- Improved implementation of CM5 charges
- Analytical gradients for the Halogen-Bond correction
- Benzene is now available as synonym for toluene for GFN2/GBSA
- Massively improved C-API
- Gaussian external input and output formats
- CMake support for building xtb
- GCC support for building xtb with both meson and CMake

This release is API compatible to version 6.2.2
