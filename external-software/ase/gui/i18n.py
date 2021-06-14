# i18n = i(18 letters omitted)n = internationalization
"""Module for localization with gettext.

When this module is imported, ASE-GUI will use translations depending
on system settings.  Usually language is taken from the LANG or LANGUAGE
environment variables.  Examples of how to override the system locale:

  LANG=da_DK.UTF-8 ase gui (Danish)
  LANGUAGE=da_DK.UTF-8 ase gui (Danish; normally overrides LANG)
  LANG=C ase gui (bare-bones ASCII locale disabling translations)

Other languages: es_ES.UTF-8, en_UK.UTF-8, ...

The encoding and/or country code can be omitted on most systems/languages.

Translations will be loaded from the mo-files when possible.  See ase/gui/po.

All modules that need translations should import _ from here,
along with ngettext if they want to translate messages with plurals
(e.g. "Save 1 file", "Save %d files")."""

import os
import sys
import gettext


domain = 'ag'
localedir = '%s/po/' % os.path.dirname(__file__)
translation = gettext.translation(domain, localedir, fallback=True)


if sys.version_info[0] == 2:
    _ = translation.ugettext
    ngettext = translation.ungettext
else:
    _ = translation.gettext
    ngettext = translation.ngettext
