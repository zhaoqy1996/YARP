"""Dialog for saving one or more configurations."""

from __future__ import unicode_literals
from ase.gui.i18n import _

import numpy as np

import ase.gui.ui as ui
from ase.io.formats import (write, parse_filename, get_ioformat, string2index,
                            filetype)
from ase.utils import basestring


text = _("""\
Append name with "@n" in order to write image
number "n" instead of the current image. Append
"@start:stop" or "@start:stop:step" if you want
to write a range of images. You can leave out
"start" and "stop" so that "name@:" will give
you all images. Negative numbers count from the
last image. Examples: "name@-1": last image,
"name@-2:": last two.""")


def save_dialog(gui, filename=None):
    dialog = ui.SaveFileDialog(gui.window.win, _('Save ...'))
    ui.Text(text).pack(dialog.top)
    filename = filename or dialog.go()
    if not filename:
        return

    filename, index = parse_filename(filename)
    if index is None:
        index = slice(gui.frame, gui.frame + 1)
    elif isinstance(index, basestring):
        index = string2index(index)
    elif isinstance(index, slice):
        pass
    else:
        if index < 0:
            index += len(gui.images)
        index = slice(index, index + 1)
    format = filetype(filename, read=False)
    io = get_ioformat(format)

    extra = {}
    remove_hidden = False
    if format in ['png', 'eps', 'pov']:
        bbox = np.empty(4)
        size = gui.window.size / gui.scale
        bbox[0:2] = np.dot(gui.center, gui.axes[:, :2]) - size / 2
        bbox[2:] = bbox[:2] + size
        extra['rotation'] = gui.axes
        extra['show_unit_cell'] = gui.window['toggle-show-unit-cell']
        extra['bbox'] = bbox
        colors = gui.get_colors(rgb=True)
        extra['colors'] = [rgb for rgb, visible
                           in zip(colors, gui.images.visible)
                           if visible]
        remove_hidden = True

    images = [gui.images.get_atoms(i, remove_hidden=remove_hidden)
              for i in range(*index.indices(len(gui.images)))]

    if len(images) > 1 and io.single:
        # We want to write multiple images, but the file format does not
        # support it.  The solution is to write multiple files, inserting
        # a number in the file name before the suffix.
        j = filename.rfind('.')
        filename = filename[:j] + '{0:05d}' + filename[j:]
        for i, atoms in enumerate(images):
            write(filename.format(i), atoms, **extra)
    else:
        try:
            write(filename, images, **extra)
        except Exception as err:
            from ase.gui.ui import showerror
            showerror(_('Error'), err)
            raise
