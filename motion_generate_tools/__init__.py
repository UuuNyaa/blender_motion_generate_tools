# -*- coding: utf-8 -*-
# Copyright 2022 UuuNyaa <UuuNyaa@gmail.com>
# This file is part of Motion Generate Tools.

# Motion Generate Tools is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# MMD UuuNyaa Tools is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "motion_generate_tools",
    "author": "UuuNyaa",
    "version": (0, 0, 4),
    "blender": (3, 3, 0),
    "location": "View3D > Sidebar > Panel",
    "description": "Utility tools for motion generating.",
    "warning": "",
    "doc_url": "https://github.com/UuuNyaa/blender_motion_generate_tools/wiki",
    "tracker_url": "https://github.com/UuuNyaa/blender_motion_generate_tools/issues",
    "category": "Animation",
}

import os

PACKAGE_PATH = os.path.dirname(__file__)
MOTION_GENERATE_TOOLS_VERSION = '.'.join(map(str, bl_info['version']))

from . import auto_load


auto_load.init()


def register():
    auto_load.register()


def unregister():
    auto_load.unregister()


if __name__ == "__main__":
    register()
