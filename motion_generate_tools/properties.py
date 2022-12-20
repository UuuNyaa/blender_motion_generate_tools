# -*- coding: utf-8 -*-
# Copyright 2022 UuuNyaa <UuuNyaa@gmail.com>
# This file is part of Motion Generate Tools.

import bpy


class MotionGeneratorToolsProperties(bpy.types.PropertyGroup):
    text_condition: bpy.props.StringProperty(name='Text Condition')
    diffusion_sampling_steps: bpy.props.IntProperty(name='Diffusion Sampling Steps', default=100, min=1, soft_max=200, max=1000)

    @staticmethod
    def register():
        # pylint: disable=assignment-from-no-return
        bpy.types.WindowManager.motion_generator_tools = bpy.props.PointerProperty(type=MotionGeneratorToolsProperties)

    @staticmethod
    def unregister():
        del bpy.types.WindowManager.motion_generator_tools
