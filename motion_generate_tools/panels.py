# -*- coding: utf-8 -*-
# Copyright 2022 UuuNyaa <UuuNyaa@gmail.com>
# This file is part of Motion Generate Tools.

import re
import sys
import time
from typing import Any, List, Optional

import bpy
from mathutils import Matrix, Quaternion, Vector

from . import executors, setup_utils
from .mdm import filepaths

_LINES: List[str] = []

_motion_generator: Optional[Any] = None


def _get_motion_generator():
    global _motion_generator

    if _motion_generator is None:
        if filepaths.WORK_PATH not in sys.path:
            sys.path.append(filepaths.WORK_PATH)

        from .mdm import motion_generator
        _motion_generator = motion_generator.InBetweenMotionGenerator()

    return _motion_generator


t2m_kinematic_chain = [[0, 3, 6, 9, 12, 15], [2, 5, 8, 11], [1, 4, 7, 10], [14, 17, 19, 21], [13, 16, 18, 20]]


SMPL_JOINT_NAMES = [
    'pelvis',   # 0
    'left_hip',    # 1
    'right_hip',    # 2
    'spine1',   # 3
    'left_knee',   # 4
    'right_knee',   # 5
    'spine2',   # 6
    'left_ankle',  # 7
    'right_ankle',  # 8
    'spine3',   # 9
    'left_foot',   # 10
    'right_foot',   # 11
    'neck',     # 12
    'left_collar',  # 13
    'right_collar',  # 14
    'head',     # 15
    'left_shoulder',   # 16
    'right_shoulder',   # 17
    'left_elbow',  # 18
    'right_elbow',  # 19
    'left_wrist',  # 20
    'right_wrist',  # 21
    'left_hand',   # 22
    'right_hand',   # 23
]


def to_blender_vector(smpl_xyz: Vector) -> Vector:
    return Vector([-smpl_xyz.x, smpl_xyz.z, smpl_xyz.y])


def to_blender_quaternion(smpl_wxyz: Quaternion) -> Quaternion:
    return Quaternion([smpl_wxyz.w, -smpl_wxyz.x, smpl_wxyz.z, smpl_wxyz.y])


class GenarateMotion(bpy.types.Operator):
    bl_idname = 'motion_generate_tools.generate_motion'
    bl_label = 'Generate Motion'
    bl_options = {'REGISTER', 'UNDO'}

    text_condition: bpy.props.StringProperty(name='Text Condition')
    diffusion_sampling_steps: bpy.props.IntProperty(name='Diffusion Sampling Steps', default=100, min=1, max=1000)

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'ARMATURE'

    def _apply_motion(self, context: bpy.types.Context, armature_object: bpy.types.Object, motion: List):
        scale = 1.109527
        pose_bones = armature_object.pose.bones

        for frame, joints in enumerate(motion, start=context.scene.frame_current):
            for bone_index_chain in t2m_kinematic_chain:
                parent_tail_location = (
                    None if bone_index_chain[0] == 0
                    else pose_bones[SMPL_JOINT_NAMES[bone_index_chain[0]]].head
                )

                for chain_index, bone_index in enumerate(bone_index_chain[:-1], start=1):
                    pose_bone = pose_bones[SMPL_JOINT_NAMES[bone_index]]
                    head_joint_location = to_blender_vector(Vector(joints[bone_index][:3]) / scale)

                    head_location = parent_tail_location or head_joint_location
                    tail_joint_index = bone_index_chain[chain_index]
                    tail_joint_location = to_blender_vector(Vector(joints[tail_joint_index][:3]) / scale)
                    quaternion = Vector((0, 1, 0)).rotation_difference(tail_joint_location-head_location)
                    parent_tail_location = head_location + quaternion @ Vector((0, pose_bone.length, 0))

                    pose_bone_matrix = armature_object.convert_space(
                        pose_bone=pose_bone,
                        matrix=Matrix.LocRotScale(head_location, quaternion, None),
                        from_space='WORLD',
                        to_space='LOCAL'
                    )

                    if not (pose_bone.bone.use_connect or any(pose_bone.lock_location)):
                        pose_bone.location = pose_bone_matrix.translation
                        pose_bone.keyframe_insert('location', frame=frame)

                    pose_bone.rotation_quaternion = pose_bone_matrix.to_quaternion()
                    pose_bone.keyframe_insert('rotation_quaternion', frame=frame)
                    context.view_layer.update()

    def execute(self, context):
        text_condition = self.text_condition
        diffusion_sampling_steps = self.diffusion_sampling_steps

        wm = context.window_manager
        wm.progress_begin(0, diffusion_sampling_steps)

        progress = []

        executor = executors.FunctionExecutor()
        executor.exec_function(
            lambda: _get_motion_generator().generate(text_condition, diffusion_sampling_steps),
            line_callback=lambda l: progress.clear() or progress.append(l)
        )

        pattern = re.compile(r'\r [0-9]+%\|[^|]+\| ([0-9]+)/')

        while executor.is_running:
            time.sleep(0.5)
            if len(progress) > 0:
                match = pattern.match(progress.pop())
                if match:
                    wm.progress_update(int(match.group(1)))

        wm.progress_end()

        self._apply_motion(context, context.active_object, executor.return_value)

        return {'FINISHED'}


class MotionGeneratorPanel(bpy.types.Panel):
    bl_idname = 'ANIMATION_PT_motion_generator_panel'
    bl_label = 'Motion Generator'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Animation'

    text_condition: bpy.props.StringProperty(name='Text Condition')

    def draw(self, context: bpy.types.Context):
        layout = self.layout

        is_clip_model_exist = setup_utils.is_clip_model_exist()
        are_modules_available = all(setup_utils.get_required_modules().values())

        if not (is_clip_model_exist and are_modules_available):
            col = layout.column(align=True)
            col.label(text="Please setup in Preferences:", icon='PREFERENCES')
            flow = col.box().grid_flow(align=True, row_major=True)

            flow.row().label(text="CLIP Model:")
            if not is_clip_model_exist:
                flow.row().label(text="Missing", icon='ERROR')
            else:
                flow.row().label(text="OK", icon='CHECKMARK')

            flow.row().label(text="Python Modules:")
            if not are_modules_available:
                flow.row().label(text="Missing", icon='ERROR')
            else:
                flow.row().label(text="OK", icon='CHECKMARK')

            return

        col = layout.column(align=False)
        op = col.operator(GenarateMotion.bl_idname)
        op.text_condition = context.window_manager.motion_generator_tools.text_condition
        op.diffusion_sampling_steps = context.window_manager.motion_generator_tools.diffusion_sampling_steps

        box = col.box().column()
        col_prop = box.column(align=True)
        col_prop.label(text="Text Condition:")
        col_prop.prop(context.window_manager.motion_generator_tools, 'text_condition', text="")
        box.prop(context.window_manager.motion_generator_tools, 'diffusion_sampling_steps', slider=True)
