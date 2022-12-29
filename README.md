# motion_generate_tools
motion_generate_tools is a Blender addon for generate motion using the following Model.

motion_generate_toolsは以下のモデルを使ってモーションを生成するためのBlenderアドオンです。

- [MDM: Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model).

## Download
Download motion_generate_tools from [the github release page](https://github.com/UuuNyaa/blender_motion_generate_tools/releases)
 - https://github.com/UuuNyaa/blender_motion_generate_tools/releases

## Installation
The easiest way to install the motion_generate_tools is to do so through directly through Blender:

1. Open the Blender User Preferences menu and select the Add-ons tab (***Edit > Preferences... > Add-ons***)
2. Click the ***Install...*** button at the top of the Add-ons menu. This will open a file menu where you can select the `motion_generate_tools-v0.0.0.zip` file.
3. After installing the addon .zip file, Blender will filter the addons list to only show the motion_generate_tools addon. ***Click the checkbox*** next to the title to enable the addon.
4. Click Required Python Modules: ***Update Python Modules*** button at the addon Preferences.
    - If you have a NVIDIA GPU that supports CUDA, you can choose ***with CUDA*** button.
5. Click CLIP Model: ***Download CLIP Model*** button at the addon Preferences.
6. ***Restart Blender***. A Blender restart is maybe required to complete the installation.


https://user-images.githubusercontent.com/70152495/210015009-0b190692-291d-422f-ac95-dcebbecd97b8.mp4

### Requirements
 - Blender **3.3 LTS** or later

## Usage
1. Open the [`smpl_model_20210421-bones.blend`](https://tstorage.info/pzmsa6pgryzr) file
2. Select the ***SMPLX-neutral*** armature
3. Goto 3D Viewport > Sidebar > Animation > ***Motion Generator*** Panel
4. Enter a ***Text Condition***
    ```
    a person jumps.
    a person throws.
    a man is locking his hans behind his back and sweeping his legs right and left, in a dance like motion.
    a person is walking in a counterclockwise circle.
    ```
5. Press ***Generate Motion*** button

[![Motion Generator Tools under development](https://img.youtube.com/vi/pTkn2qWfc60/0.jpg)](https://youtu.be/pTkn2qWfc60)


### Limitation
- Only [SMPL armature](https://tstorage.info/pzmsa6pgryzr) is available.
- The motion to be edited is fixed.

## Acknowledgments
### Source code
This addon is based on code from the following repository:
- [MDM: Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model)
  - `mdm/data_loaders/`
  - `mdm/diffusion/`
  - `mdm/model/`
  - `mdm/utils/`

### Datasets
This addon depends on the following datasets:
- [MDM: Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model)
  - `data/save/`
- [HumanML3D: 3D Human Motion-Language Dataset](https://github.com/EricGuo5513/HumanML3D)
  - `data/dataset/HumanML3D`
- [SMPL: A Skinned Multi-Person Linear Model](https://smpl.is.tue.mpg.de/index.html)
  - `data/body_models/smpl`

## Contribution
We need people to test the installation and give us feedback!
- Discord Server: https://discord.gg/zRgUkuaPWw

## License
This code is distributed under the [GPLv3](LICENSE).

Note that our code depends on other libraries, including MDM, CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
