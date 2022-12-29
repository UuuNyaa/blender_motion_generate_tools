# motion_generate_tools
motion_generate_tools is a Blender addon for generate motion using the following Model.

motion_generate_toolsは以下のモデルを使ってモーションを生成するためのBlenderアドオンです。

- [MDM: Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model).

## Installation

### Download
Download motion_generate_tools from [the github release page](https://github.com/UuuNyaa/blender_motion_generate_tools/releases)
 - https://github.com/UuuNyaa/blender_motion_generate_tools/releases

### Installing the Addon
Only one version of the addon should be installed at a time. If you are updating the addon to a new version, the previous version must be first uninstalled.

The easiest way to install the motion_generate_tools is to do so through directly through Blender:

1. Open the Blender User Preferences menu and select the Add-ons tab (***Edit > Preferences... > Add-ons***)
2. Click the ***Install...*** button at the top of the Add-ons menu. This will open a file menu where you can select the `mmd_tools-v0.0.0.zip` file.
3. After installing the addon .zip file, Blender will filter the addons list to only show the motion_generate_tools addon. ***Click the checkbox*** next to the title to enable the addon.
4. ***Restart Blender***. A Blender restart is maybe required to complete the installation.

![Install video](https://github.com/UuuNyaa/blender_motion_generate_tools/wiki/images/install.gif)

### Requirements
 - Blender **3.3 LTS** or later

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


## License
This code is distributed under the [GPLv3](LICENSE).

Note that our code depends on other libraries, including MDM, CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
