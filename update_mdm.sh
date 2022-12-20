#!/usr/bin/env bash

set -euo pipefail

linkfiles=(
	mdm/data_loaders/humanml/common/quaternion.py
	mdm/data_loaders/humanml/common/skeleton.py
	mdm/data_loaders/humanml/scripts/motion_process.py
	mdm/data_loaders/humanml/utils/paramUtil.py
	mdm/data_loaders/tensors.py
	mdm/diffusion/gaussian_diffusion.py
	mdm/diffusion/losses.py
	mdm/diffusion/nn.py
	mdm/diffusion/respace.py
	mdm/model/cfg_sampler.py
	mdm/model/mdm.py
	mdm/model/rotation2xyz.py
	mdm/model/smpl.py
	mdm/utils/config.py
	mdm/utils/dist_util.py
	mdm/utils/fixseed.py
	mdm/utils/model_util.py
	mdm/utils/rotation_conversions.py
)

main() {
	git submodule update --remote mdm

	local linkfile
	for linkfile in ${linkfiles[@]}; do
		local from_file="${linkfile}"
		local to_file="${linkfile/mdm\//motion_generate_tools\/mdm\/}"
		local to_dir=$(dirname "${to_file}")
		if [[ ! -d "${to_dir}" ]]; then
			mkdir -p "${to_dir}"
		fi
		ln -f "${from_file}" "${to_dir}"
	done
}

main