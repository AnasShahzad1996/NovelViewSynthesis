OUT_DIR=data/own/Synthetic_NeRF_Lego

set -euxo pipefail

blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_UV4 --prefix 0 --start 0   --resolution 800 --views 100 --random --cam_scale 1 --seed 0
blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_UV4 --prefix 0 --start 100 --resolution 800 --views 25  --cam_scale 3 --seed 0
# start/end -.4, .9
blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_UV4 --prefix 0 --start 125 --resolution 800 --views 25  --cam_scale 4 --seed 0
# start/end -.4, .-.2

blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_UV4 --prefix 1 --start 0   --resolution 800 --views 100 --cam_scale 1 --seed 0
# start/end -.4, .9
blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_UV4 --prefix 1 --start 100 --resolution 800 --views 50  --cam_scale 4 --seed 0

# UV grid
#blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_UV2 --prefix 0 --resolution 800 --views 150 --random
#blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_UV2 --prefix 1 --resolution 800 --views 200
#blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_UV --prefix 2 --resolution 800 --views 200

# Color grid
#blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_color --prefix 0 --resolution 800 --views 200
#blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_color --prefix 1 --resolution 800 --views 100 --random
#blender --background  ../blend_files/blend_files/lego_color.blend --python nerf_render_ori.py -- ${OUT_DIR}_color --prefix 2 --resolution 800 --views 200 --random --upper

# Texture
#blender --background  ../blend_files/blend_files/lego_texture.blend --python nerf_render_ori.py -- ${OUT_DIR}_texture --prefix 0 --resolution 800 --views 200
#blender --background  ../blend_files/blend_files/lego_texture.blend --python nerf_render_ori.py -- ${OUT_DIR}_texture --prefix 1 --resolution 800 --views 100 --random
#blender --background  ../blend_files/blend_files/lego_texture.blend --python nerf_render_ori.py -- ${OUT_DIR}_texture --prefix 2 --resolution 800 --views 200 --random --upper
