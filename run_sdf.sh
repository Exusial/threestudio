# python3 launch.py --config configs/dreamavatar-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Captain American Full Body"
# python3 launch.py --config configs/dreamavatar-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Captain American Full Body"
# python3 launch.py --config configs/dreamavatar-vsd.yaml --train --gpu 6 system.prompt_processor.prompt="Elsa" \
# python3 launch.py --config configs/dreamavatar-vsd-zoom.yaml --train --gpu 2 system.prompt_processor.prompt="Elsa" system.prompt_processor.part_prompt="Elsa" \
# python3 launch.py --config configs/dreamavatar-vsd-zoom.yaml --train --gpu 0 system.prompt_processor.prompt="Elsa" system.prompt_processor.part_prompt="Elsa" \
# system.geometry.smpl_model_dir="/home/penghy/diffusion/avatars/models" 

# for mesh export
# python launch.py --config /home/penghy/diffusion/threestudio/outputs/dreamavatar-vsd-zoom/Elsa@20230803-170446/configs/parsed.yaml --export --gpu 3 resume=/home/penghy/diffusion/threestudio/outputs/dreamavatar-vsd-zoom/Elsa@20230803-170446/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.fmt=obj

# python launch.py --config /home/penghy/diffusion/threestudio/outputs/Elsa@20230807-213758/configs/parsed.yaml --export --gpu 2 resume=/home/penghy/diffusion/threestudio/outputs/Elsa@20230807-213758/ckpts/last.ckpt \
# system.exporter_type=mesh-exporter system.geometry.isosurface_method=mc-cpu system.geometry.isosurface_resolution=192 system.geometry.isosurface_threshold=15.0 system.exporter.fmt=ply

export HTTPS_PROXY=http://ddns.shadowhome.top:7890
# for geo part
# python launch.py --config configs/dreamavatar-sds-control.yaml --train --gpu 6 \
# system.prompt_processor.part_prompt="masterpiece,best quality,Cjiang,1gril" \
# system.prompt_processor.prompt="masterpiece,best quality,Cjiang,1gril"
# system.prompt_processor.prompt="Elsa"
#system.prompt_processor.prompt="masterpiece, best quality, Cjiang, 1gril, full body, white background"
#system.prompt_processor.prompt="Elsa"
#system.prompt_processor.prompt="masterpiece, best quality, Cjiang, 1gril, looking at view, full body, white background" \
#system.prompt_processor.part_prompt="masterpiece, best quality, Cjiang, 1gril, looking at view, full body, white background"
#system.geometry_convert_from=/home/penghy/diffusion/threestudio/outputs/dreamavatar-vsd-zoom/Elsa@20230803-170446/ckpts/last.ckpt \
#system.geometry_convert_override.isosurface_threshold=10.
python launch.py --config configs/dreamavatar-sdf-control.yaml --train --gpu 1 \
system.prompt_processor.part_prompt="elsa" \
system.prompt_processor.prompt="elsa"