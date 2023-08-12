# python3 launch.py --config configs/dreamavatar-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Captain American Full Body"
# python3 launch.py --config configs/dreamavatar-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Captain American Full Body"
# python3 launch.py --config configs/dreamavatar-vsd.yaml --train --gpu 6 system.prompt_processor.prompt="Elsa" \
python3 launch.py --config configs/dreamavatar-vsd-zoom.yaml --train --gpu 2 system.prompt_processor.prompt="Elsa" system.prompt_processor.part_prompt="Elsa" \
# python3 launch.py --config configs/dreamavatar-vsd-zoom.yaml --train --gpu 0 system.prompt_processor.prompt="Elsa" system.prompt_processor.part_prompt="Elsa" \
# system.geometry.smpl_model_dir="/home/penghy/diffusion/avatars/models" 

# for mesh export
# python launch.py --config /home/penghy/diffusion/threestudio/outputs/dreamavatar-vsd-zoom/Elsa@20230803-170446/configs/parsed.yaml --export --gpu 3 resume=/home/penghy/diffusion/threestudio/outputs/dreamavatar-vsd-zoom/Elsa@20230803-170446/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.fmt=obj

# python launch.py --config /home/penghy/diffusion/threestudio/outputs/dreamavatar-vsd-zoom/Elsa@20230803-170446/configs/parsed.yaml --export --gpu 2 resume=/home/penghy/diffusion/threestudio/outputs/dreamavatar-vsd-zoom/Elsa@20230803-170446/configs/parsed.yaml \
# system.exporter_type=mesh-exporter system.geometry.isosurface_threshold=25.