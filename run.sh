# python3 launch.py --config configs/dreamavatar-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Captain American Full Body"
# python3 launch.py --config configs/dreamavatar-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Captain American Full Body"
#python3 launch.py --config configs/dreamavatar-vsd.yaml --train --gpu 6 system.prompt_processor.prompt="Captain American, Full Body" \
python3 launch.py --config configs/dreamavatar-vsd.yaml --train --gpu 0 system.prompt_processor.prompt="Elsa,Full Body" \
system.geometry.smpl_model_dir="/home/penghy/diffusion/avatars/models" 