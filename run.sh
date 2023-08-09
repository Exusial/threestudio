# python3 launch.py --config configs/dreamavatar-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Captain American Full Body"
# python3 launch.py --config configs/dreamavatar-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Captain American Full Body"
# python3 launch.py --config configs/dreamavatar-vsd.yaml --train --gpu 6 system.prompt_processor.prompt="Elsa" \
python3 launch.py --config configs/dreamavatar-vsd-xl.yaml --train --gpu 1 system.prompt_processor.prompt="Elsa" \
# python3 launch.py --config configs/dreamavatar-vsd-zoom.yaml --train --gpu 0 system.prompt_processor.prompt="Elsa" \
# system.geometry.smpl_model_dir="/home/penghy/diffusion/avatars/models" 