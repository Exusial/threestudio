export HTTPS_PROXY=http://ddns.shadowhome.top:7890
python launch.py --config configs/dreamavatar-sds-control-geo.yaml --train --gpu 7 \
system.stage="mesh" \
system.prompt_processor.part_prompt="masterpiece,best quality,Cjiang,1gril" \
system.prompt_processor.prompt="masterpiece,best quality,Cjiang,1gril" \
system.geometry_convert_from="/data/penghy/threestudio/outputs/dreamavatar-sds-control/masterpiece,best_quality,Cjiang,1gril@20230912-141729/ckpts/last.ckpt" \
system.geometry_convert_override.isosurface_threshold=10.
# system.prompt_processor.prompt="Elsa"
#system.prompt_processor.prompt="masterpiece, best quality, Cjiang, 1gril, full body, white background"
#system.prompt_processor.prompt="Elsa"
#system.prompt_processor.prompt="masterpiece, best quality, Cjiang, 1gril, looking at view, full body, white background" \
#system.prompt_processor.part_prompt="masterpiece, best quality, Cjiang, 1gril, looking at view, full body, white background"
