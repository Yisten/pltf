cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

# PLANNER="planTF"
# SPLIT="val14"
python run_training.py\
    py_func=cache\
    +training=train_planTF\
    scenario_builder=nuplan\
    cache.cleanup_cache=false\
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=16 \
    cache.cache_path="/data2/planTF_cache" \

