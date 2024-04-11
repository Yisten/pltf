# cwd=$(pwd)
# CKPT_ROOT="$cwd/checkpoints"
# PLANNER="planTF"

python run_training.py\
    py_func=train\
    +training=train_planTF\
    worker=single_machine_thread_pool\
    worker.max_workers=32 \
    scenario_builder=nuplan\
    cache.use_cache_without_dataset=true \
    data_loader.params.batch_size=32\
    data_loader.params.num_workers=32\
    lr=1e-3 epochs=100 warmup_epochs=3 weight_decay=0.0001\
    lightning.trainer.params.val_check_interval=0.5\
    cache.cache_path="/rbs/guojy/plantf/planTF_cache/" \
    wandb.mode=online wandb.project=nuplan wandb.name=plantf\
    +cache.cache_path_file="/root/planTF/cache_paths.txt"