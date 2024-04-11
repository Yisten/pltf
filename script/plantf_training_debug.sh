# cwd=$(pwd)
# CKPT_ROOT="$cwd/checkpoints"
# PLANNER="planTF"

python -m debugpy --listen localhost:5678 --wait-for-client\
    run_training.py \
    py_func=train \
    +training=train_planTF \
    worker=single_machine_thread_pool \
    worker.max_workers=16 \
    scenario_builder=nuplan \
    cache.use_cache_without_dataset=true \
    data_loader.params.batch_size=32 \
    data_loader.params.num_workers=32 \
    lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
    lightning.trainer.params.val_check_interval=0.5 \
    cache.cache_path="/data2/planTF_cache" \
    wandb.mode=online wandb.project=nuplan wandb.name=plantf \
    +cache.cache_path_file="/home/guojiayu/research/planTF/cache_paths.txt"