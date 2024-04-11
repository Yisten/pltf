cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
PLANNER="planTF"
CKP_NAME="last"
SPLIT="val14_2"
# CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"
CHALLENGES="closed_loop_nonreactive_agents"
for challenge in $CHALLENGES; do
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan \
        scenario_filter=$SPLIT \
        worker.threads_per_node=16 \
        experiment_uid=$SPLIT/$planner \
        verbose=false \
        planner.imitation_planner.planner_ckpt="$CKPT_ROOT/$CKP_NAME.ckpt"
done


