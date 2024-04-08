cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
PLANNER="planTF0403"

python -m debugpy --listen localhost:5678 --wait-for-client\
    run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    scenario_builder=nuplan \
    scenario_filter.log_names=["2021.07.24.15.54.20_veh-47_03573_05252"] \
    scenario_filter.scenario_tokens=["88eb75d4ebc856d3"] \
    scenario_filter=all_scenarios \
    worker=sequential \
    verbose=true \
    planner.imitation_planner.planner_ckpt="$CKPT_ROOT/$PLANNER.ckpt"