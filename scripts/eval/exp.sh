MAX_JOBS=20
current_jobs=0

# bash scripts/eval/exp.sh

# dataset="maze2d-umaze-v1"
dataset="maze2d-medium-v1"
# dataset="maze2d-large-v1"

# single
task="scripts/single/plan.py"
config="config.single.plan_diff"
plan_task="single"

for cond in "True" "False"
do
    for idx in {0..150}
    do
        python $task --dataset $dataset --config $config \
        --suffix "eval_$idx" --prefix plans/$plan_task --conditional $cond --device "cuda:0"\
        &

        # Increment the current_jobs counter
        current_jobs=$((current_jobs + 1))

        # If the maximum number of parallel jobs is reached, wait for all of them to finish
        if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
            wait
            current_jobs=0
        fi
    done
done


echo "Done"

