export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1


# bloom, pythia
lora_target_modules='[query_key_value,]'
models=(
        bigscience/bloom-7b1
        # bigscience/bloom-1b7
        # bigscience/bloom-560m0
        # EleutherAI/pythia-6.9b-deduped
        # EleutherAI/pythia-1.4b-deduped
        # EleutherAI/pythia-410m-deduped
    )


micro_batch_size=8
task=xnli
train_langs=(en de es fr sw tr vi ar bg el hi ru ur th zh)
test_langs=(en de es fr sw tr vi ar bg el hi ru ur th zh)
# task=xcopa
# train_langs=(en zh fr)
# test_langs=(en zh fr)

prompt_lang=en

metric_report=report.csv

for task in ${tasks[@]}
do
    for model in ${models[@]}
    do
        for test_lang in ${test_langs[@]}
        do  
            if [ ${use_native_lang} = true ] ; then
                prompt_lang=${test_lang}
            fi
            echo "Zero-shot testing on ${task}-${test_lang} with ${model}"
            prediction_output_dir=prediction${exp}/${model}/${task}/zero-shot
            python evaluation.py \
                --base_model ${model} \
                --data_path data/${task}/${test_lang}/test.jsonl \
                --output_dir ${prediction_output_dir} \
                --metric_report ${metric_report} \
                --prompt_lang ${prompt_lang} \
                --task_name ${task} \
                --batch_size 64
        done

        for train_lang in ${train_langs[@]}
        do
            if [ ${use_native_lang} = true ] ; then
                prompt_lang=${train_lang}
            fi
            run_name=${model}/${task}/${train_lang}
            export WANDB_NAME=${run_name}
            model_output_dir=output/${run_name}

            echo "Training on ${task}-${train_lang} with ${model}"
            python finetune.py \
                --base_model ${model} \
                --data_paths data/${task}/${train_lang}/train.jsonl \
                --output_dir ${model_output_dir} \
                --lora_target_modules ${lora_target_modules} \
                --micro_batch_size ${micro_batch_size} \
                --prompt_langs ${prompt_lang} \
                --task_name ${task}

            for test_lang in ${test_langs[@]}
            do
                if [ ${use_native_lang} = true ] ; then
                    prompt_lang=${test_lang}
                fi
                echo "Testing on ${task}-${test_lang} with ${task}-${train_lang} trained ${model}"
                prediction_output_dir=prediction${exp}/${run_name}
                python evaluation.py \
                    --base_model ${model} \
                    --data_path data/${task}/${test_lang}/test.jsonl \
                    --output_dir ${prediction_output_dir} \
                    --metric_report ${metric_report} \
                    --lora_weights ${model_output_dir} \
                    --prompt_lang ${prompt_lang} \
                    --task_name ${task}
            done
        done
    done
done