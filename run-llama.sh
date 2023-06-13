export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

# llama
lora_target_modules='[q_proj,v_proj]'
models=(
        huggyllama/llama-7b
        # huggyllama/llama-13b
        # huggyllama/llama-30b
    )


micro_batch_size=32
task=xnli
train_langs=(en de es fr sw tr vi ar bg el hi ru ur th zh)
test_langs=(en de es fr sw tr vi ar bg el hi ru ur th zh)
# task=logiqa
# train_langs=(en zh fr)
# test_langs=(en zh fr)
prompt_lang=en

metric_report=report.csv


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
            --batch_size 1
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
            --data_path data/${task}/${train_lang}/train.jsonl \
            --output_dir ${model_output_dir} \
            --lora_target_modules ${lora_target_modules} \
            --micro_batch_size ${micro_batch_size} \
            --prompt_lang ${prompt_lang} \
            --load_8bit true \
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
                --task_name ${task} \
                --batch_size 1
        done
    done
done