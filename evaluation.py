import os
import sys
import re
import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def process_gsm8k_prediction(predictions):
    results = []
    for p in predictions:
        splits = p.split("####")
        res = re.findall(r'\d+', splits[-1])
        if len(splits) > 1:
            # task the first number after ####
            ans = res[0] if len(res)>0 else ""
        else:
            # task the last number if no ####
            ans = res[-1] if len(res)>0 else ""
        results.append(ans)
    return results


def main(
    load_8bit: bool = False,
    base_model: str = "bigscience/bloom-560m",
    lora_weights: str = "",  # "" means no lora weights
    data_path: str = "", 
    metric_report: str = "",
    output_dir: str = "",
    prompt_lang: str = None, 
    task_name: str = "",
    batch_size: int = 32,
):
    if prompt_lang is None:
        # by default using in-language prompt
        prompt_lang = data_path.split('/')[-2]
    print(
        f"Generating params:\n"
        f"load_8bit: {load_8bit}\n"
        f"base_model: {base_model}\n"
        f"lora_weights: {lora_weights}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"prompt_lang: {prompt_lang}\n"
        f"task_name: {task_name}\n"
        f"metric_report: {metric_report}\n"
        f"batch_size: {batch_size}\n"
    )
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"


    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            resume_download=True
        )
        if lora_weights:
            print(f"Loading lora weights from {lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            resume_download=True
        )
        if lora_weights:
            print(f"Loading lora weights from {lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            device_map={"": device}, 
            low_cpu_mem_usage=True,
            resume_download=True
        )
        if lora_weights:
            print(f"Loading lora weights from {lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
        # hack for OpenICL
        data['test'] = data['train']
    else:
        data = load_dataset(data_path)
    
    output_json_filename = data_path.replace('/', '_')
    os.makedirs(output_dir, exist_ok=True)

    from prompter import Prompter
    from openicl import DatasetReader, PromptTemplate, ZeroRetriever, PPLInferencer, GenInferencer, AccEvaluator
    prompter = Prompter(task_name=task_name,prompt_lang=prompt_lang)
    template = prompter.template
    data = DatasetReader(data, 
                         input_columns=template['input_columns'], 
                         output_column=template['output_column'])
    if "choices" in template:
        tp_dict = {
            i: f"{template['template']}{c}"
            for i, c in enumerate(template['choices'])
        }
        inferencer = PPLInferencer(model_name=model, tokenizer_name=tokenizer, batch_size=batch_size)
    else:
        tp_dict = template['template']
        inferencer = GenInferencer(model_name=model, tokenizer_name=tokenizer, batch_size=batch_size, generation_kwargs={"max_new_tokens": 300})

    template = PromptTemplate(tp_dict, {k: '{'+k+'}' for k in template['input_columns']}, ice_token="")
    retriever = ZeroRetriever(data)
    
    predictions = inferencer.inference(retriever=retriever, prompt_template=template,
                                    output_json_filepath=output_dir,
                                    output_json_filename=output_json_filename)
    if task_name == "gsm8k":
        predictions = process_gsm8k_prediction(predictions)
    score = AccEvaluator().score(predictions=predictions, references=data.references)['accuracy']
    print(score)

    with open(metric_report, mode='a') as f:
        test_lang = data_path.split('/')[-2]
        train_lang = "" if lora_weights == "" else lora_weights.split('/')[-1]
        train_task = "" if lora_weights == "" else lora_weights.split('/')[-2]
        f.write(",".join([task_name, train_task, base_model, train_lang, test_lang, str(score)])+"\n")
    

if __name__ == "__main__":
    fire.Fire(main)
