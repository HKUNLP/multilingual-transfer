# Language Versatilists vs. Specialist
This repository contains the code for our paper “[Language Versatilists vs. Specialists: An Empirical Revisiting on Multilingual Transfer Ability](https://arxiv.org/abs/2306.06688)”.
The implementation is built on the source code from [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) and [OpenICL](https://github.com/Shark-NLP/OpenICL). Thanks for their work.

## Setup
All requirements can be found in `requirements.txt`. You can install all required packages by:
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
cd OpenICL
pip install .
```

## Usage

The multilingual datasets used in the work includes XNLI, LogiQA, GSM8K and XCOPA. We additionally translate the training set of LogiQA, XCOPA and GSM8K into French and Chinese with Google Translate.

Running fine-tuning and evaluation with command `bash run-bloom.sh` and `bash run-llama.sh`.

## Cite
```
@article{ye2023language,
      title={Language Versatilists vs. Specialists: An Empirical Revisiting on Multilingual Transfer Ability}, 
      author={Jiacheng Ye and Xijia Tao and Lingpeng Kong},
      year={2023},
      eprint={2306.06688},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```