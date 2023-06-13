"""
A dedicated helper to manage templates and prompt building.
"""

class Prompter():
    template_dict = {
        "xnli": {
            "en": {"template": "Question:\n{premise} Based on the previous passage, is it true that \"{hypothesis}\"? Yes, No, or Maybe?\n\nAnswer:\n",
                   "choices": ['Yes', 'No', 'Maybe'],
                   "input_columns": ['premise', 'hypothesis'],
                   "output_column": 'label',         
                  }
        },
        "logiqa": {
            "en": {"template": "Question:\n{input}\n\nAnswer:\n",
                   "choices": ['A', 'B', 'C', 'D'],
                   "input_columns": ['input'],
                   "output_column": 'label'}
        },
        "xcopa": {
            "en": {"template": "Question:\n{premise} Based on the previous passage, choose the most reasonable {question}.\nA:{choice1}\nB:{choice2}\n\nAnswer:\n",
                   "choices": ['A', 'B'],
                   "input_columns": ['premise', 'question', 'choice1', 'choice2'],
                   "output_column": 'label'}
        },
        "gsm8k": {
            "en": {"template": "Question:\n{question}\n\nAnswer:\n",
                   "input_columns": ['question'],
                   "output_column": 'answer'}
        }
        
    }
    def __init__(self, task_name, prompt_lang) -> None:
        super().__init__()
        self.template = self.template_dict[task_name][prompt_lang]

    def get_prompt(self, data_point, with_label=False):
        
        prompt = self.template['template'].format(**{k: data_point[k] for k in self.template['input_columns']})
        if with_label:
            if 'choices' in self.template:
                prompt += self.template['choices'][data_point[self.template['output_column']]]
            else:
                prompt += data_point[self.template['output_column']]
        return prompt