# Patrick Young, Stanford CS106EA

import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

from typing import List, TypedDict

import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML, Button, Output, Text, Label

from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
    # without attn_implementation="eager" we get a warning when actually running the model

## Tokenizer

class TokenizerOutput(TypedDict):
    input_ids: torch.Tensor # token IDs for subwords in input sequence
    attention_mask: torch.Tensor # determine which elements are actually padding
    token_type_ids: torch.Tensor # associated with Next-Sentence-Prediction Task (we ignore)
    
def tokenize(sentence: str) -> TokenizerOutput: 
    return tokenizer(sentence, return_tensors="pt")

def tokens_to_str(tokens: TokenizerOutput) -> List[str]:
    return tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

# Used to show students how subword tokenization works
def tokenize_and_print(sentence: str):
    tokens = tokenize(sentence)
    strings = tokens_to_str(tokens)
    print(strings)

subword_output = Output()
subword_sentence_label = HTML(
    value="<b>Enter text:</b>",
    layout={'width': '70px'}
)
subword_input_text = Text(
    value="unbelievably rapidly flowing waters with bioluminescent organisms run to the seashore",
    layout={'width': '550px'}
)
subword_generate_button = Button(description="Generate Subwords", button_style='info')

def generate_subwords(_):
    sentence = subword_input_text.value
    tokens = tokenize(sentence)
    subwords = tokens_to_str(tokens)

    # Create an HTML representation with styled subwords
    html_subwords = " ".join([f'<span style="color: blue; font-weight: bold;">{token}</span>' for token in subwords])

    with subword_output:
        subword_output.clear_output()
        display(HTML(f"<p><b>Subwords:</b> {html_subwords}</p>"))

subword_generate_button.on_click(generate_subwords)

def display_subwords():
    display(VBox([HBox([subword_sentence_label,subword_input_text, subword_generate_button]), subword_output]))
