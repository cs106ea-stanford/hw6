
### Standard Imports
import torch
import transformers

from typing import List, TypedDict

import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML, Button, Output, Text, Label
from IPython.display import clear_output

import matplotlib.pyplot as plt
import numpy as np

plt.ion()

suppress_warnings = False
if suppress_warnings:
    from transformers import logging
    logging.set_verbosity_error()

### Load Model and Tokenizer

from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
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

def calculate_attention(tokens: TokenizerOutput, layer: int = 0) -> np.ndarray:
    """
    Calculate the attention matrix for a given layer in the BERT model.

    Args:
        tokens: The tokenized input (output of the tokenizer).
        layer: The layer of the model from which to extract attention scores.
               Default is -1, which corresponds to the highest (last) layer.

    Returns:
        attention_mean_matrix: A 2D NumPy array with attention scores averaged across all heads.
    """
    outputs = model(**tokens)

    # Get attention scores from the specified layer
    attention = outputs.attentions[layer]  # Shape: [batch_size, num_heads, seq_length, seq_length]
    attention_mean_matrix = attention.mean(dim=1).squeeze().detach().numpy()

    return attention_mean_matrix

def draw_attention_grid(tokens: TokenizerOutput, attention_matrix: np.ndarray,
                            include_special_tokens: bool = False) -> None:
    # include_special_tokens determines whether or not to include the [CLS] and [SEP]
    # start of sequence and end of sequence tokens
                    
    # graphing based on code from Claude.AI
    
    token_strings = tokens_to_str(tokens)
    if not include_special_tokens:
        token_strings = token_strings[1:-1]
        attention_matrix = attention_matrix[1:-1,1:-1]

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_matrix, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention", rotation=-90, va="bottom")

    # Set up axes
    ax.set_xticks(np.arange(len(token_strings)))
    ax.set_yticks(np.arange(len(token_strings)))
    ax.set_xticklabels(token_strings)
    ax.set_yticklabels(token_strings)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(token_strings)):
        for j in range(len(token_strings)):
            text = ax.text(j, i, f"{attention_matrix[i, j]:.2f}",
                        ha="center", va="center", color="black")

    ax.set_title("BERT Attention Heatmap")
    ax.set_xlabel('Target Tokens')
    ax.set_ylabel('Source Tokens')
    fig.tight_layout()

    # Show the plot
    plt.show()
    
def process_and_show_map(input_string: str, include_special_tokens: bool = False, layer: int = 0) -> None:
    """
    Tokenize the input string, calculate the attention matrix for a specific layer,
    and display it as a heatmap.

    Args:
        input_string: The input text to process.
        include_special_tokens: Whether to include [CLS] and [SEP] tokens in the visualization.
        layer: The layer of the model from which to extract attention scores.
               Default is -1, which corresponds to the highest (last) layer.
               Pass in 0 for the lowest layer.
    """
    tokens = tokenize(input_string)
    attention_matrix: np.ndarray = calculate_attention(tokens, layer)
    draw_attention_grid(tokens, attention_matrix, include_special_tokens)

attention_output = Output()
attention_sentence_label = HTML(
    value="<b>Enter text:</b>",
    layout={'width': '70px'}
)
attention_input_text = Text(
    value="The tree bark had lichen growing on it",
    layout={'width': '550px'}
)
show_special_tokens = widgets.Checkbox(
    value=False,
    description='Show Special Tokens',
    indent=False
)
layer_selection = widgets.Dropdown(
    options=list(range(12)),  # Assuming 12 layers in BERT
    value=0,
    description='Layer:',
    layout={'width': '150px'}
)
attention_generate_button = Button(description="Generate Attention Map", button_style='info')

def generate_attention(_):
    with attention_output:
        clear_output(wait=True)
        input_text = attention_input_text.value
        include_special_tokens = show_special_tokens.value
        selected_layer = layer_selection.value
        tokens = tokenize(input_text)
        attention_matrix = calculate_attention(tokens, selected_layer)
        draw_attention_grid(tokens, attention_matrix, include_special_tokens)

attention_generate_button.on_click(generate_attention)

def display_attention():
    display(VBox([
        HBox([attention_sentence_label, attention_input_text, show_special_tokens]),
        # HBox([layer_selection]),
        HBox([attention_generate_button]),
        attention_output
    ]))
    