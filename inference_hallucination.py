# If you are using a different directory for your data, update the path below
DATA_DIR = "./data"

# To download the data uncomment and run the following line
# !bash ../download_data.sh $DATA_DIR

# If your working directory is not the GenIE fodler, include the path to it in your PATH variable to make the library
# available
import os
import sys

sys.path.append("../")

"""Load the Model"""
from genie.models import GeniePL
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ckpt_name = "./logs/runs/2024-01-21-00-11-32---train_t5-large-20k/checkpoints/model-epoch_003-step_79999-val_nll_loss_0.2020.ckpt"
model = GeniePL.load_from_checkpoint(checkpoint_path=ckpt_name, device=device)

# load dataset
import json
with open(os.path.join(DATA_DIR, "Hallucination", "D3_test_final_hallucination (1).json"), "r") as f:
    test_dataset = json.load(f)

sentences = [example["description"] for example in test_dataset.values()]
# num_relations = [len(example["meta_obj"]["non_formatted_wikidata_id_output"]) for example in test_dataset]
print("Number of sentences:", len(sentences))
# print("Max Number of relations:", max(num_relations))

results = []

for i, s in enumerate(sentences[:100]):
    override_models_default_hf_generation_parameters = {
        "num_beams": 15,
        "num_return_sequences": 15,
        "return_dict_in_generate": False,
        "output_scores": False,
        "seed": 123
    }
    output = model.sample(s,
                          **override_models_default_hf_generation_parameters)
    print(f"Finished sentence {i} t5-large")
    results.append(output)

with open("t5-large_results_D3_hallucination.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

ckpt_name = "./logs/runs/2024-01-21-11-37-17---train_t5-base-20k/checkpoints/model-epoch_003-step_19999-val_nll_loss_0.1929.ckpt"
model = GeniePL.load_from_checkpoint(checkpoint_path=ckpt_name, device=device)
results = []

for i, s in enumerate(sentences[:100]):
    override_models_default_hf_generation_parameters = {
        "num_beams": 15,
        "num_return_sequences": 15,
        "return_dict_in_generate": False,
        "output_scores": False,
        "seed": 123
    }
    output = model.sample(s,
                          **override_models_default_hf_generation_parameters)
    print(f"Finished sentence {i} t5-base")
    results.append(output)

with open("t5-base_results_D3_hallucination.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
