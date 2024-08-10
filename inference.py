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

dataset = "conll04"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ckpt_name = "/home/admin1/Documents/GenIE/logs/runs/2024-06-11-21-02-10---train_t5-large-CONLL04-full/checkpoints/model-epoch_001-step_1843-val_nll_loss_0.1426.ckpt"
model = GeniePL.load_from_checkpoint(checkpoint_path=ckpt_name, device=device)

# load dataset
import json
with open(os.path.join(DATA_DIR, dataset, "test_dataset.jsonl"), "r") as f:
    test_dataset = [json.loads(line) for line in f]

sentences = [example["input"] for example in test_dataset]
num_relations = [len(example["meta_obj"]["non_formatted_wikidata_id_output"]) for example in test_dataset]
print("Number of sentences:", len(sentences))
print("Max Number of relations:", max(num_relations))

results = []

for i, s in enumerate(sentences[:200]):
    override_models_default_hf_generation_parameters = {
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "output_scores": False,
        "seed": 123
    }
    output = model.sample(s,
                          **override_models_default_hf_generation_parameters)
    print(f"Finished sentence {i} t5-large")
    results.append(output)

with open(f"t5-large_results_{dataset}.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

ckpt_name = "/home/admin1/Documents/GenIE/logs/runs/2024-06-11-20-33-43---train_t5-base-CONLL04-full/checkpoints/model-epoch_003-step_3687-val_nll_loss_0.1254.ckpt"
model = GeniePL.load_from_checkpoint(checkpoint_path=ckpt_name, device=device)
results = []

for i, s in enumerate(sentences[:200]):
    override_models_default_hf_generation_parameters = {
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "output_scores": False,
        "seed": 123
    }
    output = model.sample(s,
                          **override_models_default_hf_generation_parameters)
    print(f"Finished sentence {i} t5-base")
    results.append(output)

with open(f"t5-base_results_{dataset}.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
