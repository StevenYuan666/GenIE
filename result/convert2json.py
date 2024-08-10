import json


def convert2json(path, target_file):
    # open target file
    with open(target_file, "r") as f:
        target_file = json.load(f)
    description = [example["description"] for example in target_file.values()]
    # 1. Read the jsonl file
    with open(path, "r") as f:
        lines = f.readlines()
    print("Number of lines:", len(lines))
    # 2. Convert the file to json
    json_data = {}
    count = 0
    for line in lines:
        temp_line = json.loads(line)
        temp_line = temp_line[0][0]
        generated_entities = temp_line.split("et>")
        entities = {}
        entity_names = []
        for i, entity in enumerate(generated_entities):
            if entity == "":
                continue
            subject = entity[entity.find("sub>") + 5: entity.find("rel>") - 1]
            relation = entity[entity.find("rel>") + 5: entity.find("obj>") - 1]
            object = entity[entity.find("obj>") + 5: entity.find("et>")]
            if subject not in entity_names:
                entity_names.append(subject)
                entities[entity_names.index(subject)] = {"entity name": subject}
            if relation != "instance of":
                entities[entity_names.index(subject)].update({relation: object})
            else:
                entities[entity_names.index(subject)].update({"type": object})
        json_data[count] = {"description": description[count], "entities": entities}
        count += 1
    # 3. Write the json file
    with open(f"{path[:-6]}_GenIE.json", "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    # convert2json("t5-base_results_D3_hallucination.jsonl")
    # convert2json("t5-large_results_D3_hallucination.jsonl")
    # convert2json("t5-base_results_REBEL.jsonl", "REBEL_test_100.json")
    # convert2json("t5-large_results_REBEL.jsonl", "REBEL_test_100.json")
    # convert2json("t5-base_results_TREX.jsonl", "TREX_test_100.json")
    # convert2json("t5-large_results_TREX.jsonl", "TREX_test_100.json")
    convert2json("t5-base_results_nyt.jsonl", "NYT_test_300.json")
    convert2json("t5-large_results_nyt.jsonl", "NYT_test_300.json")
    convert2json("t5-base_results_conll04.jsonl", "CONLL04_test_288.json")
    convert2json("t5-large_results_conll04.jsonl", "CONLL04_test_288.json")
