import torch
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel
from openelm.tokens_map import TYPE_TOKEN_MAP_DICT
from openelm.model import LlamaForEmbeddingLM, Gemma3ForEmbeddingLM

##########
## Collate function for dynamic padding
##########
def collate_function_dynamic_padding_gemma3(examples):
    return collate_function_dynamic_padding(examples, model="gemma3")

def collate_function_dynamic_padding_llama(examples):
    return collate_function_dynamic_padding(examples, model="llama")

def collate_function_dynamic_padding(examples, model="llama"):
    input_ids = []
    labels = []
    # -1 because we will not include generation token in the resultant input_ids and labels
    max_length = max([len(example["input_ids"]) for example in examples]) - 1

    pad_token_id = TYPE_TOKEN_MAP_DICT[model]["pad_tok_id"]
    gen_token_id = TYPE_TOKEN_MAP_DICT[model]["gen_tok_id"]

    for example in examples:
        gen_tok_pos = example["input_ids"].index(gen_token_id)
        # create an array with max_length, filled up by pad_token_id
        input_ids_padded = torch.full((max_length,), pad_token_id, dtype=torch.long)
        # filter out generation token
        ids_without_gen_token = example["input_ids"][:gen_tok_pos] + example["input_ids"][gen_tok_pos+1:]
        input_ids_padded[:len(ids_without_gen_token)] = torch.tensor(ids_without_gen_token) 
        input_ids.append(input_ids_padded)

        labels_padded = torch.full((max_length,), -100, dtype=torch.long)
        # set prompt [:gen_tok_pos] as -100
        # set pads [len(ids_without_gen_token):] as -100
        # only learn target, which is [gen_tok_pos:len(ids_without_gen_token)]
        labels_padded[gen_tok_pos:len(ids_without_gen_token)] = input_ids_padded[gen_tok_pos:len(ids_without_gen_token)]
        labels.append(labels_padded)

    embs = [torch.tensor(x) for example in examples for x in example["domain_embeddings"]] 

    return {"input_ids": torch.stack(input_ids), "domain_embeddings": embs, "labels": torch.stack(labels)}

##########
## Helper function to load elm model
##########
def load_elm_model(configs):
    """
    Load the elm model from the config file.

    Args:
        configs: dictionary containing the config file

    Returns:
        tokenizer: tokenizer for the basemodel
        model_config: config for the basemodel
        lora_elm: basemodel with PEFT applied (elm = basemodel + PEFT)
    """

    # check if configs is a dictionary
    # if check if configs has backbone_model_path, peft_model_id, and device
    if not configs.get('backbone_model_path'):
        raise ValueError("backbone_model_path is required in the config file")
    if not configs.get('peft_model_id'):
        raise ValueError("peft_model_id is required in the config file")
    if not configs.get('device'):
        raise ValueError("device is required in the config file")

    print(f"Loading backbone model from {configs['backbone_model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(configs['backbone_model_path'])

    model_config = AutoConfig.from_pretrained(configs['backbone_model_path'])
    print(f"Backbone model type: {model_config.model_type}")

    # load elm model based on the base model type
    if model_config.model_type == "llama":
        model_class = LlamaForEmbeddingLM
    elif model_config.model_type in ["gemma3", "gemma3_text"]:
        model_class = Gemma3ForEmbeddingLM
    elm = model_class.from_pretrained(
        configs['backbone_model_path'], 
        torch_dtype=torch.bfloat16,
        device_map=configs['device'])

    print(f"Loading PEFT model from {configs['peft_model_id']}")
    lora_elm = PeftModel.from_pretrained(elm, configs['peft_model_id'])
    lora_elm = lora_elm.merge_and_unload()

    # ensure eos_token_id is set correctly based on the base model type
    if model_config.model_type == "llama":
        lora_elm.config.eos_token_id = tokenizer.eos_token_id
    elif model_config.model_type in ["gemma3", "gemma3_text"]:
        lora_elm.config.eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    print(f"EOS token ID: {lora_elm.config.eos_token_id}")
    
    return tokenizer, model_config, lora_elm

##########
## Helper function to generate input for inference
##########
def batched_inference_input_generator(prompt_template, emb_token, tokenizer, embeddings, batch_size=8, device="cuda"):
    """
    Generate input for inference in batches.

    Args:
        prompt_template: str
        emb_token: str
        tokenizer: AutoTokenizer
        embeddings: dictionary of numpy arrays
        batch_size: int
        device: str
    """
    # Check the number of embeddings and number of "emb_token" in the prompt template are matched
    if prompt_template.count("{emb_token}") != len(embeddings):
        raise ValueError("The number of embeddings and the number of 'emb_token' in the prompt template are not matched")
    
    # Check the number of rows in each embedding file are matched
    number_of_rows = len(embeddings[0])
    for j in range(len(embeddings)):
        if len(embeddings[j]) != number_of_rows:
            raise ValueError("The number of rows in the embedding files are not matched")
    
    # Process in batches
    for batch_start in range(0, number_of_rows, batch_size):
        batch_end = min(batch_start + batch_size, number_of_rows)
        batch_indices = range(batch_start, batch_end)
        
        # Generate input ids for this batch
        input_ids_list = []
        for i in batch_indices:        
            input_ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt_template.format(emb_token=emb_token)},
                ],
                return_tensors="pt", add_generation_prompt=True
            ).to(device)
            input_ids_list.append(input_ids)
        
        # Pad input ids for this batch
        max_length = max(ids.shape[1] for ids in input_ids_list)
        padded_inputs = []
        prompt_lengths = []
        for input_ids in input_ids_list:
            # Pad to max length within this batch
            prompt_length = input_ids.shape[1]
            prompt_lengths.append(prompt_length)
            padding_length = max_length - prompt_length
            if padding_length > 0:
                padding = torch.full((1, padding_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
                padded_input = torch.cat([input_ids, padding], dim=1)
            else:
                padded_input = input_ids
            
            padded_inputs.append(padded_input)
            
        # Stack all inputs into a batch
        batch_input_ids = torch.cat(padded_inputs, dim=0)
        # Convert embeddings to tensor (interleaved by row)
        batch_embs_tensor = []
        for i in batch_indices:
            for _, emb in embeddings.items():
                emb_tensor = torch.tensor(emb[i], dtype=torch.bfloat16).to(device)
                batch_embs_tensor.append(emb_tensor)
        
        yield batch_input_ids, batch_embs_tensor, prompt_lengths