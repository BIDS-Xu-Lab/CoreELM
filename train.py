import argparse
import os
from openelm.model import LlamaForEmbeddingLM, Gemma3ForEmbeddingLM
from openelm.utils import collate_function_dynamic_padding_llama, collate_function_dynamic_padding_gemma3
from datasets import Dataset
from transformers import TrainingArguments, AutoConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import torch

def main():
    parser = argparse.ArgumentParser(description="Train a embedding language model.")
    parser.add_argument("--datahome", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--save_steps", type=float, default=0.1, help="Steps between saving model checkpoints")
    parser.add_argument("--basemodel_path", type=str, default="initial_elm_model", help="Path to basemodel")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from, 'latest' will use most recent")

    args = parser.parse_args()

    # Load datasets
    training_dataset = Dataset.load_from_disk(os.path.join(args.datahome, "encoded_training_dataset"))
    dev_dataset = Dataset.load_from_disk(os.path.join(args.datahome, "encoded_validation_dataset"))

    # Determine the model class and collate function based on the model type
    config = AutoConfig.from_pretrained(args.basemodel_path)
    if config.model_type == "llama":
        model_class = LlamaForEmbeddingLM
        collate_fn = collate_function_dynamic_padding_llama
    elif config.model_type in ["gemma3", "gemma3_text"]:
        model_class = Gemma3ForEmbeddingLM
        collate_fn = collate_function_dynamic_padding_gemma3
    else:
        raise ValueError(f"ERROR: Model type {config.model_type} not supported")

    # Load the base model
    elm = model_class.from_pretrained(
        args.basemodel_path,
        torch_dtype=torch.bfloat16,
        device_map={"":torch.cuda.current_device()}
    )

    # Define PEFT config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj"],
        modules_to_save=["adapter"],
    )

    elm_lora = get_peft_model(elm, peft_config)
    print(elm_lora.print_trainable_parameters())

    # Calculate training steps
    # Account for multi-GPU training in effective batch size calculation
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"We will train the model using {world_size} process(es).")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    num_training_steps = (args.num_train_epochs * len(training_dataset)) // effective_batch_size

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.output_dir + "/logs",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        save_steps=args.save_steps,
        max_steps=num_training_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        remove_unused_columns=False,
        bf16=True,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        elm_lora,
        train_dataset=training_dataset,
        eval_dataset=dev_dataset,
        peft_config=peft_config,
        args=training_args,
        data_collator=collate_fn,
        max_seq_length=2048,
    )

    resume_checkpoint = None
    if args.resume_from_checkpoint == "latest":
        # when resume_checkpoint==True, trainer will resume from the latest checkpoint
        resume_checkpoint = True 
        print("Resuming from the latest checkpoint")
    elif args.resume_from_checkpoint:
        resume_checkpoint = args.resume_from_checkpoint
        print(f"Resuming from specified checkpoint: {resume_checkpoint}")

    # Start training
    trainer.train(resume_from_checkpoint=resume_checkpoint)

if __name__ == "__main__":
    main()