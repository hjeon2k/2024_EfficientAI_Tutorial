"""
    Tutorials on https://github.com/huggingface/peft/blob/main/docs/source/quicktour.md
"""
import torch

from datasets import load_dataset
from utils import Prompter

from peft import LoraConfig, TaskType
from peft import get_peft_model
from peft import AutoPeftModelForCausalLM

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
import transformers

def finetune(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_data: load_dataset,
    val_data: load_dataset,
    data_collator: transformers.DataCollatorForSeq2Seq,
    out_dir: str,
    inputs: str,
    ):
    # finetune the model with lora
    model.train()

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=4, lora_alpha=8, lora_dropout=0.05)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
            output_dir=out_dir,
            learning_rate=1e-3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
    )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
    )
    trainer.train()

    model.save_pretrained(out_dir)


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: str,
    ):
    # generate responses
    model = model.to("cuda")
    model.eval()
    inputs = tokenizer(inputs, return_tensors="pt")

    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])


def main():

    pretrained_dir = "/home/intern1/data/opt-350m"
    data_dir = "/home/intern1/data/alpaca_data_cleaned_archive.json"
    out_dir = "./out/opt-350m-lora"
    inputs = "Preheat the oven to 350 degrees and place the cookie dough"

    model = AutoModelForCausalLM.from_pretrained(pretrained_dir)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    generate(model=model, tokenizer=tokenizer, inputs=inputs)

    def tokenize(prompt, cutoff_len=548, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result


    def generate_and_tokenize_prompt(data_point):
        prompter = Prompter("alpaca")
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    val_set_size=1000
    data = load_dataset("json", data_files=data_dir)
    train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
    val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    finetune(model=model, tokenizer=tokenizer, train_data=train_data, val_data=val_data, data_collator=data_collator, out_dir=out_dir, inputs=inputs)

    model = AutoPeftModelForCausalLM.from_pretrained(out_dir)
    generate(model=model, tokenizer=tokenizer, inputs=inputs)


if __name__ == '__main__':
    main()


