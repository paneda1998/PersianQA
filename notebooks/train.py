from datasets import load_dataset,DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
import sys
sys.path.insert(1, '/tf/local/baneshi/PersianQA/')


# model_checkpoint = "HooshvareLab/bert-fa-base-uncased"
max_length = 2048 # The maximum length of a feature (question and context)
doc_stride = 256 # The authorized overlap between two part of the context when splitting it is needed.
batch_size = 8
lr = 8e-5
epoch = 20


datasets = load_dataset("SajjadAyoubi/persian_qa")
datasets['train'][0]



model_checkpoint = "../models/bert-fa-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,)
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

tokenized_ds = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

args = TrainingArguments(
    output_dir=f"result"+"/"+model_checkpoint.split("/")[-1]+"__PQA_32",
    evaluation_strategy = "steps",
    save_steps = 2000,
    logging_strategy = "steps",
    logging_steps = 2000,
    logging_dir = f"result"+"/"+model_checkpoint.split("/")[-1]+"__PQA_32"+"/logs",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch,
    hub_strategy = "checkpoint",
    save_total_limit = 12,
    hub_model_id = "1",
    weight_decay=0.01) 

import transformers
grouped_params = model.parameters()
optimizer=transformers.AdamW(grouped_params,lr=lr)

# optimizer = transformers.Adafactor(
#     model.parameters(),
#     lr=4e-6,
#     eps=(1e-30, 1e-3),
#     clip_threshold=1.0,
#     decay_rate=-0.8,
#     beta1=None,
#     weight_decay=0.0,
#     relative_step=False,
#     scale_parameter=False,
#     warmup_init=False,
# )


scheduler=transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000, num_training_steps=20000)
optimizers = optimizer, scheduler

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['validation'],
    tokenizer=tokenizer,
#     optimizers=optimizers
)

# start training
trainer.train()
# trainer.train(resume_from_checkpoint="result/distil-bigbird-fa-zwnj__PQA_4/checkpoint-2000")