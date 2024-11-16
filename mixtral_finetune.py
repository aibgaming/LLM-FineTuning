import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1",
                                             load_in_4bit=True,
                                             torch_dtype=torch.float16,
                                             device_map="auto")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

tokenizer.pad_token = "!"  # Not EOS, will explain another time.

CUTOFF_LEN = 256  # Our dataset has short text
LORA_R = 4
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.2

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["w1", "w2", "w3"],  # Just targeting the MoE layers.
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

## Load the data
dataWP = np.load('TestAutri/datasets-brainteasers/WP_train 1.npy', allow_pickle=True)
dataSP = np.load('TestAutri/datasets-brainteasers/SP_train 1.npy', allow_pickle=True)

# Convert numpy arrays to lists (if they aren't already)
dataWP = dataWP.tolist()
dataSP = dataSP.tolist()

# Combine the datasets
data = dataWP + dataSP  # Concatenate the two lists

# Prepare inputs
def generate_prompt(item):
    question = item['question']
    answer = item['answer']
    distractor1 = str(item['distractor1'])
    distractor2 = str(item['distractor2'])
    distractor_unsure = str(item['distractor(unsure)'])
    
    # Create choice_list and reorder based on choice_order
    choice_list = [answer, distractor1, distractor2, distractor_unsure]
    choice_order = item['choice_order']
    ordered_choices = [choice_list[i] for i in choice_order]
    
    # Create messages as per the specified format
    sys_msg = "You are an assistant answering questions for a test."
    content = (
        "<s> [INST]" + sys_msg + "\n" +
        question + 
        "\nChoose one of the following answers and give an explanation below the answer.\n" +
        "\n".join(ordered_choices) + "[/INST]" + answer + "</s>"
    )
    
    return content

def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length"
    )

# Prepare the dataset
prompts = [generate_prompt(item) for item in data]
tokenized_data = [tokenize(prompt) for prompt in prompts]
input_ids = [td['input_ids'] for td in tokenized_data]

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({
    'input_ids': input_ids,
})

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.2,  # Increased weight decay
    max_grad_norm=0.3,  # Gradient clipping
    logging_steps=2,
    optim="adamw_torch",
    save_strategy="epoch",
    output_dir="mixtral-moe-lora-instruct-brainteasers"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    #eval_dataset=dev_dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

# Train the model
trainer.train()

# Save the fine-tuned model
# model.save_pretrained('./fine_tuned_model')
# tokenizer.save_pretrained('./fine_tuned_model')

print("Model fine-tuned and saved to ./fine_tuned_model")
