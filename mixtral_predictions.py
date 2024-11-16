import torch
import transformers
from datasets import load_dataset, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import numpy as np

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Load the fine-tuned model from the checkpoint
checkpoint_dir = "mixtral-moe-lora-instruct-brainteasers/checkpoint-360"  #saved fine-tuned model checkpoint
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir,
                                             load_in_4bit=True,
                                             torch_dtype=torch.float16,
                                             device_map="auto")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

tokenizer.pad_token = "!"  # Not EOS, will explain another time.

CUTOFF_LEN = 256  # Our dataset has short text

# Function to generate prompt
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

# Function to tokenize prompt
def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors="pt"
    )

# Load the test data
test_data = np.load('TestAutri/datasets-brainteasers/WP_test 1.npy', allow_pickle=True)
test_data = test_data.tolist()  # Convert from numpy object array to list of dicts

# Initialize counters and storage for metrics
true_answers = []
predicted_answers = []

# Generate predictions
for item in test_data:
    prompt = generate_prompt(item)
    inputs = tokenize(prompt)
    
    # Move inputs to the appropriate device
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the predicted answer from the output text
    predicted_answer = output_text.split("[/INST]")[-1].strip()
    
    # Store true and predicted answers
    true_answers.append(item['answer'])
    predicted_answers.append(predicted_answer)

# Convert to numpy arrays
true_answers = np.array(true_answers)
predicted_answers = np.array(predicted_answers)

# Calculate metrics
accuracy = accuracy_score(true_answers, predicted_answers)
precision = precision_score(true_answers, predicted_answers, average='macro')
recall = recall_score(true_answers, predicted_answers, average='macro')
f1 = f1_score(true_answers, predicted_answers, average='macro')

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(true_answers, predicted_answers))
