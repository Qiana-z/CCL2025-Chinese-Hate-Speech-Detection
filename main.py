import json
import os
import sys
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel

def prepare_dataset(datapath):

    # Load the data
    train_data = json.load(open(os.path.join(datapath, "train.json"), "r"))
    test_data = json.load(open(os.path.join(datapath, "test1.json"), "r"))

    # Split the training data into train and validation sets
    X = [item['content'] for item in train_data]
    Y = [item['output'] for item in train_data]

    sys.stdout.write("Number of training samples: " + str(len(X)))
    sys.stdout.write("Number of test samples: " + str(len(Y)))

    # Split the training data into train set and validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_test = [item['content'] for item in test_data]

    return X_train, Y_train, X_val, Y_val, X_test

if __name__ == "__main__":
    
    # device setting
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the dataset
    datapath = "data"
    X_train, Y_train, X_val, Y_val, X_test = prepare_dataset(datapath)

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = LlamaModel.from_pretrained("Qwen/Qwen2.5-7B").to(device) # AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

    # tokenization