import os

training_dataset = "train"

for character_dir in os.listdir(training_dataset):
    for file_name in os.listdir(f'{training_dataset}/{character_dir}'):
        print(file_name)
