import numpy as np

# Function to load and process data
def load_and_process_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    # Process data to load the first 8 lines, skip 25, and repeat till end of file
    processed_data = []
    i = 0
    while i < len(data):
        processed_data.extend(data[i:i+8])
        i += 33  # Skip 25 lines after reading 8 lines
    
    # Further process each line to remove the first 2 characters and the last 11 characters
    processed_data = [line[2:-12].strip() for line in processed_data]

    # Make every entry a real number and make it square to go from amplitude to probability
    processed_data = [str(float(x)**2) for x in processed_data]
    # Normalize the data
    processed_data = [str(float(x)/sum([float(y) for y in processed_data])) for x in processed_data]
    
    return processed_data

# Save processed data to new files
def save_data(file_path, data):
    with open(file_path, 'w') as file:
        i = 0
        for line in data:
            file.write(f"{line}\n")
            i += 1
            if i % 8 == 0:
                file.write("\n")

# Load data from files
data_0 = load_and_process_data('./output_data/digit_zero_statevectors.txt')
data_1 = load_and_process_data('./output_data/digit_one_statevectors.txt')

save_data('../nn/dataset/0_dataset.dat', data_0)
save_data('../nn/dataset/1_dataset.dat', data_1)