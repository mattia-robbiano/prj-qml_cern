import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from IPython.display import clear_output
import json
import time
import warnings

from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
from qiskit_algorithms.utils import algorithm_globals

from functions import *

# Options:
# TRAINING_MODE: - "TRAIN" for training the model, new parameters will be saved in ./OPTIMIZATION/parameters.json
#                - "TEST" for loading the model, parameters will be loaded from ./OPTIMIZATION/parameters.json
# OUTPUT_MODE:   - "FULL" input images will be passed through the full autoencoder and reconstructed images 
#                   will be saved in ./output_data/ 
#                - "COMPRESSED" input images will be passed through the encoder and output statevectors
#                   will be saved in ./output_data/
#                - "COMPRESSED-check" input images will be passed through the encoder and output statevectors 
#                   will be passed through the decoder. The output statevectors will be saved in ./output_data/
# DEPTH:         - Depth of the ansatz circuit
# ITERATIONS:    - Number of iterations for the optimization algorithm
# SAMPLES_NUM:   - Number of samples produced

# num_latent:    - Number of latent qubits not tunable yet
# num_trash:     - Number of trash qubits not tunable yet


#Loading options and parameters
options_FILEPATH = "./options.json"

depth, ITERATIONS, TRAINING_MODE, OUTPUT_MODE, num_latent, num_trash, SAMPLES_NUM = options_setup(options_FILEPATH)

# Training (if needed)
if TRAINING_MODE == "TRAIN":
    training(num_latent, num_trash, depth, ITERATIONS)

# Loading parameters
print("Loding parameters from file: ./OPTIMIZATION/parameters.json")
with open("./OPTIMIZATION/parameters.json", "r") as file:
    parameters = json.load(file)
print()

if OUTPUT_MODE == "COMPRESSED" or OUTPUT_MODE == "COMPRESSED-check":
    LatentCircuit = LatentAE_Builder(num_latent, num_trash, depth)
    Decoder = Decoder_Builder(num_latent, num_trash, depth)
    test_images, test_labels = GetDatasetDigits(SAMPLES_NUM, draw=False)

    digit_one_statevectors = []
    digit_zero_statevectors = []

    i = 0
    for image, label in zip(test_images, test_labels):
        # Circuit pass: bind image parameters to LatentCircuit
        param_values = np.concatenate((image, parameters))
        latent_circuit = LatentCircuit.assign_parameters(param_values)

        # Obtain statevector from LatentCircuit
        output_sv_partial = Statevector.from_instruction(latent_circuit)
        
        if OUTPUT_MODE == "COMPRESSED-check":
            # Testing: pass the statevector to the Decoder circuit
            decoder_circuit = QuantumCircuit(num_latent + num_trash)
            decoder_circuit.initialize(output_sv_partial.data, range(num_latent + num_trash))

            # Bind parameters to Decoder circuit
            temp = Decoder.assign_parameters(parameters)
            decoder_circuit = decoder_circuit.compose(temp)

            # Compute the final statevector
            output_sv = Statevector.from_instruction(decoder_circuit).data

            output_sv = np.reshape(np.abs(output_sv) ** 2, (8, 4))
            output_sv = output_sv / np.linalg.norm(output_sv)
            output_sv = output_sv[:10]

            # Figure generation
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image.reshape(8, 4))
            ax1.set_title("Input Data")
            ax2.imshow(output_sv)
            ax2.set_title("Output Data")
            plt.savefig(f'./output_data/{1-i}_image.png')
            print(f'Image saved to file: ./output_data/{1-i}_image.png')

            # Saving data
            np.savetxt(f'./output_data/{1-i}_input.txt', image.reshape(8, 4), fmt='%f')
            np.savetxt(f'./output_data/{1-i}_output.txt', output_sv, fmt='%f')
            print(f'Data saved to file: ./output_data/{1-i}_output.txt')
            print(f'Data saved to file: ./output_data/{1-i}_input.txt')
        
        else:
            # Save the statevector
            if label == 1:
                digit_one_statevectors.append(output_sv_partial)
            else:
                digit_zero_statevectors.append(output_sv_partial)

        i += 1

    # Save statevectors to files
    with open('./output_data/digit_one_statevectors.txt', 'w') as f:
        for sv in digit_one_statevectors:
            np.savetxt(f, sv, fmt='%f')
            f.write('\n')

    with open('./output_data/digit_zero_statevectors.txt', 'w') as f:
        for sv in digit_zero_statevectors:
            np.savetxt(f, sv, fmt='%f')
            f.write('\n')

    print(f'Data saved to file: ./output_data/digit_one_statevectors.txt')
    print(f'Data saved to file: ./output_data/digit_zero_statevectors.txt')

if OUTPUT_MODE == "FULL":
    FullAutoEncoder = FullAE_Builder(num_latent, num_trash, depth)

    test_images, test_labels = GetDatasetDigits(2, draw=False)
    i = 0
    for image, label in zip(test_images, test_labels):
        # Circuit pass: bind image parameters to FullAutoEncoder
        param_values = np.concatenate((image, parameters))
        output_qc = FullAutoEncoder.assign_parameters(param_values)

        # Obtain statevector from FullAutoEncoder
        output_sv = Statevector(output_qc).data
        output_sv = np.reshape(np.abs(output_sv) ** 2, (8, 4))
        output_sv = output_sv / np.linalg.norm(output_sv)

        # Figure generation
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image.reshape(8, 4))
        ax1.set_title("Input Data")
        ax2.imshow(output_sv)
        ax2.set_title("Output Data")
        plt.savefig(f'./output_data/{1-i}_image.png')
        print(f'Image saved to file: ./output_data/{1-i}_image.png')

        # Saving data
        np.savetxt(f'./output_data/{1-i}_input.txt', image.reshape(8, 4), fmt='%f')
        np.savetxt(f'./output_data/{1-i}_output.txt', output_sv, fmt='%f')
        print(f'Data saved to file: ./output_data/{1-i}_output.txt')
        print(f'Data saved to file: ./output_data/{1-i}_input.txt')

        i += 1
