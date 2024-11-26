import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import json
import time
import warnings

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, transpile, assemble
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from qiskit_machine_learning.neural_networks import SamplerQNN 
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals


def AnsatzBuilder(QubitNumber, Depth):
    return RealAmplitudes(QubitNumber, reps=Depth)

def SwaptestBuilder(TrashSpaceDimension):
    QubitNumber = 2*TrashSpaceDimension + 1
    QuantReg = QuantumRegister(QubitNumber,"q")
    ClassReg = ClassicalRegister(1,"c")
    SwaptestCircuit = QuantumCircuit(QuantReg, ClassReg)
    AuxiliaryQubitLable = QubitNumber-1
    
    for i in range(TrashSpaceDimension,2*TrashSpaceDimension): SwaptestCircuit.reset(i)
    SwaptestCircuit.h(AuxiliaryQubitLable)
    for i in range(TrashSpaceDimension):
        SwaptestCircuit.cswap(AuxiliaryQubitLable, i, TrashSpaceDimension+i)
    SwaptestCircuit.h(AuxiliaryQubitLable)
    SwaptestCircuit.measure(AuxiliaryQubitLable, ClassReg[0]) #HERE MESURE
    
    return SwaptestCircuit

def EncoderBuilder(InputStateDimension, EncodedStateDimension, Depth):
    LatentSpaceDimension = EncodedStateDimension
    TrashSpaceDimension = InputStateDimension - EncodedStateDimension
    ReferenceSpaceDimension = TrashSpaceDimension
    TotalQubitNumber = LatentSpaceDimension + TrashSpaceDimension + ReferenceSpaceDimension +1 #+1 (for auxiliary qubit)

    QuantReg = QuantumRegister(TotalQubitNumber,"q")
    ClassReg = ClassicalRegister(1,"c")
    Circuit = QuantumCircuit(QuantReg, ClassReg)
    Ansatz = AnsatzBuilder(LatentSpaceDimension+TrashSpaceDimension, Depth)
    Circuit.compose(Ansatz,range(0, InputStateDimension), inplace=True)
    Circuit.barrier()
    Circuit.compose(SwaptestBuilder(TrashSpaceDimension),range(LatentSpaceDimension, TotalQubitNumber), inplace=True)
    
    return Circuit

def Identity(x):
    return x

def ZeroMask(j, i):
    return [[i, j],[i - 1, j - 1],[i - 1, j + 1],[i - 2, j - 1],[i - 2, j + 1],[i - 3, j - 1],[i - 3, j + 1],[i - 4, j - 1],[i - 4, j + 1],[i - 5, j],
    ]

def OneMask(i, j):
    return [[i, j - 1], [i, j - 2], [i, j - 3], [i, j - 4], [i, j - 5], [i - 1, j - 4], [i, j]]

def GetDatasetDigits(num, draw=True):
    # Create Dataset containing zero and one
    train_images = []
    train_labels = []
    for i in range(int(num / 2)):

        # First we introduce background noise
        empty = np.array([algorithm_globals.random.uniform(0, 0.1) for i in range(32)]).reshape(8, 4) #original: 0.1
        # Now we insert the pixels for the one
        for i, j in OneMask(2, 6):
            empty[j][i] = algorithm_globals.random.uniform(0.9, 1) #original: 0.9
        train_images.append(empty)
        train_labels.append(1)
        if draw:
            plt.title("This is a One")
            plt.imshow(train_images[-1])
            plt.show()

    for i in range(int(num / 2)):
        # First we introduce background noise
        empty = np.array([algorithm_globals.random.uniform(0, 0.1) for i in range(32)]).reshape(8, 4)
        # Now we insert the pixels for the zero
        for k, j in ZeroMask(2, 6):
            empty[k][j] = algorithm_globals.random.uniform(0.9, 1)
        train_images.append(empty)
        train_labels.append(0)
        if draw:
            plt.imshow(train_images[-1])
            plt.title("This is a Zero")
            plt.show()

    train_images = np.array(train_images)
    train_images = train_images.reshape(len(train_images), 32)

    # Normalize the data
    for i in range(len(train_images)):
        sum_sq = np.sum(train_images[i] ** 2)
        train_images[i] = train_images[i] / np.sqrt(sum_sq)

    return train_images, train_labels

def TrainingCircuitBuilder(num_latent, num_trash, depth):

    fm = RawFeatureVector(2 ** (num_latent + num_trash))
    ae = EncoderBuilder(num_latent+num_trash, num_latent, depth)
    qc = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
    qc = qc.compose(fm, range(num_latent + num_trash))
    qc = qc.compose(ae)

    return fm, ae, qc

def cost_func_digits(params_values):
    # cost_func_digits: arguments: (params_values = parameters of the model)
    # The function computes the cost of the model for the given parameters. The cost is the sum of the 
    # probabilities of the model to predict the label 1 on the auxiliary qubit (see notebook for details).
    # The random training images are passed through the model and the cost is computed.

    global qnn
    global objective_func_vals

    train_images, __ = GetDatasetDigits(2,False)
    probabilities = qnn.forward(train_images, params_values)
    cost = np.sum(probabilities[:, 1]) / train_images.shape[0]

    print(f"Iteration {len(objective_func_vals)}: {cost}")

    # plotting
    clear_output(wait=True)
    objective_func_vals.append(cost)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)

    return cost

def training(num_latent, num_trash, depth, ITERATIONS):
    # training: arguments: (num_latent = number of qubits in the latent space, 
    #                       num_trash = number of qubits in the trash space, 
    #                       depth = depth of the ansatz RealAmplitudes, 
    #                       ITERATIONS = number of iterations) 
    # The function builds the training circuit through TrainingCircuitBuilder, composed by RawFeatureVector, 
    # the encoder and the swap test, and trains the model using the COBYLA optimizer. The parameters are saved 
    # to a file.

    global qnn
    global objective_func_vals

    fm, ae, qc = TrainingCircuitBuilder(num_latent, num_trash, depth)

    qnn = SamplerQNN(
        circuit=qc,
        input_params=fm.parameters,
        weight_params=ae.parameters,
        interpret=Identity,
        output_shape=2,)

    print("Optimization started:")
    opt = COBYLA(maxiter=ITERATIONS, disp=True)             # optimizer: COBYLA
    objective_func_vals = []                                # list to store the objective function values
    initial_point = np.random.rand(ae.num_parameters)       # initial random parameters
    plt.rcParams["figure.figsize"] = (12, 6)
    start = time.time()
    opt_result = opt.minimize(fun=cost_func_digits, x0=initial_point, bounds=[(0, 2 * np.pi)] * ae.num_parameters)
    elapsed = time.time() - start
    
    parameters = opt_result.x
    print(f"Fit in {elapsed:0.2f} seconds")
    print(f"Final cost: {opt_result.fun}")
    print("Saving loss function plot to file: OPTIMIZATION/ObjectiveFunction.png")
    plt.savefig('./OPTIMIZATION/ObjectiveFunction.png')

    with open("./OPTIMIZATION/parameters.json", "w") as file: # save parameters to file
        json.dump(list(opt_result.x), file)
    print("Parameters saved to file: ./OPTIMIZATION/parameters.json")
    print()

def FullAE_Builder(num_latent, num_trash, depth):
    fm = RawFeatureVector(2 ** (num_latent + num_trash))
    FullAutoEncoder = QuantumCircuit(num_latent + num_trash)
    FullAutoEncoder = FullAutoEncoder.compose(fm)
    ansatz_qc = AnsatzBuilder(num_latent + num_trash, depth)
    FullAutoEncoder = FullAutoEncoder.compose(ansatz_qc)
    FullAutoEncoder.barrier()
    FullAutoEncoder.reset(4)
    FullAutoEncoder.reset(3)
    FullAutoEncoder.barrier()
    FullAutoEncoder = FullAutoEncoder.compose(ansatz_qc.inverse())
    return FullAutoEncoder

def LatentAE_Builder(num_latent, num_trash, depth):
    fm = RawFeatureVector(2 ** (num_latent + num_trash))
    LatentCircuit = QuantumCircuit(num_latent + num_trash)
    LatentCircuit = LatentCircuit.compose(fm)
    ansatz_qc = AnsatzBuilder(num_latent + num_trash, depth)
    LatentCircuit = LatentCircuit.compose(ansatz_qc)
    LatentCircuit.barrier()
    LatentCircuit.reset(4)
    LatentCircuit.reset(3)
    LatentCircuit.barrier()

    #draw circuit and save it
    LatentCircuit.draw('mpl')
    plt.savefig('LatentCircuit.png')
    plt.show
    
    return LatentCircuit

def Decoder_Builder(num_latent, num_trash, depth):
    Decoder = QuantumCircuit(num_latent + num_trash)
    ansatz_qc = AnsatzBuilder(num_latent + num_trash, depth)
    Decoder = Decoder.compose(ansatz_qc.inverse())
    return Decoder

def options_setup(PATH):
    print("Loading options from file: ./options.json")
    with open(PATH, "r") as file:
        namelist = json.load(file)
    depth = int(namelist["DEPTH"])
    ITERATIONS = int(namelist["ITERATIONS"])
    TRAINING_MODE = str(namelist["TRAINING_MODE"])
    OUTPUT_MODE = str(namelist["OUTPUT_MODE"])
    num_latent = 3
    num_trash = 2

    print("Training mode:", TRAINING_MODE)
    print("Output mode:", OUTPUT_MODE)
    print()

    if TRAINING_MODE == "TRAIN":
        print("Depth:", depth)
        print("Iterations:", ITERATIONS)
        print("Do you want to continue? [y/n]")
        user_input = input()
        if user_input.lower() != "y":
            exit()
        else:
            print("Starting...")
            print()
            
    return depth, ITERATIONS, TRAINING_MODE, OUTPUT_MODE, num_latent, num_trash

