# Text Generation with LSTM

This repository contains a text generation model using *LSTM (Long Short-Term Memory), built on **TensorFlow/Keras*, that can generate text based on any provided dataset. The model is flexible and can adapt to different text styles depending on the training data.

## Features

- *Custom Text Generation*: Generates text based on the dataset you provide.
- *Temperature Control*: Allows users to adjust the "creativity" of the generated text via temperature settings.
- *Pre-trained Model*: A pre-trained model (text_generator_best.keras) is provided for immediate use (trained on a sample dataset).

## Requirements

Before running the project, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow >= 2.0
- NumPy
- Keras (included with TensorFlow)

You can install the required dependencies by running the following:

bash
pip install tensorflow numpy


## Getting Started

### 1. Clone the Repository
bash
git clone https://github.com/your-username/LSTMCharacterModel.git
cd textgen-lstm


### 2. Prepare Your Dataset

- Replace the placeholder dataset file (input.txt) with your own text data. Ensure the file is a plain text file.
- The script processes the dataset to create sequences for training the model.

### 3. Training the Model

If you want to train the model from scratch, run:

bash
python train_model.py


This will train the LSTM model on the dataset, and the best model will be saved as text_generator_best.keras.

### 4. Generating Text

To generate text, simply use the following code in the Python console or a script:

python
from textgen import generate_text

# Generate text with a temperature of 0.6
generated_text = generate_text(length=300, temperature=0.6)

# Print the generated text
print(generated_text)


You can modify the temperature (ranging from 0.2 to 1.0) to adjust the randomness and creativity of the output.

## Model Architecture

- *LSTM*: The model uses an LSTM layer to capture sequential dependencies in the text data.
- *Embedding Layer*: An embedding layer converts the characters or tokens into dense vectors, improving memory efficiency and performance.
- *Softmax Output*: A softmax layer at the output produces a probability distribution over possible next characters or tokens.

## Example Output

Here’s an example of generated text using a temperature of 0.6 (trained on a sample dataset):

--- Generated Text (Temperature: 0.6) ---

In the beginning of knowledge, there lies an endless quest to 
understand the unknown, driven by curiosity and the search for truth. 
Each word forms a thread that weaves the fabric of learning…

---

## Contributing

Feel free to open an issue or submit a pull request if you want to contribute to this project. Contributions, bug reports, and suggestions are always welcome!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

The sample dataset used in this project is inspired by the Shakespeare dataset provided by TensorFlow. However, this repository is designed to be adaptable to any text dataset.