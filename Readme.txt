# CZ4045 - NLP Assignment 2

## Question One

### Setup
**1. Set up a conda environment**
- Create **conda environment** via `conda create --name myenv`
- Activate the conda environment with `conda activate myenv`

**2. Install the relevant packages to be used in the conda environment**
- pytorch `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` or refer to [Pytorch Official Guide](https://pytorch.org/)
- matplotlib `conda install -c conda-forge matplotlib`

**3. Run the Python Notebooks**
- Launch Jupyter Notebook
- Run the Python Notebook file
- **FNNModel (with Dropout).ipynb:** Notebook that contains final FNN Model 

**4. Sample Output**
- The output from the Python Notebook is a model.
- Model with the best perplexity score is saved in the file `model_dropout.pt`

**5. Run the Anaconda Prompt**
- Launch Anaconda Prompt
- Change to directory `Q1`
- Run **generate.py** file using command line `python generate.py` to generate texts for part (vii) 
- Generated texts are saved in the text file `generated.txt`

**6. Other folders/files found in the directory**
- **data:** folder which contains dataset split into 3 text files - train, valid and test
- **data.py:** Python file that does data preprocessing
- **old_model.py:** original Python file that contains RNN Model provided in the base code
- **main.py:** original Python file provided in the base code
- **model.py:** Python file which contains FNN Model without the use of dropout
- **Q1_main.py:** Python file which contains code to train FNN Model in model.py

#### Running FNN Model with no dropout
To train the FNN Model with no dropout, one can do so using the Anaconda Prompt. During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the test dataset.

The model is trained using the following commands.
```
python Q1_main.py --cuda --epochs 6           # Train the FNN on Wikitext-2 with CUDA
python generate.py                         # Generate samples from the trained FNN model.
```

The `Q1_main.py` script accepts the following arguments: 
```
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU,
                        Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --ngram_size          size of ngram
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the
                        transformer model
```

## Question Two

### Setup
**1. Set up a conda environment**
- Create **conda environment** via `conda create --name myenv`
- Activate the conda environment with `conda activate myenv`

**2. Install the relevant packages used in the conda environment**
- pytorch `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` or refer to [Pytorch Official Guide](https://pytorch.org/)
- numpy `conda install numpy`
- matplotlib `conda install -c conda-forge matplotlib`

**3. Download GloVe embeddings and extract glove.6B.100d.txt into the data folder**
- `wget http://nlp.stanford.edu/data/glove.6B.zip`

**4. Run the Python Notebooks**
- Launch Jupyter Notebook
- Run each of the Python Notebook files
- **LSTM.ipynb:** Sample biLSTM code provided in the assignment
- **1CNNLayer.ipynb:** For part (iv) where we replaced the LSTM-based word-level encoder with CNN
- **2/3/4/5CNNLayer.ipynb:** For part (vi) where we experimented with 2/3/4/5-layered CNNs
- **1CNNLayer_with_Conv2d_Maxpool.ipynb and 1CNNLayer_with_Conv2d_no_maxpool.ipynb:** Compared 1CNN Model with and without Max Pooling
- **5CNNLayer Different Kernel.ipynb:** Experimented 5CNN with a different kernel size

**5. Sample Output**
- The output from running each file is a model
- Model name is found in `parameters['name']`
- The model will be saved in the `./models` folder

## File Organization
```
├── Q1
    ├── FNNModel (with Dropout).ipynb                       # Final FNN Model 
    ├── data.py                                             # Data Preprocessing 
    ├── generate.py                                         # Program to generate sentences with model
    ├── generated.txt                                       # Sample sentences generated with our FNN with Dropout model
    ├── old_model.py                                        # Original RNN Model
    ├── main.py                                             # Original Python file that runs original RNN Model
    ├── model.py                                            # FNN Model with no Dropout
    ├── Q1_main.py                                          # Python file that runs FNN Model with no dropout
    ├── data    
    |   ├── wikitext-2
    |       ├── train.txt
    |       ├── valid.txt
    |       └── test.txt
    └── model_dropout.pt                                    # Generated FNN Model
├── Q2
    ├── 1CNNLayer.ipynb                                     # 1-layered CNN for part (iv)
    ├── 1CNNLayer_with_Conv2d_Maxpool.ipynb                 # 1-layered CNN (maxpool) for part (iv)
    ├── 1CNNLayer_with_Conv2d_no_maxpool.ipynb              # 1-layered CNN (no maxpool) for part (iv)
    ├── 2CNNLayer.ipynb                                     # 2-layered CNN for part (vi)
    ├── 3CNNLayer.ipynb                                     # 3-layered CNN for part (vi)
    ├── 4CNNLayer.ipynb                                     # 4-layered CNN for part (vi)
    ├── 5CNNLayer Different Kernel.ipynb                    # 5-layered CNN with different kernel size
    ├── 5CNNLayer.ipynb                                     # 5-layered CNN for part (vi)
    ├── LSTM.ipynb                                          # Sample code for biLSTM given in assignment
    ├── data
    |   ├── eng.testa
    |   ├── eng.testb
    |   ├── eng.train
    |   ├── eng.train54019
    |   └── mapping.pkl
    └── models                                              # Models output from running python notebooks
        ├── 1-layer-CNN-model                               # 1-layered CNN model
        ├── 1-layer-CNN-model-conv2d                        # 1-layered CNN model (Conv2d)
        ├── 1-layer-CNN-model-conv2d-maxpool                # 1-layered CNN model (Conv2d with max pooling)
        ├── 2-layer-CNN-model                               # 2-layered CNN model
        ├── 3-layer-CNN-model                               # 3-layered CNN model
        ├── 4-layer-CNN-model                               # 4-layered CNN model
        ├── 5-layer-CNN-model                               # 5-layered CNN model
        ├── 5-layer-CNN-model-kernel                        # 5-layered CNN model with different kernel size
        ├── pre-trained-model                               # Pre-trained model for biLSTM by the original repository
        └── self-trained-model                              # Self-trained model for biLSTM
└── README.md
```
