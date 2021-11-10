# CZ4045 - NLP Assignment 2

## Question One

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
        ├── pre-trained-model                               # Pre-trained model for biLSTM
        └── self-trained-model
└── README.md
```
