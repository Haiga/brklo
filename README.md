# Deep Learning Rainging

### 1. Quick Start

```shell script
# clone the project 
git clone git@github.com:celsofranssa/DLR.git

# change directory to project folder
cd DLR/

# Create a new virtual environment by choosing a Python interpreter 
# and making a venv/ directory to hold it:
virtualenv -p python3 venv/

# activate the virtual environment using a shell-specific command:
source venv/bin/activate

# install dependecies
pip install -r requirements.txt

# setting python path
export PYTHONPATH=PYTHONPATH:$pwd

# (if you need) to exit virtualenv later:
deactivate
```

### 2. Datasets
Download the datasets from [kaggle](https://www.kaggle.com/datasets/celsofranssa/MSMARCO):

```
kaggle datasets download celsofranssa/MSMARCO -p resource/dataset/ --unzip
```
After downloading the datasets from it should be placed inside the `resources/datasets/` folder as shown below:

```
DLR/
├── resource
│   ├── dataset
│   │   ├── MSMARCO_RERANKING
│   │   │   ├── fold_0
│   │   │   │   ├── test.pkl
│   │   │   │   ├── train.pkl
│   │   │   │   └── val.pkl

        ...     

│   │   │   ├── fold_4
│   │   │   │   ├── test.pkl
│   │   │   │   ├── train.pkl
│   │   │   │   └── val.pkl
│   │   │   └── samples.pkl

        ..

│   │   └── MSMARCO_RETRIEVING
│   │       ├── fold_0
│   │       │   ├── test.pkl
│   │       │   ├── train.pkl
│   │       │   └── val.pkl

        ...

│   │       ├── fold_4
│   │       │   ├── test.pkl
│   │       │   ├── train.pkl
│   │       │   └── val.pkl
│   │       └── samples.pkl

```

### 3. Train, Predict, Eval
The following bash command fits the BERT encoder over EURLEX57K dataset using batch_size=64 and a single epoch.
```
nohup bash run.sh &
tail -f nohup.out
```
If all goes well the following output should be produced:
```
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs

  | Name     | Type                | Params
-------------------------------------------------
0 | encoder  | RerankerBERTEncoder | 109 M 
1 | dropout  | Dropout             | 0     
2 | cls_head | Sequential          | 6.1 K 
3 | loss     | CrossEntropyLoss    | 0     
4 | mrr      | RerankerMetric      | 0     
-------------------------------------------------

Sanity Checking DataLoader 0: 100%|██████████| 2/2 [00:00<00:00,  2.26it/s]
Epoch 0:  53%|██▎       | 58130/250000 [52:43<1:48:18,  12.35it/s, train_Loss=0.079]
```
