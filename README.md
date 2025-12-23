# RNA: Entrenamiento de Redes Neuronales: MNIST y CIFAR-10

Basic PyTorch implementation for training on MNIST and CIFAR-10 datasets. Nothing fancy, just straightforward training loops with some hyperparameter tuning capabilities.

### Paper repo
You can find a paper about this project (here)[https://github.com/MiquelGomezCorral/RNA-MNIST-CIFAR-Paper]

## Setup

Pretty standard setup here. You'll need Python 3.12 or close enough:

```bash
python3.12 -m venv venv
source venv/bin/activate

# install the app module
pip install -e app/

# grab dependencies (using uv because it's faster)
pip install uv
uv pip install -r requirements.txt

# if you want to use notebooks
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

## Running Training

The main CLI is in [app/main.py](app/main.py). Two subcommands: `mnist` and `cifar`.

### MNIST

```bash
cd app
python main.py mnist -b 1024 -lr 0.002 -wd 1e-6 -e 200 -des "your experiment description"
```

Parameters:
- `-b, --batch_size`: batch size (default: 1024)
- `-lr, --lr`: learning rate (default: 0.002)
- `-wd, --weight_decay`: weight decay for regularization (default: 1e-6)
- `-e, --epochs`: training epochs (default: 200)
- `-des, --description`: experiment description for logging

### CIFAR-10

```bash
cd app
python main.py cifar -b 256 -lr 0.001 -wd 2e-6 -e 125 -des "your experiment description"
```

Similar parameters as MNIST but with different defaults that work better for CIFAR. You can also add:
- `-ls, --label_smoothing`: label smoothing factor (helps with overfitting)
- `-dr, --dropout`: dropout rate

## What's Inside

```
.
├── app
│   ├── __init__.py
│   ├── main.py                    # CLI entry point
│   ├── scripts
│   │   ├── cifar.py               # CIFAR-10 training loop
│   │   ├── __init__.py
│   │   └── mnist.py               # MNIST training loop
│   ├── setup.py                   # package setup
│   └── src
│       ├── CIFAR                  # CIFAR-specific models/utils
│       ├── config                 # configuration dataclasses
│       ├── data                   # dataset loaders
│       ├── MNIST                  # MNIST-specific models/utils
│       ├── models                 # shared model architectures
│       └── utils                  # general utilities
├── data
│   ├── cifar-10-batches-py        # CIFAR-10 dataset files
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   ├── cifar-10-python.tar.gz     # original archive
│   └── MNIST                      # MNIST dataset
│       └── raw
├── docs                           # assignment PDFs probably
│   ├── p1-1.pdf
│   └── p2-1.pdf
├── logs                           # experiment logs with scores
│   ├── config_score_83.4000.txt
│   ├── config_score_99.4700.txt
│   └── ...                        # lots of these
├── models                         # saved model checkpoints
│   ├── best_model.pth
│   ├── best_model-113475.pth
│   └── ...                        # one per experiment basically
├── notebooks                      # jupyter notebooks for analysis
├── example.env                    # environment variables template
├── LICENSE
├── README.md                      # you're reading it
└── requirements.txt               # python dependencies
```

## How It Works

Each training script loads the dataset, sets up the model (MLP for MNIST, CNN for CIFAR), and runs the training loop. Models get saved to `models/` when they beat the previous best validation accuracy. Training configs and results get logged to `logs/` with the final score in the filename for easy comparison.

The config system uses dataclasses so you can pass args from the CLI and they get converted into typed configuration objects. Pretty clean actually.

## Dependencies

Main ones:
- PyTorch (via torchvision)
- torchinfo (model summaries)
- tqdm (progress bars)
- pandas (data handling)
- maikol-utils (some custom utilities)

Check [requirements.txt](requirements.txt) for the full list.

## Results

If you're curious about performance, check the `logs/` folder. Filenames have the validation accuracy, so `config_score_99.4700.txt` means that run got 99.47% accuracy. MNIST scores are in the high 99s, CIFAR is in the high 80s to low 90s depending on the architecture and hyperparameters.

## License

Check the [LICENSE](LICENSE) file.
