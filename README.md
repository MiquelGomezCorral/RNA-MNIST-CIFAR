# Deep Learning Lab
Some of the lab examples and exercises of my deep learning lectures. 

## Examples

+ Two different tasks:

  - [MNIST](http://yann.lecun.com/exdb/mnist/) and
  - [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) 

+ Two different toolkits:
  - [Keras](https://keras.io)
  - [Pythorch](https://pytorch.org)


## Recommendations

Use [Googe Colab](https://colab.research.google.com) to run the experiments in virtual machines equiped with GPUs just in case you don't have a GPU computer to run them

## Env

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install uv

uv pip install -r requirements.txt
uv pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118

uv pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```