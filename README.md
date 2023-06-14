## Requirements

Our model was trained on GPU Tesla P40 of Nvidia DGX.  

- Python 3 (tested on 3.6.8)

- PyTorch (tested on 1.6.0)

- CUDA (tested on 10.1)

- transformers (tested on 2.1.0)


The pre-trained model we used is downloaded from [huggingface](https://huggingface.co/)

- Download the [albert-large-v2](https://huggingface.co/albert-large-v2/tree/main) pre-training model, included ```config.json``` ```pytorch_model.bin``` ```spiece.model``` 

- Put ```config.json``` ```pytorch_model.bin``` ```spiece.model``` into the ```pretrained_models/albert_large``` folder


