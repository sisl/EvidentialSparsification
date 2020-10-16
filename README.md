# Evidential Sparsification of Multimodal Latent Spaces in CVAEs

This repository contains the code for the paper: "Evidential Sparsification of Multimodal Latent Spaces in Conditional Variational Autoencoders" by Masha Itkina, Boris Ivanovic, Ransalu Senanayake, Mykel J. Kochenderfer, and Marco Pavone, presented at NeurIPS 2020.

The code runs the qualitative and quantitative training iteration experiments for the tasks of image generation and behavior prediction. The required dependencies are listed in dependencies.txt. 

To run the MNIST, FashionMNIST, and NotMNIST image generation experiments, please run the following files:

run_mnist.sh
run_fashion.sh
run_notmnist.sh

Note that the NotMNIST data has to be downloaded from: http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html and placed into "NotMNIST/data/archive/notMNIST_large/notMNIST_large/".

For the VQVAE image generation experiments, the miniImageNet dataset should be downloaded from: https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view. The "train.csv", "val.csv", and "test.csv" files should be placed into VQVAE/data/miniimagenet, and the image data should be copied to VQVAE/data/miniimagenet/images. As an example, Table 1 in the paper can be reproduced using the following commands: 

For training:

run_vqvae.sh

If you want to use the pre-trained models, download them from: https://drive.google.com/drive/folders/1MOM9WZQd3tB-aG-hlwhHZFrDehJqLsbE?usp=sharing. Then place "best_prior.pt" into VQVAE/VQVAE/models_pretrained/ and place "wide_resnet50_2.pt" into VQVAE/classifier/models. To evaluate the pre-trained models, run:

run_vqvae_pretrained.sh

For the behavior prediction experiments, please see the README.md in the "Behavior Prediction/" folder.
