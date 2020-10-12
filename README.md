The attached code runs the qualitative and quantitative training iteration experiments for the MNIST and FashionMNIST datasets. The required dependencies are listed in dependencies.txt. To run the experiments, please run the following files:

run_mnist.sh
run_fashion.sh
run_notmnist.sh

Note that the NotMNIST data has to be downloaded from: http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html and placed into "NotMNIST/data/archive/notMNIST_large/notMNIST_large/".
For the behavior prediction experiments, please see the README.md in the "Behavior Prediction/" folder.

As an example, Table 1 for VQVAE (below) can be reproduced using the command below. The miniImageNet dataset should be downloaded from: https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view. The "train.csv", "val.csv", and "test.csv" files should be placed into VQVAE/data/miniimagenet, and the image data should be copied to VQVAE/data/miniimagenet/images.

For training:

run_vqvae.sh

For evaluation on pre-trained models:

run_vqvae_pretrained.sh

Table:

							Softmax 	Sparsemax 		Ours 		Original Images
Accuracy (%)         		20.688 		6.125			19.937		71.719
Top 5 Class Accuracy (%)  	47.750		17.500			47.875		90.625

