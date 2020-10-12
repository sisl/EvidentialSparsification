cd VQVAE

python VQVAE/vqvae.py --data-folder data/miniimagenet --output-folder vqvae --dataset miniimagenet --device cuda

python VQVAE/pixelcnn_prior.py --data-folder data/miniimagenet --model VQVAE/models/vqvae/best.pt --output-folder pixelcnn_prior --dataset miniimagenet --device cuda --batch-size 32 --num-layers 20 --hidden-size-prior 128

python VQVAE/generated_dataset.py --data-folder data/miniimagenet --model VQVAE/models/vqvae/best.pt --prior VQVAE/models/pixelcnn_prior/best_prior.pt --output-folder generated --dataset miniimagenet --device cuda --num-layers 20 --hidden-size-prior 128 --batch-size 32

python classifier/train.py --classifier wide_resnet50_2 --dataset miniimagenet --data_dir data/miniimagenet --batch_size 128

python classifier/test.py --classifier wide_resnet50_2 --dataset miniimagenetgenerated --data_dir data/generated/dataset_softmax --batch_size 128

python classifier/test.py --classifier wide_resnet50_2 --dataset miniimagenetgenerated --data_dir data/generated/dataset_sparsemax --batch_size 128

python classifier/test.py --classifier wide_resnet50_2 --dataset miniimagenetgenerated --data_dir data/generated/dataset_dst --batch_size 128


