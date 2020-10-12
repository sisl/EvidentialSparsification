cd VQVAE

python VQVAE/generated_dataset.py --data-folder data/miniimagenet --model VQVAE/models_pretrained/best_vqvae.pt --prior VQVAE/models_pretrained/best_prior.pt --output-folder generated --dataset miniimagenet --device cuda --num-layers 20 --hidden-size-prior 128 --batch-size 32

python classifier/train.py --classifier wide_resnet50_2 --dataset miniimagenet --data_dir data/miniimagenet --batch_size 128

python classifier/test.py --classifier wide_resnet50_2 --dataset miniimagenetgenerated --data_dir data/generated/dataset_softmax --batch_size 128

python classifier/test.py --classifier wide_resnet50_2 --dataset miniimagenetgenerated --data_dir data/generated/dataset_sparsemax --batch_size 128

python classifier/test.py --classifier wide_resnet50_2 --dataset miniimagenetgenerated --data_dir data/generated/dataset_dst --batch_size 128

