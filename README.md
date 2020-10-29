# Federated Deep Unfolding for Sparse Recovery
This repository is for "Fed-CS" network proposed in the following paper:
Komal Krishna Mogilipalepu, Sumanth Kumar Modukuri, Amarlingam Madapu, Sundeep Prabhakar Chepuri, "Federated Deep Unfolding for Sparse Recovery", submitted to ICASSP2021, the pdf can be found at https://arxiv.org/abs/2010.12616.
The code is tested in Linux environment (Python: 3, Tensorflow: 1.15) with Nvidia GTX 2080Ti GPU.

The following work is inspired from the work in https://github.com/VITA-Group/ALISTA

* [Introduction](#introduction)
* [Run the codes](#run-the-codes)
	* [Generate problem files](#generate-problem-files)
		* [Synthetic Data](#synthetic-data)
		* [Real Data](#real-data)
	* [Run model](#Run-model)
		* [Run on Synthetic Data](#run-on-synthetic-data)
		* [Run on Real Data](#run-on-real-data)

## Introduction
In this work we propose a federated learning technique for deep algorithm unfolding with applications to sparse signal recovery and compressed sensing. We refer to this architecture as Fed-CS. Specifically, we unfold and learn the iterative shrinkage thresholding algorithm for sparse signal recovery without transporting to a central location, the training data distributed across many clients. We propose a layer-wise federated learning technique, in which each client uses local data to train a common model. Then we transmit only the model parameters of that layer from all the clients to the server, which aggregates these local models to arrive at a consensus model. The proposed layer-wise federated learning for sparse recovery is communication efficient and preserves data privacy. Through numerical experiments on synthetic and real datasets, we demonstrate Fed-CS's efficacy and present various trade-offs in terms of the number of participating clients and communications involved compared to a centralized approach of deep unfolding. 

## Run the codes

### Generate problem files

It contains, measurement matrix A with the specified dimention.
    
#### Synthetic Data
```
python utils/prob.py --M 250 --N 500 \
    --pnz 0.1 --SNR inf --con_num 0.0 --column_normalized True
```
#### Real Data

It generates the measurement matrix $\Psi$
```
python utils/prob.py --M 128 --N 256 \
	--pnz 0.1 --SNR inf --con_num 0.0 --column_normalized True
```
Explanation for the options:

* `--M`: the dimension of measurements.
* `--N`: the dimension of sparse signals.
* `--pnz`: the approximate of non-zero elements in sparse signals.
* `--SNR`: the signal-to-noise ratio in dB unit in the measurements. inf means noiseless setting.
* `--con_num`: the condition number. 0.0 (default) means the condition number will not be changed.
* `--column_normalized`: whether normalize the columns of the measurement matrix to unit l-2 norm.

The resultant file is saved at the path experiments/m250_n500_k0.0_p0.1_s40/prob.npz, where the prob.npz is the problem file. If you want to generate a problem file from an existing measurement matrix, which is a numpy array, use --load_A option. In this case, options --M and --N will be overwriiten by the shape of loaded matrix.

### Run model

#### Run on Synthetic Data
Training on synthetic data:
```
python3 main.py --task_type sc -g 0 \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
    --net LISTA -T 3 --scope LISTA --exp_id 0 --num_cl 2 --maxit 50
```
Testing on synthetic data:
```
python3 main.py --task_type sc -g 0 -t\
     --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
     --net LISTA -T 3 --scope LISTA --exp_id 0 --num_cl 2 --maxit 50
```
Explanation for the options (all optinos are parsed in config.py):

* `--task_type`: the task on which you will train/test your model. Possible values are:
    * `sc` - standing for normal simulated sparse coding algorithm;
    * `cs` - for natural image compressive sensing;
* `-g/--gpu`: the id of GPU used. GPU 0 will be used by default.
* `-t/--test`: option indicates training or testing mode. Use this option for testing.
* `-n/--net`: specifies the network to use.
* `-T`: the number of layers.
* `--scope`: the name of variable scope of model variables in TensorFlow.
* `--exp_id`: experiment id, used to differentiate experiments with the same setting.  
* `--num_cl`: Number of clients or users in the federation setting.
* `--maxit`: Number of local iterations at every client.
            
#### Run on Real Data

1. Download BSD500 dataset. Split into train, validation and test sets, sizes are optional.
2. Generate the tfrecords using:
```
python3 utils/data.py --task_type cs \
    --dataset_dir /path/to/your/[train,val,test]/folder \
    --out_dir path/to/the/folder/to/store/tfrecords \
    --out_file [train,val,test].tfrecords \
    --sensing ./experiments/m128_n256_k0.0_p0.1_sinf/prob.npz \
    --patch_size 16 \
    --patches_per_img 10000 \
    --suffix jpg
```
Training on real data:
```
python3 main.py --task_type cs -g 0 --train_file training_tfrecords_filename --val_file validation_tfrecords_filename \
    --M 128 --N 512 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
    --net LISTA_cs -T 3 --sensing ./experiments/m128_n256_k0.0_p0.1_sinf/prob.npz \
    --dict ./data/dictionary.npz --scope LISTA_cs --exp_id 0 --num_cl 2 --maxit 50
```
Testing on real data:
```
python3 main.py --task_type cs -g 0 -t --train_file training_tfrecords_filename --val_file validation_tfrecords_filename \
    --M 128 --N 512 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
    --net LISTA_cs -T 3 --sensing ./experiments/m128_n256_k0.0_p0.1_sinf/prob.npz \
    --dict ./data/dictionary.npz --scope LISTA_cs --exp_id 0 --num_cl 2 --maxit 50
                  
```
