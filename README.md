Section 2:
    
        Generate Problem files for synthetic data:
            
            !python utils/prob.py --M 250 --N 500 \
                    --pnz 0.1 --SNR inf --con_num 0.0 --column_normalized True
            
        Generate Problem files for real data:
        
            !python utils/prob.py --M 128 --N 256 \
	                --pnz 0.1 --SNR inf --con_num 0.0 --column_normalized True
                    
        Explanation for the options:

            --M: the dimension of measurements.
            --N: the dimension of sparse signals.
            --pnz: the approximate of non-zero elements in sparse signals.
            --SNR: the signal-to-noise ratio in dB unit in the measurements. inf means noiseless setting.
            --con_num: the condition number. 0.0 (default) means the condition number will not be changed.
            --column_normalized: whether normalize the columns of the measurement matrix to unit l-2 norm.

        The generated will be saved to the experiments/m250_n500_k0.0_p0.1_s40/prob.npz. If you want to generate a problem from an existing 
        measurement matrix, which should be saved in Numpy npy file format, use --load_A option with the path to the matrix file. In this case, 
        options --M and --N will be overwriiten by the shape of loaded matrix.
                    
Section 3:

        Training on synthetic data:
        
            !python3 main.py --task_type sc -g 0 \
                    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
                    --net LISTA -T 3 --scope LISTA --exp_id 0 --num_cl 2 --maxit 50
                    
        Testing on synthetic data:
    
            !python3 main.py --task_type sc -g 0 -t\
                     --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
                     --net LISTA -T 3 --scope LISTA --exp_id 0 --num_cl 2 --maxit 50
                     
        Explanation for the options (all optinos are parsed in config.py):

            --task_type: the task on which you will train/test your model. Possible values are:
                    sc - standing for normal simulated sparse coding algorithm;
                    cs - for natural image compressive sensing;
            -g/--gpu: the id of GPU used. GPU 0 will be used by default.
            -t/--test option indicates training or testing mode. Use this option for testing.
            -n/--net: specifies the network to use.
            -T: the number of layers.
            --scope: the name of variable scope of model variables in TensorFlow.
            --exp_id: experiment id, used to differentiate experiments with the same setting.  
            --num_cl: Number of clients or users in the federation setting.
            --maxit: Number of local iterations at every client.
            
Section 4:

        1. Download BSD500 dataset. Split into train, validation and test sets as you wish.
        2. Generate the tfrecords using:
            
            !python3 utils/data.py --task_type cs \
                    --dataset_dir /path/to/your/[train,val,test]/folder \
                    --out_dir path/to/the/folder/to/store/tfrecords \
                    --out_file [train,val,test].tfrecords \
                    --sensing ./experiments/m128_n256_k0.0_p0.1_sinf/prob.npz \
                    --patch_size 16 \
                    --patches_per_img 10000 \
                    --suffix jpg
                    
        Training on real data:
        
            !python3 main.py --task_type cs -g 0 --train_file ./results/train50.tfrecords --val_file ./results/val50.tfrecords \
                    --M 128 --N 512 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
                    --net LISTA_cs -T 3 --sensing ./experiments/m128_n256_k0.0_p0.1_sinf/prob.npz \
                    --dict ./data/dictionary.npz --scope LISTA_cs --exp_id 0 --num_cl 2 --maxit 50
                    
        Testing on real data:
        
            !python3 main.py --task_type cs -g 0 -t --train_file ./results/train50.tfrecords --val_file ./results/val50.tfrecords \
                    --M 128 --N 512 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
                    --net LISTA_cs -T 3 --sensing ./experiments/m128_n256_k0.0_p0.1_sinf/prob.npz \
                    --dict ./data/dictionary.npz --scope LISTA_cs --exp_id 0 --num_cl 2 --maxit 50
                  
