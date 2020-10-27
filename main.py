"""
Main script. Start running model from main.py.
"""

import os , sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!

# timing
import time
from datetime import timedelta
from scipy.io import loadmat

from config import get_config
import utils.prob as problem
import utils.data as data
import utils.train as train
from utils.train import save_trainable_variables
import numpy as np
import tensorflow as tf
try :
    from sklearn.feature_extraction.image \
            import extract_patches_2d, reconstruct_from_patches_2d
except Exception as e :
    pass


def setup_model(config , **kwargs) :
    untiedf = 'u' if config.untied else 't'
    coordf = 'c' if config.coord  else 's'

    if config.net == 'LISTA' :
        """LISTA"""
        config.model = ("LISTA_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf,exp_id=config.exp_id))
        from models.LISTA import LISTA
        model = LISTA (kwargs['A'], T=config.T, lam=config.lam,
                       untied=config.untied, coord=config.coord,
                       scope=config.scope, name=kwargs['name'])

    if config.net == 'LISTA_cs':
        """LISTA-CS"""
        config.model = ("LISTA_cs_T{T}_lam{lam}_llam{llam}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf,
                                 exp_id=config.exp_id))
        from models.LISTA_cs import LISTA_cs
        model = LISTA_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                          lam=config.lam, untied=config.untied,
                          coord=config.coord, scope=config.scope, name=kwargs['name'])

    config.modelfn = os.path.join(config.expbase, config.model)
    config.resfn = os.path.join(config.resbase, config.model)
    print ("model disc:", config.model)

    return model


############################################################
######################   Training    #######################
############################################################

def run_train(config) :
    if config.task_type == "sc":
        run_sc_train(config)
    elif config.task_type == "cs":
        run_cs_train(config)

def get_weights(model, sess):
    layers = model.vars_in_layer
    B_layers = [layers[i][0].eval(session=sess) for i in range(len(layers))]
    W_layers = [layers[i][1].eval(session=sess) for i in range(len(layers))]
    theta_layers = [layers[i][2].eval(session=sess) for i in range(len(layers))]

    weights = {}
    weights['B'] = B_layers
    weights['W'] = W_layers
    weights['theta'] = theta_layers
    return weights

def get_weights_cs(model, sess):
    layers = model.vars_in_layer
    B_layers = [layers[i][0].eval(session=sess) for i in range(len(layers)-1)]
    W_layers = [layers[i][1].eval(session=sess) for i in range(len(layers)-1)]
    theta_layers = [layers[i][2].eval(session=sess) for i in range(len(layers)-1)]
    B_layers.append(layers[len(layers)-1][0].eval(session=sess))
    W_layers.append(layers[len(layers)-1][1].eval(session=sess))
    theta_layers.append(layers[len(layers)-1][2].eval(session=sess))
    D = [layers[len(layers)-1][3].eval(session=sess)]
    weights = {}
    weights['B'] = B_layers
    weights['W'] = W_layers
    weights['theta'] = theta_layers
    weights['D'] = D
    return weights

def get_weight_obj(B_layers, W_layers, theta_layers):
    weights = {}
    weights['B'] = B_layers
    weights['W'] = W_layers
    weights['theta'] = theta_layers
    return weights

def get_weight_obj_cs(B_layers, W_layers, theta_layers,D,layer,k):
    weights = {}
    weights['B'] = B_layers
    weights['W'] = W_layers
    weights['theta'] = theta_layers
    if layer == k:
      weights['D'] = D
    return weights

def run_sc_train(config) :
    """Load problem."""
    if not os.path.exists(config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem(config.probfn)
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    y_, x_, y_val_, x_val_ = (
        train.setup_input_sc (
            config.test, p, config.tbs, config.vbs, config.fixval,
            config.supp_prob, config.SNR, config.magdist, **config.distargs))
    """Set up model."""
    global_model = setup_model (config, A=p.A, name='global')
    comm_rounds = 2
    num_clients = config.num_cl
    do_3_stage_training = True
    new_Layer_model = setup_model (config, A=p.A, name='layer')
    client_data = y_.shape[1]//num_clients
    client_val_data = y_val_.shape[1]//num_clients
    client_models_dict = {}
    client_stages_dict = {}
    for i in range(num_clients):
        client_models_dict[i] = setup_model(config,A=p.A, name='client%d'%(i))
        client_stages_dict[i] = train.setup_sc_training (
            client_models_dict[i], y_[:,i*client_data:(i+1)*client_data], x_[:,i*client_data:(i+1)*client_data],
            y_val_[:,i*client_val_data:(i+1)*client_val_data], x_val_[:,i*client_val_data:(i+1)*client_val_data], None,
            config.init_lr, config.decay_rate, config.lr_decay,i,do_3_stage_training)
    Layer_wise_lnmse = []
    with tf.Session (config=tfconfig) as sess:
        nmse_for_all_rounds = []
        sess.run (tf.global_variables_initializer ())
        for layer in range(global_model._T):
            print("Layer ", layer+1)
            for rounds in range(comm_rounds):
                global_layer = global_model.vars_in_layer[layer]
                global_weights = get_weight_obj(global_layer[0].eval(sess), global_layer[1].eval(sess), global_layer[2].eval(sess))
                client_weight_list = get_weight_obj([], [], [])
                for client in range(num_clients):
                    client_model = client_models_dict[client]
                    client_model.set_weights_at_layer(global_weights, layer, sess)
                    print('--------------------------------------------------------------')
                    print(f'Round: {rounds+1:02} | client no: {client+1:02}')
                    stages = client_stages_dict[client]
                    start = time.time ()
                    if do_3_stage_training:
                        for stage_number in range(layer*3, layer*3 + 3):
                            client_model.do_training_one_stage(sess, stages[stage_number],config.modelfn, config.scope,config.val_step, config.maxit, config.better_wait)    
                    else:
                        client_model.do_training_one_stage(sess, stages[layer],config.modelfn, config.scope,config.val_step, config.maxit, config.better_wait)
                    end = time.time ()
                    elapsed = end - start
                    print ("elapsed time of training = " + str (timedelta (seconds=elapsed)))
                    client_layers = client_model.vars_in_layer[layer]
                    client_weights = get_weight_obj(client_layers[0].eval(sess), client_layers[1].eval(sess), client_layers[2].eval(sess))
                    client_weight_list['B'].append(client_weights['B'])    
                    client_weight_list['W'].append(client_weights['W'])
                    client_weight_list['theta'].append(client_weights['theta'])
                new_weights = {}
                new_weights['B'] = tf.convert_to_tensor(np.mean(client_weight_list['B'],axis=0))
                new_weights['W'] = tf.convert_to_tensor(np.mean(client_weight_list['W'],axis=0))
                new_weights['theta'] = tf.convert_to_tensor(np.mean(client_weight_list['theta'],axis=0))
                global_model.set_weights_at_layer(new_weights,layer, sess)
            if do_3_stage_training:
                for layer_in in range(layer+1):
                    client_weight_list = get_weight_obj([], [], [])
                    for client in range(num_clients):
                        client_layer = client_models_dict[client].vars_in_layer[layer_in]
                        client_weights = get_weight_obj(client_layer[0].eval(sess), client_layer[1].eval(sess), client_layer[2].eval(sess))
                        client_weight_list['B'].append(client_weights['B'])    
                        client_weight_list['W'].append(client_weights['W'])
                        client_weight_list['theta'].append(client_weights['theta'])
                    new_weights = {}
                    new_weights['B'] = tf.convert_to_tensor(np.mean(client_weight_list['B'],axis=0))
                    new_weights['W'] = tf.convert_to_tensor(np.mean(client_weight_list['W'],axis=0))
                    new_weights['theta'] = tf.convert_to_tensor(np.mean(client_weight_list['theta'],axis=0))
                    new_Layer_model.set_weights_at_layer(new_weights,layer_in, sess)
                lnmse2 = run_sc_test1(config,sess,new_Layer_model)
                Layer_wise_lnmse.append(lnmse2[layer+1])
        np.savez('Layer_lnmse_'+str(num_clients)+"_"+str(config.maxit),Layer_wise_lnmse)
        print("Layer wise performance")
        print(Layer_wise_lnmse)
        if do_3_stage_training:
            new_global_model = setup_model (config, A=p.A, name='new_global')
            for layer in range(new_global_model._T):
                client_weight_list = get_weight_obj([], [], [])
                for client in range(num_clients):
                    client_layer = client_models_dict[client].vars_in_layer[layer]
                    client_weights = get_weight_obj(client_layer[0].eval(sess), client_layer[1].eval(sess), client_layer[2].eval(sess))
                    client_weight_list['B'].append(client_weights['B'])    
                    client_weight_list['W'].append(client_weights['W'])
                    client_weight_list['theta'].append(client_weights['theta'])
                new_weights = {}
                new_weights['B'] = tf.convert_to_tensor(np.mean(client_weight_list['B'],axis=0))
                new_weights['W'] = tf.convert_to_tensor(np.mean(client_weight_list['W'],axis=0))
                new_weights['theta'] = tf.convert_to_tensor(np.mean(client_weight_list['theta'],axis=0))
                new_global_model.set_weights_at_layer(new_weights,layer, sess)
            lnmse2 = run_sc_test1(config,sess,new_global_model)
            np.savez('lnmse_new_global'+str(num_clients)+"_"+str(config.maxit),lnmse2)
            print("New Global model performance")
            print(lnmse2)
            save_trainable_variables(sess ,config.modelfn,config.scope)


def run_cs_train (config) :
    """Load dictionary and sensing matrix."""
    Phi = np.load (config.sensing)['A']
    D   = np.load (config.dict)['arr_0']
    """Set up model."""
    global_model = setup_model (config, Phi=Phi, D=D, name="global")
    """Set up inputs."""
    y_, f_, y_val_, f_val_ = train.setup_input_cs(config.train_file,
                                                  config.val_file,
                                                  config.tbs, config.vbs)
    comm_rounds = 2
    num_clients = config.num_cl
    do_3_stage_training = True
    new_Layer_model = setup_model (config, Phi=Phi, D=D, name='layer')
    client_data = y_.shape[1]//num_clients
    client_val_data = y_val_.shape[1]//num_clients
    client_models_dict = {}
    client_stages_dict = {}
    for i in range(num_clients):
        client_models_dict[i] = setup_model(config, Phi=Phi, D=D, name='client%d'%(i))
        client_stages_dict[i] = train.setup_cs_training (
            client_models_dict[i], y_[:,i*client_data:(i+1)*client_data], f_[:,i*client_data:(i+1)*client_data],
            y_val_[:,i*client_val_data:(i+1)*client_val_data], f_val_[:,i*client_val_data:(i+1)*client_val_data], None,
            config.init_lr, config.decay_rate, config.lr_decay,config.lasso_lam,i,do_3_stage_training)
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    Layer_wise_lnmse = []
    print("stages ",len(client_stages_dict[0]))
    new_global_model = setup_model (config, Phi=Phi, D=D, name='new_global')
    with tf.Session (config=tfconfig) as sess:
        nmse_for_all_rounds = []
        sess.run (tf.global_variables_initializer ())
        for layer in range(global_model._T):
            print("Layer ", layer+1)
            for rounds in range(comm_rounds):
                global_layer = global_model.vars_in_layer[layer]
                if layer == global_model._T-1:
                  global_weights = get_weight_obj_cs(global_layer[0].eval(sess), global_layer[1].eval(sess), global_layer[2].eval(sess),global_layer[3].eval(sess),layer,global_model._T-1)
                  client_weight_list = get_weight_obj_cs([], [], [],[],layer,global_model._T-1)
                else:
                  global_weights = get_weight_obj_cs(global_layer[0].eval(sess), global_layer[1].eval(sess), global_layer[2].eval(sess),None,0,global_model._T-1)
                  client_weight_list = get_weight_obj_cs([], [], [],None,0,global_model._T-1)
                for client in range(num_clients):
                    client_model = client_models_dict[client]
                    client_model.set_weights_at_layer(global_weights, layer, sess)
                    print('--------------------------------------------------------------')
                    print(f'Round: {rounds+1:02} | client no: {client+1:02}')
                    stages = client_stages_dict[client]
                    start = time.time ()
                    if do_3_stage_training:
                      if layer == global_model._T-1:
                        for stage_number in range(layer*3, layer*3 + 3):
                            client_model.do_training_one_stage(sess, stages[stage_number],config.modelfn, config.scope,config.val_step, config.maxit, config.better_wait)    
                      else:
                        for stage_number in range(layer*3, layer*3 + 3):
                            client_model.do_training_one_stage(sess, stages[stage_number],config.modelfn, config.scope,config.val_step, config.maxit, config.better_wait)
                    else:
                        client_model.do_training_one_stage(sess, stages[layer],config.modelfn, config.scope,config.val_step, config.maxit, config.better_wait)
                    end = time.time ()
                    elapsed = end - start
                    print ("elapsed time of training = " + str (timedelta (seconds=elapsed)))
                    layers = client_model.vars_in_layer[layer]
                    if layer == global_model._T-1:
                      client_weights = get_weight_obj_cs(layers[0].eval(sess), layers[1].eval(sess), layers[2].eval(sess),layers[3].eval(sess),layer,global_model._T-1)
                      client_weight_list['B'].append(client_weights['B'])    
                      client_weight_list['W'].append(client_weights['W'])
                      client_weight_list['theta'].append(client_weights['theta'])
                      client_weight_list['D'].append(client_weights['D'])
                    else:
                      client_weights = get_weight_obj_cs(layers[0].eval(sess), layers[1].eval(sess), layers[2].eval(sess),None,0,global_model._T-1)
                      client_weight_list['B'].append(client_weights['B'])    
                      client_weight_list['W'].append(client_weights['W'])
                      client_weight_list['theta'].append(client_weights['theta'])
                new_weights = {}
                new_weights['B'] = tf.convert_to_tensor(np.mean(client_weight_list['B'],axis=0))
                new_weights['W'] = tf.convert_to_tensor(np.mean(client_weight_list['W'],axis=0))
                new_weights['theta'] = tf.convert_to_tensor(np.mean(client_weight_list['theta'],axis=0))
                if layer == global_model._T-1:
                  new_weights['D'] = tf.convert_to_tensor(np.mean(client_weight_list['D'],axis=0))
                global_model.set_weights_at_layer(new_weights,layer, sess)
        if do_3_stage_training:
            for layer in range(new_global_model._T):
              if layer == new_global_model._T-1:
                client_weight_list = get_weight_obj_cs([], [], [],[],layer,new_global_model._T-1)
              else:
                client_weight_list = get_weight_obj_cs([], [], [],None,0,new_global_model._T-1)
                for client in range(num_clients):
                    client_layer = client_models_dict[client].vars_in_layer[layer]
                    if layer == new_global_model._T-1:
                      client_weights = get_weight_obj_cs(client_layer[0].eval(sess), client_layer[1].eval(sess), client_layer[2].eval(sess),client_layer[3].eval(sess),layer_in,global_model._T-1)
                      client_weight_list['B'].append(client_weights['B'])    
                      client_weight_list['W'].append(client_weights['W'])
                      client_weight_list['theta'].append(client_weights['theta'])
                      client_weight_list['D'].append(client_weights['D'])
                    else:
                      client_weights = get_weight_obj_cs(client_layer[0].eval(sess), client_layer[1].eval(sess), client_layer[2].eval(sess),None,0,global_model._T-1)
                      client_weight_list['B'].append(client_weights['B'])    
                      client_weight_list['W'].append(client_weights['W'])
                      client_weight_list['theta'].append(client_weights['theta'])
                new_weights = {}
                new_weights['B'] = tf.convert_to_tensor(np.mean(client_weight_list['B'],axis=0))
                new_weights['W'] = tf.convert_to_tensor(np.mean(client_weight_list['W'],axis=0))
                new_weights['theta'] = tf.convert_to_tensor(np.mean(client_weight_list['theta'],axis=0))
                if layer == new_global_model._T-1:
                  new_weights['D'] = tf.convert_to_tensor(np.mean(client_weight_list['D'],axis=0))
                new_global_model.set_weights_at_layer(new_weights,layer, sess)
            
            PSNR = run_cs_test1(config,sess,new_global_model)
            np.savez('PSNR'+str(new_global_model._M),PSNR)
            save_trainable_variables(sess ,config.modelfn,config.scope)
    # end of run_cs_train


############################################################
######################   Testing    ########################
############################################################

def run_test (config):
    if config.task_type == "sc":
        run_sc_test (config)
    elif config.task_type == "cs":
        run_cs_test (config)
        
def run_sc_test1(config,sess,model) :
    """
    Test model.
    """

    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Load testing data."""
    xt = np.load (config.xtest)
    """Set up input for testing."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    input_, label_ = (
        train.setup_input_sc (True, p, xt.shape [1], None, False,
                              config.supp_prob, config.SNR,
                              config.magdist, **config.distargs))

    xhs_ = model.inference (input_, None)

    nmse_denom = np.sum (np.square (xt))

    lnmse  = []
 
    # test model
    for xh_ in xhs_ :
        xh = sess.run (xh_ , feed_dict={label_:xt})

        # nmse:
        loss = np.sum (np.square (xh - xt))
        nmse_dB = 10.0 * np.log10 (loss / nmse_denom)
        # print (nmse_dB)
        lnmse.append (nmse_dB)

    res = dict (nmse=np.asarray  (lnmse))

    # print(lnmse)

    np.savez (config.resfn , **res)
    # end of test
    return lnmse

def run_sc_test(config) :
    """
    Test model.
    """

    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Load testing data."""
    xt = np.load (config.xtest)
    """Set up input for testing."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    input_, label_ = (
        train.setup_input_sc (config.test, p, xt.shape [1], None, False,
                              config.supp_prob, config.SNR,
                              config.magdist, **config.distargs))

    """Set up model."""
    model = setup_model (config , A=p.A,name='new_global')
    xhs_ = model.inference (input_, None)

    """Create session and initialize the graph."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())
        # load model
        model.load_trainable_variables (sess , config.modelfn)

        nmse_denom = np.sum (np.square (xt))
      
        lnmse  = []

        # test model
        for xh_ in xhs_ :
            xh = sess.run (xh_ , feed_dict={label_:xt})

            # nmse:
            loss = np.sum (np.square (xh - xt))
            nmse_dB = 10.0 * np.log10 (loss / nmse_denom)
            print (nmse_dB)
            lnmse.append (nmse_dB)


    res = dict (nmse=np.asarray  (lnmse))

    np.savez (config.resfn , **res)
    # end of test

def run_cs_test1 (config,sess,model) :
    from utils.cs import imread_CS_py, img2col_py, col2im_CS_py
    from skimage.io import imsave
    """Load dictionary and sensing matrix."""
    Phi = np.load (config.sensing) ['A']
    D   = np.load (config.dict)
    D = D["arr_0"]

    # loading compressive sensing settings
    M = Phi.shape [0]
    F = Phi.shape [1]
    N = D.shape [1]
    assert M == config.M and F == config.F and N == config.N
    patch_size = int (np.sqrt (F))
    assert patch_size ** 2 == F

    """Inference."""
    y_ = tf.placeholder (shape=(M, None), dtype=tf.float32)
    _, fhs_ = model.inference (y_, None)


    """Start testing."""
    # calculate average NMSE and PSRN on test images
    test_dir = './data/test_images/'
    test_files = os.listdir (test_dir)
    avg_nmse = 0.0
    avg_psnr = 0.0
    overlap = 0
    stride = patch_size - overlap
    out_dir = "./data/recon_images"
    if 'joint' in config.net :
        D = sess.run (model.D_)
    for test_fn in test_files :
        # read in image
        out_fn = test_fn[:-4] + "_recon_{}.png".format(config.sample_rate)
        out_fn = os.path.join(out_dir, out_fn)
        test_fn = os.path.join (test_dir, test_fn)
        test_im, H, W, test_im_pad, H_pad, W_pad = \
                imread_CS_py (test_fn, patch_size, stride)
        test_fs = img2col_py (test_im_pad, patch_size, stride)

        # remove dc from features
        test_dc = np.mean (test_fs, axis=0, keepdims=True)
        test_cfs = test_fs - test_dc
        test_cfs = np.asarray (test_cfs) / 255.0

        # sensing signals
        test_ys = np.matmul (Phi, test_cfs)
        test_ys = test_ys.astype(np.float32)
        num_patch = test_ys.shape [1]

        rec_cfs = sess.run (fhs_ [-1], feed_dict={y_: test_ys}) 
        rec_fs  = rec_cfs * 255.0 + test_dc

        # patch-level NMSE
        patch_err = np.sum (np.square (rec_fs - test_fs))
        patch_denom = np.sum (np.square (test_fs))
        avg_nmse += 10.0 * np.log10 (patch_err / patch_denom)

        rec_im = col2im_CS_py (rec_fs, patch_size, stride,
                                H, W, H_pad, W_pad)

        import cv2
        cv2.imwrite('%s'%out_fn, np.clip(rec_im, 0.0, 255.0))

        # image-level PSNR
        image_mse = np.mean (np.square (np.clip(rec_im, 0.0, 255.0) - test_im))
        avg_psnr += 10.0 * np.log10 (255.**2 / image_mse)

    num_test_ims = len (test_files)
    print ('Average Patch-level NMSE is {}'.format (avg_nmse / num_test_ims))
    print ('Average Image-level PSNR is {}'.format (avg_psnr / num_test_ims))

    return avg_psnr / num_test_ims
    # end of cs_testing


def run_cs_test (config) :
    from utils.cs import imread_CS_py, img2col_py, col2im_CS_py
    from skimage.io import imsave
    """Load dictionary and sensing matrix."""
    Phi = np.load (config.sensing) ['A']
    # D   = loadmat(config.dict)['D']
    # D = D.astype(np.float32)
    D   = np.load (config.dict)
    D = D["arr_0"]

    # loading compressive sensing settings
    M = Phi.shape [0]
    F = Phi.shape [1]
    N = D.shape [1]
    print(N)
    assert M == config.M and F == config.F and N == config.N
    patch_size = int (np.sqrt (F))
    assert patch_size ** 2 == F

    """Set up model."""
    model = setup_model (config, Phi=Phi, D=D,name='new_global')

    """Inference."""
    y_ = tf.placeholder (shape=(M, None), dtype=tf.float32)
    _, fhs_ = model.inference (y_, None)

    """Start testing."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:

        # graph initialization
        sess.run (tf.global_variables_initializer ())
        # load model
        model.load_trainable_variables (sess , config.modelfn)

        # calculate average NMSE and PSRN on test images
        test_dir = './data/test_images/'
        test_files = os.listdir (test_dir)
        avg_nmse = 0.0
        avg_psnr = 0.0
        overlap = 0
        stride = patch_size - overlap
        out_dir = "./data/recon_images"
        if 'joint' in config.net :
            D = sess.run (model.D_)
        for test_fn in test_files :
            # read in image
            out_fn = test_fn[:-4] + "_recon_{}.jpg".format(config.sample_rate)
            # print(out_fn)
            out_fn = os.path.join(out_dir, out_fn)
            test_fn = os.path.join (test_dir, test_fn)
            test_im, H, W, test_im_pad, H_pad, W_pad = \
                    imread_CS_py (test_fn, patch_size, stride)
            test_fs = img2col_py (test_im_pad, patch_size, stride)

            # remove dc from features
            test_dc = np.mean (test_fs, axis=0, keepdims=True)
            test_cfs = test_fs - test_dc
            test_cfs = np.asarray (test_cfs) / 255.0

            # sensing signals
            test_ys = np.matmul (Phi, test_cfs)
            num_patch = test_ys.shape [1]

            rec_cfs = sess.run (fhs_ [-1], feed_dict={y_: test_ys})
            rec_fs  = rec_cfs * 255.0 + test_dc

            # patch-level NMSE
            patch_err = np.sum (np.square (rec_fs - test_fs))
            patch_denom = np.sum (np.square (test_fs))
            avg_nmse += 10.0 * np.log10 (patch_err / patch_denom)

            rec_im = col2im_CS_py (rec_fs, patch_size, stride,
                                   H, W, H_pad, W_pad)
            import cv2
            cv2.imwrite('%s'%out_fn, np.clip(rec_im, 0.0, 255.0))
     
            # image-level PSNR
            image_mse = np.mean (np.square (np.clip(rec_im, 0.0, 255.0) - test_im))
            avg_psnr += 10.0 * np.log10 (255.**2 / image_mse)

    num_test_ims = len (test_files)
    print ('Average Patch-level NMSE is {}'.format (avg_nmse / num_test_ims))
    print ('Average Image-level PSNR is {}'.format (avg_psnr / num_test_ims))

    # end of cs_testing

#######################    Main    #########################

def main ():
    # parse configuration
    config, _ = get_config()
    # set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    if config.test:
        run_test (config)
    else:
        run_train (config)
    # end of main

if __name__ == "__main__":
    main ()

