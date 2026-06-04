import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob
import pandas as pd

from utils import *
import networks

np.random.seed(1)
eps = 1e-5

##############################################################################################
def infer(net_core=None, model_path=None,model_num=None,result_path=None,suffix_save=None,include_light=None, antig_path=None, antib_path=None,batch_size=1):

    def data_process(data, header=[''], seq_length=[300], min_seq_length=10,str_rep=''):
        seq_vecs = [[] for _ in range(len(header))]
        seq_max_length = 0
        out_data = pd.Series({})
        drop_idx = []
        for d in data:
            dtmp = d
            seq_num = len(d[header[0]])
            drop_name = ['rbd', 'not_rbd']
            for i in range(seq_num):
                seqs = [str(d[h].loc[i]) for h in header]
                # rbd_binding = any([dn not in d.columns or d[dn].loc[i] > 0 for dn in drop_name])
                # rbd_binding = any([dn not in d or d[dn].loc[i] > 0 for dn in drop_name])
                rbd_binding = True
                flag=str(d[header[0]].loc[i]).replace('_', str_rep).replace('\n', str_rep).replace('\t', str_rep).replace(' ', str_rep).isalpha()

                if all([len(s) > min_seq_length for s in seqs]) and rbd_binding and all([len(s) <= seq_length[0] for s in seqs]) and flag:
                    if True:
                        for j, seq in enumerate(seqs):
                            seq = seq.replace(' ', str_rep)
                            seq = seq.replace('_', str_rep)
                            seq = seq.replace('\n', str_rep)
                            seq = seq.replace('\t', str_rep)
                            seq_v = np.zeros([seq_length[j], 20])
                            seq_v[0:len(seq), :] = one_hot_encoder(s=seq)
                            seq_vecs[j].append([seq_v, seq_length[j] - len(seq), len(seq)])
                            seq_max_length = max(seq_max_length, len(seq))
                else:
                    drop_idx.append(i)
            dtmp.drop(dtmp.index[drop_idx], inplace=True)
            out_data = pd.concat([out_data, dtmp])
        print(seq_max_length)
        return seq_vecs, out_data


    restore_pre_train = True

    # suffix='*.xlsx' #'*.csv'

    shape_heavy = [320, 20]
    shape_light = [320, 20]
    shape_antig = [320, 20]

    print(antig_path)

    cross_ab_ag = not(antig_path == antib_path)

    # antig_data = read_files(antig_path, '*ESM.xlsx')
    antig_data = read_files(antig_path, 'None')

    # antib_data = read_files(antib_path, '*ESM.xlsx')
    antib_data = read_files(antib_path, 'None')


    
    if cross_ab_ag:
        antig_data = [df.drop_duplicates(subset=['variant_name','variant_seq','rbd'], keep='first').reset_index(drop=False) for df in antig_data]

        [seq_antig], antig_series = data_process(antig_data, ['variant_seq'], seq_length=[shape_antig[0]])
        if include_light:
            [seq_heavy, seq_light], antib_series = data_process(antib_data, ['Heavy', 'Light'], seq_length=[shape_heavy[0], shape_light[0]])
        else:
            [seq_heavy], antib_series = data_process(antib_data, ['Heavy'], seq_length=[shape_heavy[0], shape_light[0]])
            seq_light=[[np.zeros_like(X[0]),1,200] for X in seq_heavy]
    else:
        # [seq_antig], antig_series = data_process(antig_data, ['variant_seq'], seq_length=[shape_antig[0]])
        [seq_heavy, seq_light, seq_antig], antib_series = data_process(antib_data, ['Heavy', 'Light','variant_seq'], seq_length=[shape_heavy[0], shape_light[0],shape_antig[0]])
        antig_series=antib_series
        if not include_light:
            seq_light=[[np.zeros_like(X[0]),1,200] for X in seq_heavy]

    num_heavy_light = len(seq_heavy)
    num_antig = len(seq_antig)
    
    if not cross_ab_ag:
        assert num_heavy_light == num_antig, 'The number of antibodies and antigens should be the same'
        num_sample = num_heavy_light
        out_data = pd.concat(
            [antig_series.loc[antig_series.index].reset_index(), antib_series.loc[antib_series.index].reset_index()],
            # keys=['a', 'b'],
            axis=1,
        )
    else:
        num_sample = num_heavy_light * num_antig
        combinations = np.array(np.meshgrid(range(num_antig), range(num_heavy_light),indexing='ij')).reshape([2,-1])
        out_data = pd.concat(
            [antig_series.loc[antig_series.index[combinations[0]]].reset_index(), antib_series.loc[antib_series.index[combinations[1]]].reset_index()],
            # keys=['a', 'b'],
            axis=1,
        )

    # ===============================================================================
    input_heavy_seq = tf.placeholder(tf.float32, [None, *shape_heavy])
    input_light_seq = tf.placeholder(tf.float32, [None, *shape_light])
    input_antig_seq = tf.placeholder(tf.float32, [None, *shape_antig])

    net = net_core([shape_heavy, shape_light, shape_antig])


    pred_bind,_=net([input_heavy_seq,input_light_seq,input_antig_seq])

    # restore the trained weights
    # saver = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if restore_pre_train:
        saver.restore(sess, model_path + "_rbd_" + str(model_num) + ".tf")
    save_path = model_path + "_rbd_" + str(model_num) + ".tf"
    print(save_path)

    print('sample data:',num_sample,' heavy_light:',num_heavy_light,' antig:',num_antig)
    idx_antig = list(range(num_antig))
    idx_heavy_light = list(range(num_heavy_light))


    prob_array = []
    BATCH_SIZE = 256
    if cross_ab_ag:
        for ia, idx_a in enumerate(idx_antig):
            for batch_start in range(0, num_heavy_light, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_heavy_light)
                batch_idx = list(range(batch_start, batch_end))
                bs = len(batch_idx)
                # Repeat single antigen for the whole antibody batch
                antig_one = get_seq_data([seq_antig], [[idx_a]], rand_shift=False)
                antig_batch = np.repeat(antig_one, bs, axis=0)
                inferFeed = {
                    input_heavy_seq: get_seq_data([seq_heavy], [batch_idx], rand_shift=False),
                    input_light_seq: get_seq_data([seq_light], [batch_idx], rand_shift=False),
                    input_antig_seq: antig_batch,
                }
                prob_bind = sess.run([pred_bind], feed_dict=inferFeed)
                prob_array.append(prob_bind[0])
            if (ia + 1) % 100 == 0 or ia == 0:
                print(f'  Antigen {ia+1}/{num_antig}', flush=True)
    else:
        for batch_start in range(0, num_sample, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, num_sample)
            batch_idx = list(range(batch_start, batch_end))
            inferFeed = {input_heavy_seq: get_seq_data([seq_heavy], [batch_idx], rand_shift=False),
                        input_light_seq: get_seq_data([seq_light], [batch_idx], rand_shift=False),
                        input_antig_seq: get_seq_data([seq_antig], [batch_idx], rand_shift=False),
                        }
            prob_bind = sess.run([pred_bind], feed_dict=inferFeed)
            prob_array.append(prob_bind[0])
            # ===========================================================================================================
    prob_array=np.concatenate(prob_array,axis=0)
    out_data['pred_prob'] = prob_array

    if suffix_save=='.csv':
        writer = pd.DataFrame.to_csv
        writer(out_data, path_or_buf=result_path, sep=',', index=False)
    else:
        out_data.to_excel(result_path)
    sess.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        "-mn",
        help="network for training."
        "-mn for ",
        type=str,
        default="XBCR_net",
        required=False,
    )
    parser.add_argument(
        "--data_name",
        "-dn",
        help="data name for training."
        "-dn for ",
        type=str,
        # default="xbcr_train_folder",
        default="A1-A11_testset",
        # default="BNT162b2_testset",
        # default="guoyu_testset",
        # default="unseen_testset",
        required=False,
    )

    parser.add_argument(
        "--type",
        help="Training type, full or rbd or multi",
        # default="full",
        default="rbd",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--model_num",
        help="The model number.",
        type=int,
        default=0,
        # default=1,
    )

    parser.add_argument(
        "--include_light",
        help="include light or not.",
        type=int,
        default=1,
    )
    parser.add_argument('--model_path', type=str, default='data/Github/general_eval/Model/XBCR-net/models/binding/binding-XBCR_net/model')
    parser.add_argument('--suffix_save', type=str, default='.xlsx')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    # network setting
    net_core = networks.get_net(args.model_name)
    result_path = os.path.join("data","Github","general_eval","Data","bcr_seq","XBCR_net_binding",args.data_name, 'results_'+str(args.type)+'_'+str(args.model_name)+'-'+str(args.model_num) + args.suffix_save)
    antig_path = os.path.join("data","Github","general_eval","Data","bcr_seq","XBCR_net_binding",args.data_name)
    antib_path = os.path.join("data","Github","general_eval","Data","bcr_seq","XBCR_net_binding",args.data_name)
    infer(net_core=net_core,model_path=args.model_path,model_num=args.model_num,result_path=result_path,suffix_save=args.suffix_save,include_light=args.include_light,antig_path=antig_path,antib_path=antib_path,batch_size=args.batch_size)
    