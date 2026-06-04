import argparse
import os

os.getcwd()

from infer_rbd import infer
import networks


name = "binding"

parser = argparse.ArgumentParser()

#=======================================================================================================================
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
        default=name,
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
parser.add_argument(
    "--paired",
    help="Use paired mode (predict actual test pairs, not cross-product).",
    type=int,
    default=0,
)
#=======================================================================================================================
args = parser.parse_args()
#=======================================================================================================================
model_num=args.model_num
suffix_save='.csv'
# suffix_save='.xlsx'

model_name=args.model_name
data_name=args.data_name
include_light=args.include_light

# network setting
net_core = networks.get_net(model_name)

os.getcwd()
print(os.getcwd())
model_path=os.path.join('.','models',data_name,data_name+'-'+model_name,'model')
data_path=os.path.join('.','data',data_name)
print('model:',model_path,'  data:',data_path)
print(os.path.abspath(data_path))

if args.paired:
    # Paired mode: predict actual test pairs directly (no cross-product)
    paired_path = os.path.join(data_path, 'test', 'paired')
    antig_path = paired_path
    antib_path = paired_path  # same path → cross_ab_ag=False in pred_save
    print(f'Paired mode: {paired_path}')
else:
    # Cross-product mode: predict all antibody × antigen combinations
    antig_path = os.path.join(data_path,'test','ag_to_pred')
    antib_path = os.path.join(data_path,'test','ab_to_pred')

result_path = os.path.join('.', 'data', data_name, 'test','results', 'results_'+str(args.type)+'_'+str(model_name)+'-'+str(model_num) + suffix_save)

#=======================================================================================================================
infer(net_core=net_core, model_path=model_path,model_num=model_num,result_path=result_path, suffix_save=suffix_save,include_light=include_light, antig_path=antig_path, antib_path=antib_path)

