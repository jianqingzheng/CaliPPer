'''
Generalization evaluation
Jianqing Zheng
2024.11.16
'''


import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
import numpy as np
import pandas as pd
import os


#################### put implement funcs here ####################

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

def seq_filt(s,STR_REP=''):
    s = s.replace(' ', STR_REP)
    s = s.replace('\n', STR_REP)
    s = s.replace('\t', STR_REP)
    s = s.replace('_', STR_REP)
    s = s.replace('|', STR_REP)
    s = s.replace('>', STR_REP)
    s = s.replace('<', STR_REP)
    s = s.replace('=', STR_REP)
    s = s.replace('-', STR_REP)
    s = s.replace('.', STR_REP)
    return s

def column_filter(column, STR_REP=''):
    column = column.apply(seq_filt,STR_REP)
    return column

def one_hot_encoder(s, alphabet=ALPHABET, random_disturb_scale=0.0):
    """
    One hot encoding of a biological sequence.

    Parameters
    ---
    s: str, sequence which should be encoded
    alphabet: Alphabet object, downloaded from
        http://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC-module.html

    Example
    ---
    sequence = 'CARGSSYSSFAYW'
    one_hot_encoder(s=sequence, alphabet=IUPAC.protein)

    Returns
    ---
    x: array, n_size_alphabet, n_length_string
        Sequence as one-hot encoding
    """


    # Build dictionary
    # d = {a: i for i, a in enumerate(alphabet.letters)}
    d = {a: i for i, a in enumerate(alphabet)}

    # Encode
    x = np.zeros((len(s), len(d) + 1))
    # x[range(len(s)),[d[c] if c in alphabet.letters else len(d) for c in s]] = 1
    x[range(len(s)), [d[c] if c in alphabet else len(d) for c in s]] = 1
    # if any(x[:,len(d)]>0):
    #     print(s)

    # disturbance
    if random_disturb_scale > 0.:
        x += np.random.uniform(0, random_disturb_scale, x.shape)
        x = x / np.sum(x, axis=-1, keepdims=True)
    return x[:, :len(d)]


def read_table(file, headers=None):
    if isinstance(file, list):
        data = pd.concat([read_table(f) for f in file])
    else:
        try:
            data = pd.read_excel(file)
        except:
            data = pd.read_csv(file)
        # return data
    if headers is not None:
        for header in headers:
            if header not in data.columns:
                data[header] = data[header].apply(seq_filt)
    return data

def rand_str_generator(sequence_length_range=[20,80],alphabet=ALPHABET):
    # create random sequence with the random length within the specific range  
    return




if __name__ == '__main__':
    
    path_ref = os.path.join(os.path.dirname(__file__), "..", "Data/20241007-abag_sars-neu-new_trainmeta_neg_fold0_unique_randomseed-3.xlsx")
    path_qry = os.path.join(os.path.dirname(__file__), "..", "Data/20241007-abag_sars-neu-new_trainmeta_neg_fold0_unique_randomseed-3.xlsx")
    header_dict={
                "sequ": "Heavy",
                "label": "rbd",
                "pred": "rbd",
            }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ref', 
        type=str, 
        default=path_ref,
        help="reference data dir"
        )
    parser.add_argument(
        '--qry', 
        type=str, 
        default=path_qry,
        help="query data dir"
        )
    args = parser.parse_args()

    if args.ref=='random':
        ref_num = 100
        ref_data={}
        ref_data['sequence']=rand_str_generator(sequence_length_range=[10,100])
        ref_data['label']=np.random.randint(0,1,size=ref_num)
    else:
        ref_data = read_table(args.ref)

    if args.qry=='random':
        qry_num = 50
        qry_data={}
        qry_data['sequence']=rand_str_generator(sequence_length_range=[10,100])
        qry_data['label']=np.random.randint(0,1,size=ref_num)
    else:
        qry_data = read_table(args.qry)

        
    # print(qry_data['Heavy'])
    ref_data=ref_data.iloc[0:50]
    qry_data=qry_data.iloc[50:100]

    general_eval = General_Evaluator(ref_data=ref_data, qry_data=qry_data,header_dict=header_dict)
    eval_data = general_eval.eval_generalization(split_by="seq_edit_dist",eval_by="acc")
    print("Evaluation Data:", eval_data)


# ============================================================================
# Inlined from General_Eval/robust_combination_search.py (which is intentionally
# excluded from the published release). Used by report_generator.py.
# ============================================================================

def df_to_markdown(df) -> str:
    """Convert a pandas DataFrame to a markdown table without requiring tabulate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    headers = df.columns.tolist()
    header_line = '| ' + ' | '.join(str(h) for h in headers) + ' |'
    separator_line = '|' + '|'.join(['---' for _ in headers]) + '|'

    rows = []
    for _, row in df.iterrows():
        row_values = []
        for val in row:
            if isinstance(val, float):
                row_values.append(f'{val:.4f}')
            else:
                row_values.append(str(val))
        rows.append('| ' + ' | '.join(row_values) + ' |')

    return '\n'.join([header_line, separator_line] + rows)
