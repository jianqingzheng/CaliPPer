'''
Generalization evaluation
Jianqing Zheng
2024.11.16
'''


import argparse
try:
    import torch
except ImportError:
    torch = None
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, precision_recall_curve, average_precision_score,
                             matthews_corrcoef)
import numpy as np
import pandas as pd
from scipy.spatial import distance
import Levenshtein
import os
import time
from scipy.stats import pearsonr, spearmanr, linregress

from .utils import read_table, column_filter

_base_class = torch.nn.Module if torch is not None else object

class General_Evaluator(_base_class):
    def __init__(
            self,
            perf_type = ["acc","f1","prec","recall","aucroc","prc"],
            dist_type = ["seq_edit_dist","struct_msd","sequ_embed_dist","struct_embed_dist"],
            split_type = "equal",
            # split_type = "cluster",
            bin_num = 32,
            reduce_method='min',
            ref_data = None,
            qry_data = None,
            header_dict={
                "sequ": "sequence",
                "stru": "structure",
                "pred": "predict",
                "label": "label",
            },
            device='cuda',
            chain_cols=None,
            logdist_k=0.1,
            logdist_b=0.1,
            logdist_K=50,
            chain_weights='auto',
            entropy_scale=5.0,
            entropy_subsample=500,
            weight_formula='sigma_C',
            logdist_reduction='topk',
            logdist_n_hist_bins=50,
            logdist_blend_alpha=0.5,
            logdist_combine_order='reduce_first',
            ):
        super(General_Evaluator, self).__init__()
        self.header_dict=header_dict
        self.bin_num = bin_num
        self.perf_type = perf_type
        self.dist_type = dist_type
        self.split_type = split_type
        self.device = device
        self.chain_cols = chain_cols
        self.logdist_k = logdist_k
        self.logdist_b = logdist_b
        self.logdist_K = logdist_K
        self.chain_weights = chain_weights
        self.entropy_scale = entropy_scale
        self.entropy_subsample = entropy_subsample
        self.weight_formula = weight_formula
        self.logdist_reduction = logdist_reduction
        self.logdist_n_hist_bins = logdist_n_hist_bins
        self.logdist_blend_alpha = logdist_blend_alpha
        self.logdist_combine_order = logdist_combine_order
        self._chain_ratio_maps = {}
        self._chain_unique_refs = {}  # cached unique ref lists for combine-first
        self._chain_ref_probs = {}  # cached P(s) per chain for prob_weighted
        self._chain_ref_profiles = {}  # cached (P_ref, bin_edges) per chain for divergence
        self._chain_taus = {}  # cached tau per chain for info_gain
        self._chain_ref_freqs = {}  # cached frequency counts per chain for mutual_info
        # Define performance functions
        self.perf_func = {
            "acc": accuracy_score,
            "f1": f1_score,
            "prec": precision_score,
            "recall": recall_score,
            "aucroc": roc_auc_score,
            "prc": self._precision_recall_curve
        }

        # Define distribution distance functions
        self.dist_func = {
            "seq_edit_dist": self._sequ_edit_dist,
            "struct_msd": self._struct_msd,
            "sequ_embed_dist": self._sequ_embed_dist,
            "struct_embed_dist": self._struct_embed_dist,
            "logdist": self._logdist,
        }
        self.reduce_method=reduce_method
        # self.reduce_method='avg'
        # self.reduce_method='med'

        if "struct_embed_dist" in self.dist_type:
            self.sequ2struct_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
            self.sequ2struct_model.eval()
            self.batch_converter = alphabet.get_batch_converter()
            self.embed_layer_num=33

        self.set_ref_data(ref_data)
        self.set_qry_data(qry_data, dist_type) if qry_data is not None else None

        return

    def _transform_header(self,data_in=None,header_dict=None):
        header_dict=self.header_dict if header_dict is None else header_dict
        # check the input data is legal
        if not isinstance(data_in, pd.DataFrame):
            raise ValueError("Invalid data type. It should be a DataFrame.")
        data_out=pd.DataFrame()
        for key, value in header_dict.items():
            if isinstance(value, list):
                data_out[key] = data_in[value].agg(' '.join, axis=1)
            elif isinstance(value, str):
                if value in data_in.columns:
                    # transform the sequence data and filter the sequence
                    if key == "sequ":
                        data_out[key] = column_filter(data_in[value])
                    else:
                        data_out[key] = data_in[value]
                else:
                    raise ValueError(f"Missing required column: {key}")
        # Preserve individual chain columns for logdist multi-chain computation
        if self.chain_cols is not None:
            for col in self.chain_cols:
                if col in data_in.columns:
                    data_out[f'_chain_{col}'] = column_filter(data_in[col])
                else:
                    raise ValueError(
                        f"Chain column '{col}' not found in data. "
                        f"Available: {list(data_in.columns)}")
        return data_out

    def _drop_duplicates(self,data_in=None,drop_by="sequ"):
        # drop the duplicated data that appear in self.ref_data
        if data_in is None:
            raise ValueError("Input data cannot be None.")
        # Ensure data_in is a DataFrame
        if not isinstance(data_in, pd.DataFrame):
            raise ValueError("Invalid data type. It should be a DataFrame.")
        # Drop duplicates based on the reference data
        data_out = data_in[~data_in[drop_by].isin(self.ref_data[drop_by])]
        return data_out

    def set_ref_data(self,ref_data=None, header_dict=None):
        # change the default reference data (dict)
        header_dict=self.header_dict if header_dict is None else header_dict
        self.ref_data = self._transform_header(ref_data, header_dict)
        self._chain_ratio_maps = {}  # invalidate logdist cache
        self._chain_unique_refs = {}  # invalidate combine-first cache
        self._chain_ref_probs = {}   # invalidate ref probs cache
        self._chain_ref_profiles = {}  # invalidate divergence profile cache
        self._chain_taus = {}  # invalidate info_gain tau cache
        self._chain_ref_freqs = {}  # invalidate mutual_info freq cache
        # Auto-compute entropy weights from training data
        if isinstance(self.chain_weights, str) and self.chain_weights == 'auto' \
                and self.chain_cols is not None:
            self.chain_weights = self.compute_entropy_weights()
        return self.ref_data

    def compute_entropy_weights(self):
        """Compute information-theoretic chain weights from training data.

        Supports multiple formulas (controlled by self.weight_formula):

        Per-chain formulas:
        'var_log':      w_i = var_i / log(scale * N_train / N_unique_i)
        'sigma_H':      w_i = sigma_i * H_i
        'sigma_expDH':  w_i = sigma_i * exp(H_max_i - H_i)
        'sigma_DH':     w_i = sigma_i * (H_max_i - H_i)
        'sigma_C':      w_i = sigma_i * C_i   (C = Simpson index = sum P(s)^2)
        'sigma_Pmax':   w_i = sigma_i * max(P(s))
        'inv_H':        w_i = 1 / H_i
        'uniform':      w_i = 1 / N_chains

        Cross-chain formulas:
        'mi_chains':    w_i = sigma_i * MI(chain_i_dists; other_chains_dists)
        'cond_H':       w_i = sigma_i / H(chain_i | other_chains)
        'cov_chains':   w_i = sigma_i * mean(|cov(D_i, D_j)|) for j!=i
        'freq_ratio':   w_i = sigma_i * (N_train / N_unique_combos_including_i)
        'cross_H':      w_i = sigma_i * H(chain_i) / H(chain_i | epitope)

        Returns:
            numpy array of normalized weights, one per chain.
        """
        if self.chain_cols is None:
            return None

        chain_keys = [f'_chain_{c}' for c in self.chain_cols]
        n_train = len(self.ref_data)
        n_sub = min(self.entropy_subsample, n_train)
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(n_train, size=n_sub, replace=False)

        variances = []
        ratios = []
        empirical_entropies = []
        n_uniques = []
        simpson_indices = []
        p_maxes = []
        chain_sub_dists = []  # per-chain subsample distances for cross-chain formulas
        print(f"  Computing chain weights (formula={self.weight_formula}, "
              f"subsample={n_sub})...")

        for chain_key in chain_keys:
            ref_seqs = self.ref_data[chain_key].tolist()
            sub_seqs = [ref_seqs[i] for i in sub_idx]

            # Pairwise ratios: subsample vs full training
            _, _, train_rmap = compute_pairwise_ratios(sub_seqs, ref_seqs)

            # LogDist for each subsampled training sample
            train_dists = np.array([
                logdist_from_ratios(train_rmap[s], self.logdist_k,
                                    self.logdist_b, self.logdist_K)
                for s in sub_seqs])

            var_i = float(np.var(train_dists))
            n_unique = len(set(ref_seqs))
            ratio_i = n_train / n_unique

            # Empirical sequence entropy: H = -sum(P(s) * log(P(s)))
            from collections import Counter
            freq = Counter(ref_seqs)
            counts = np.array(list(freq.values()), dtype=float)
            p = counts / counts.sum()
            H_i = float(-np.sum(p * np.log(p)))
            C_i = float(np.sum(p ** 2))     # Simpson index
            Pmax_i = float(np.max(p))        # max probability

            variances.append(var_i)
            ratios.append(ratio_i)
            empirical_entropies.append(H_i)
            n_uniques.append(n_unique)
            simpson_indices.append(C_i)
            p_maxes.append(Pmax_i)
            chain_sub_dists.append(train_dists)
            print(f"    {chain_key}: var={var_i:.6f}, std={np.sqrt(var_i):.6f}, "
                  f"unique={n_unique}, ratio={ratio_i:.1f}, "
                  f"H_emp={H_i:.4f}, H_max={np.log(n_unique):.4f}")

        variances = np.array(variances)
        stds = np.sqrt(variances)
        ratios = np.array(ratios)
        H_emp = np.array(empirical_entropies)
        n_uniques_arr = np.array(n_uniques, dtype=float)
        C_arr = np.array(simpson_indices)
        Pmax_arr = np.array(p_maxes)

        if self.weight_formula == 'var_log':
            # Original formula: var / log(scale * ratio)
            denom = np.log(self.entropy_scale * ratios)
            if np.any(denom <= 0):
                print(f"  WARNING: log(scale*ratio) <= 0, "
                      f"falling back to uniform weights.")
                weights = np.ones(len(self.chain_cols)) / len(self.chain_cols)
            else:
                raw = variances / denom
                if raw.sum() <= 0:
                    weights = np.ones(len(self.chain_cols)) / len(self.chain_cols)
                else:
                    weights = raw / raw.sum()
        elif self.weight_formula == 'sigma_H':
            # std * empirical_entropy
            raw = stds * H_emp
            if raw.sum() <= 0:
                print(f"  WARNING: sigma*H <= 0, falling back to uniform weights.")
                weights = np.ones(len(self.chain_cols)) / len(self.chain_cols)
            else:
                weights = raw / raw.sum()
        elif self.weight_formula == 'sigma_expDH':
            # sigma * exp(H_max - H): concentration-weighted
            H_max = np.log(n_uniques_arr)
            delta_H = H_max - H_emp
            raw = stds * np.exp(delta_H)
            if raw.sum() <= 0:
                weights = np.ones(len(self.chain_cols)) / len(self.chain_cols)
            else:
                weights = raw / raw.sum()
        elif self.weight_formula == 'sigma_DH':
            # sigma * (H_max - H): entropy deficit
            H_max = np.log(n_uniques_arr)
            delta_H = H_max - H_emp
            raw = stds * delta_H
            if raw.sum() <= 0:
                weights = np.ones(len(self.chain_cols)) / len(self.chain_cols)
            else:
                weights = raw / raw.sum()
        elif self.weight_formula == 'sigma_C':
            # sigma * Simpson index (sum P(s)^2)
            raw = stds * C_arr
            if raw.sum() <= 0:
                weights = np.ones(len(self.chain_cols)) / len(self.chain_cols)
            else:
                weights = raw / raw.sum()
        elif self.weight_formula == 'sigma_Pmax':
            # sigma * max(P(s))
            raw = stds * Pmax_arr
            if raw.sum() <= 0:
                weights = np.ones(len(self.chain_cols)) / len(self.chain_cols)
            else:
                weights = raw / raw.sum()
        elif self.weight_formula == 'inv_H':
            # 1 / H (inverse entropy — no sigma involved)
            raw = 1.0 / np.maximum(H_emp, 1e-10)
            weights = raw / raw.sum()
        elif self.weight_formula == 'uniform':
            weights = np.ones(len(self.chain_cols)) / len(self.chain_cols)

        # --- Cross-chain weight formulas ---
        elif self.weight_formula == 'mi_chains':
            # sigma_i * MI(chain_i_dists; other_chains_dists)
            # Discretize per-chain distances into bins, compute MI
            n_chains = len(chain_keys)
            n_mi_bins = 10
            dist_matrix = np.column_stack(chain_sub_dists)  # (n_sub, n_chains)
            mi_values = np.zeros(n_chains)
            for i in range(n_chains):
                # Discretize chain_i distances
                d_i = dist_matrix[:, i]
                d_i_binned = np.digitize(
                    d_i, np.linspace(d_i.min(), d_i.max() + 1e-10, n_mi_bins + 1))
                # Concat other chains' binned distances as joint label
                other_cols = [j for j in range(n_chains) if j != i]
                if len(other_cols) == 0:
                    mi_values[i] = 0.0
                    continue
                other_binned = []
                for j in other_cols:
                    d_j = dist_matrix[:, j]
                    d_j_b = np.digitize(
                        d_j, np.linspace(d_j.min(), d_j.max() + 1e-10, n_mi_bins + 1))
                    other_binned.append(d_j_b)
                # Create joint label from other chains
                other_labels = other_binned[0].copy()
                for ob in other_binned[1:]:
                    other_labels = other_labels * (n_mi_bins + 1) + ob
                # Compute MI using histogram approach
                from sklearn.metrics import mutual_info_score
                mi_values[i] = mutual_info_score(d_i_binned, other_labels)
                print(f"    MI({chain_keys[i]}; others) = {mi_values[i]:.4f}")
            raw = stds * mi_values
            if raw.sum() <= 0:
                weights = np.ones(n_chains) / n_chains
            else:
                weights = raw / raw.sum()

        elif self.weight_formula == 'cond_H':
            # sigma_i / H(chain_i | other_chains)
            # H(X|Y) = H(X,Y) - H(Y), estimated via discretized distances
            n_chains = len(chain_keys)
            n_ce_bins = 10
            dist_matrix = np.column_stack(chain_sub_dists)
            cond_H_values = np.zeros(n_chains)
            for i in range(n_chains):
                d_i = dist_matrix[:, i]
                d_i_b = np.digitize(
                    d_i, np.linspace(d_i.min(), d_i.max() + 1e-10, n_ce_bins + 1))
                other_cols = [j for j in range(n_chains) if j != i]
                if len(other_cols) == 0:
                    cond_H_values[i] = max(H_emp[i], 1e-10)
                    continue
                other_binned = []
                for j in other_cols:
                    d_j = dist_matrix[:, j]
                    d_j_b = np.digitize(
                        d_j, np.linspace(d_j.min(), d_j.max() + 1e-10, n_ce_bins + 1))
                    other_binned.append(d_j_b)
                other_labels = other_binned[0].copy()
                for ob in other_binned[1:]:
                    other_labels = other_labels * (n_ce_bins + 1) + ob
                # H(chain_i, others) via joint histogram
                joint_labels = d_i_b * (other_labels.max() + 1) + other_labels
                from collections import Counter
                joint_freq = Counter(joint_labels)
                joint_p = np.array(list(joint_freq.values()), dtype=float)
                joint_p = joint_p / joint_p.sum()
                H_joint = float(-np.sum(joint_p * np.log(joint_p + 1e-15)))
                # H(others)
                other_freq = Counter(other_labels)
                other_p = np.array(list(other_freq.values()), dtype=float)
                other_p = other_p / other_p.sum()
                H_others = float(-np.sum(other_p * np.log(other_p + 1e-15)))
                cond_H_values[i] = max(H_joint - H_others, 1e-10)
                print(f"    H({chain_keys[i]}|others) = {cond_H_values[i]:.4f}")
            raw = stds / cond_H_values
            if raw.sum() <= 0:
                weights = np.ones(n_chains) / n_chains
            else:
                weights = raw / raw.sum()

        elif self.weight_formula == 'cov_chains':
            # sigma_i * mean(|cov(D_i, D_j)|) for j!=i
            n_chains = len(chain_keys)
            dist_matrix = np.column_stack(chain_sub_dists)
            cov_matrix = np.cov(dist_matrix.T)  # (n_chains, n_chains)
            cov_means = np.zeros(n_chains)
            for i in range(n_chains):
                off_diag = [abs(cov_matrix[i, j])
                            for j in range(n_chains) if j != i]
                cov_means[i] = np.mean(off_diag) if off_diag else 0.0
                print(f"    mean(|cov({chain_keys[i]}, others)|) = {cov_means[i]:.6f}")
            raw = stds * cov_means
            if raw.sum() <= 0:
                weights = np.ones(n_chains) / n_chains
            else:
                weights = raw / raw.sum()

        elif self.weight_formula == 'freq_ratio':
            # sigma_i * (N_train / N_unique_combos_including_i)
            # Count unique pairwise (chain_i, chain_j) combinations
            from collections import Counter
            n_chains = len(chain_keys)
            combo_counts = np.zeros(n_chains)
            for i in range(n_chains):
                seqs_i = self.ref_data[chain_keys[i]].tolist()
                # Average unique combos across all other chains
                pair_counts = []
                for j in range(n_chains):
                    if j == i:
                        continue
                    seqs_j = self.ref_data[chain_keys[j]].tolist()
                    pairs = set(zip(seqs_i, seqs_j))
                    pair_counts.append(len(pairs))
                combo_counts[i] = np.mean(pair_counts) if pair_counts else n_uniques[i]
                print(f"    N_unique_combos({chain_keys[i]}) = {combo_counts[i]:.0f}")
            raw = stds * (n_train / combo_counts)
            if raw.sum() <= 0:
                weights = np.ones(n_chains) / n_chains
            else:
                weights = raw / raw.sum()

        elif self.weight_formula == 'cross_H':
            # sigma_i * H(chain_i) / H(chain_i | epitope)
            # Requires variant_seq column for epitope grouping
            from collections import Counter
            n_chains = len(chain_keys)
            cross_H_values = np.zeros(n_chains)
            variant_key = self.header_dict.get('variant_seq', None)
            if variant_key is None:
                print(f"  WARNING: cross_H requires 'variant_seq' in header_dict. "
                      f"Falling back to uniform.")
                weights = np.ones(n_chains) / n_chains
            else:
                # variant_seq is stored as _chain_{variant_key} if in chain_cols,
                # otherwise look for it directly
                if f'_chain_{variant_key}' in self.ref_data.columns:
                    epitopes = self.ref_data[f'_chain_{variant_key}'].tolist()
                elif variant_key in self.ref_data.columns:
                    epitopes = self.ref_data[variant_key].tolist()
                else:
                    print(f"  WARNING: variant_seq column '{variant_key}' not found. "
                          f"Falling back to uniform.")
                    weights = np.ones(n_chains) / n_chains
                    epitopes = None
                if epitopes is not None:
                    # Group training data by epitope
                    from collections import defaultdict
                    ep_groups = defaultdict(list)
                    for idx_s in range(n_train):
                        ep_groups[epitopes[idx_s]].append(idx_s)
                    for i in range(n_chains):
                        seqs_i = self.ref_data[chain_keys[i]].tolist()
                        # H(chain_i | epitope) = sum_ep P(ep) * H(chain_i | ep)
                        cond_h = 0.0
                        for ep, indices in ep_groups.items():
                            p_ep = len(indices) / n_train
                            ep_seqs = [seqs_i[idx_s] for idx_s in indices]
                            freq = Counter(ep_seqs)
                            counts = np.array(list(freq.values()), dtype=float)
                            p = counts / counts.sum()
                            h_given_ep = float(-np.sum(p * np.log(p + 1e-15)))
                            cond_h += p_ep * h_given_ep
                        ratio_h = H_emp[i] / max(cond_h, 1e-10)
                        cross_H_values[i] = ratio_h
                        print(f"    H({chain_keys[i]})/H({chain_keys[i]}|epitope) "
                              f"= {H_emp[i]:.4f}/{cond_h:.4f} = {ratio_h:.4f}")
                    raw = stds * cross_H_values
                    if raw.sum() <= 0:
                        weights = np.ones(n_chains) / n_chains
                    else:
                        weights = raw / raw.sum()

        else:
            raise ValueError(f"Unknown weight_formula: {self.weight_formula}. "
                             f"Use 'var_log', 'sigma_H', 'sigma_expDH', "
                             f"'sigma_DH', 'sigma_C', 'sigma_Pmax', "
                             f"'inv_H', 'uniform', 'mi_chains', 'cond_H', "
                             f"'cov_chains', 'freq_ratio', or 'cross_H'.")

        w_str = ', '.join(f'{c}={weights[i]:.3f}'
                          for i, c in enumerate(self.chain_cols))
        print(f"  Chain weights ({self.weight_formula}): {w_str}")
        return weights

    def set_qry_data(self,qry_data=None,dist_type=None, header_dict=None, drop_duplicates=True):
        # change the default query data (dict)
        header_dict=self.header_dict if header_dict is None else header_dict
        self.dist_type = dist_type if dist_type==None else self.dist_type
        qry_data = self._transform_header(qry_data, header_dict)
        qry_data = self._drop_duplicates(qry_data) if drop_duplicates else qry_data
        self._chain_ratio_maps = {}  # invalidate logdist cache
        self._chain_unique_refs = {}  # invalidate combine-first cache
        self._chain_ref_probs = {}   # invalidate ref probs cache
        self._chain_ref_profiles = {}  # invalidate divergence profile cache
        self._chain_taus = {}  # invalidate info_gain tau cache
        self._chain_ref_freqs = {}  # invalidate mutual_info freq cache
        self.qry_data = self.eval_dist(qry_data, dist_type)
        return self.qry_data

    def eval_dist(self,qry_data=pd.DataFrame(),dist_type=None):
        # evaluate the distrbution difference between query data (dict) and the reference data (dict)
        qry_data = self.qry_data if qry_data.empty else qry_data
        dist_type = self.dist_type if dist_type==None else dist_type
        dist_diff = qry_data.copy()
        for dt in dist_type:
            dist_diff[dt] = self.dist_func[dt](qry_data,self.ref_data)
        return dist_diff


    def split_equally(self,qry_data=None,split_by="stru_msd",is_sorted=True):
        # equally split the query data wrt distribution difference
        if not is_sorted:
            sorted_data = qry_data.sort_values(by=split_by)
        else:
            sorted_data = qry_data
        # Determine the size of each bin
        bin_size = len(sorted_data) // self.bin_num
        # Split the data into bins
        bins = []
        bin_values = []
        numbers = []
        for i in range(self.bin_num):
            start_index = i * bin_size
            # Ensure the last bin captures any remaining data
            if i == self.bin_num - 1:
                end_index = len(sorted_data)
            else:
                end_index = (i + 1) * bin_size
            bins.append(sorted_data.iloc[start_index:end_index])
            bin_values.append(np.mean(bins[-1][split_by]))
            numbers.append(len(bins[-1]))
        return bins, bin_values, numbers

    def split_cluster(self,qry_data=None, min_bin=10,split_by="stru_msd"):
        # sort by clustering and split the query dataframe wrt distribution difference
        return

    def eval_perf(self,qry_data=None,eval_by="acc", thresh=0.5):
        """
        Evaluate the performance for each bin of query data.

        Parameters:
        - qry_data: A list of DataFrames, each representing a bin of query data.
        - eval_by: The performance metric to use (e.g., "acc", "f1").

        Returns:
        - A list of performance scores, one for each bin.
        """
        if qry_data is None or not isinstance(qry_data, list):
            raise ValueError("Invalid query data. It should be a list of DataFrames.")

        performance_scores = []

        # Iterate through each bin and calculate the performance metric
        for bin_data in qry_data:
            # Extract the true labels and predictions
            y_true = bin_data["label"].values
            y_pred = bin_data["pred"].values
            if thresh is not None:
                y_pred = (y_pred >= thresh).astype(int)

            # Check if there are valid entries to calculate performance
            if len(y_true) > 0 and len(y_pred) > 0:
                # Calculate the performance metric
                if eval_by in self.perf_func:
                    score = self.perf_func[eval_by](y_true, y_pred)
                    performance_scores.append(score)
                else:
                    raise ValueError(f"Invalid performance metric: {eval_by}")
            else:
                performance_scores.append(None)  # Append None if the bin has no valid data

        return performance_scores

    def eval_generalization(self,qry_data=None,ref_data=None,split_by="stru_msd",eval_by="acc",bin_num=20,dist_type=None,perf_type=None,drop_zero=True):
        # evaluate the generalization with split bin of query data and evaluate the query data in each bin ()
        qry_data = self.qry_data if qry_data==None else qry_data
        ref_data = self.ref_data if ref_data==None else ref_data
        bin_num = self.bin_num if bin_num==None else bin_num
        dist_type = self.dist_type if dist_type==None else dist_type
        
        dist_diff = self.qry_data if qry_data is None else self.eval_dist(qry_data=qry_data,dist_type=dist_type)

        perf_type = self.perf_type if perf_type==None else perf_type
        
        split_func = self.split_equally if self.split_type == "equal" else self.split_cluster
        # Sort the query data by the specified distribution difference
        sorted_data = dist_diff.sort_values(by=split_by)
        
        # Split the data into bins
        split_data, dist_values, numb_in_bin = split_func(sorted_data,split_by)
        # Evaluate the performance for each bin
        eval_data = self.eval_perf(split_data,eval_by)
        return [eval_data,dist_values,numb_in_bin]  


    def _sequ_edit_dist(self, qry_data, ref_data):
        """
        Compute the Levenshtein edit distance between the sequences in qry_data and ref_data.
        
        Parameters:
        - qry_data: DataFrame containing query sequences
        - ref_data: DataFrame containing reference sequences
        
        Returns:
        - edit_distances: A list of edit distances between each query sequence and the reference sequence
        """
        # Extract sequences from the DataFrame
        qry_sequences = qry_data['sequ']
        ref_sequences = ref_data['sequ']

        # Ensure both are lists of strings
        qry_sequences = qry_sequences.tolist()
        ref_sequences = ref_sequences.tolist()

        # Calculate edit distance for each query sequence against the first reference sequence
        # (or you can modify this logic to compare against all reference sequences if needed)
        
        edit_distances = [
            self._reduce_func([
                # self._levenshteinDistance(qry_seq, ref_seq)
                1-Levenshtein.ratio(qry_seq, ref_seq)
                for ref_seq in ref_sequences
            ])
            for qry_seq in qry_sequences
        ]

        return edit_distances

    def _levenshteinDistance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def _reduce_func(self,array=None,method=None):
        method = self.reduce_method if method==None else method
        if method=='min':
            reduced_value = np.min(array)
        elif method=='avg' or method=='mean':
            reduced_value = np.mean(array)
        elif method=='med' or method=='median':
            reduced_value = np.median(array)
        return reduced_value
    
    def _get_struct_embedding(self, seqs, reduce_axis=1):
        data_tuples = [(ss, ss) for ss in seqs]
        # print(max([len(ss) for ss in seqs]))
        _, _, batch_tokens = self.batch_converter(data_tuples)
        with torch.no_grad():
            results = self.sequ2struct_model(batch_tokens, repr_layers=[self.embed_layer_num], return_contacts=True)
            token_representations = results["representations"][self.embed_layer_num]
            return torch.max(token_representations, dim=reduce_axis).values

    def _struct_embed_dist(self, qry_data, ref_data,batch_size=50):
        qry_sequences = qry_data['sequ'].tolist()
        ref_sequences = ref_data['sequ'].tolist()

        # Precompute embeddings for reference sequences
        ref_embeddings = []
        for id_start in range(0, len(ref_sequences), batch_size):
            id_end = min(id_start + batch_size, len(ref_sequences))
            ref_embeddings.append(self._get_struct_embedding(ref_sequences[id_start:id_end]).cpu().numpy())
        ref_embeddings = np.concatenate(ref_embeddings, axis=0)
        # ref_embeddings = self._get_struct_embedding(ref_sequences).cpu().numpy()

        embed_distances = []
        for qry_seq in qry_sequences:
            qry_embed = self._get_struct_embedding([qry_seq]).cpu().numpy()
            distances = np.mean(np.square(qry_embed - ref_embeddings), axis=1)
            # distances = np.array([])
            # for id_start in range(0, len(ref_embeddings), batch_size):
            #     id_end = min(id_start + batch_size, len(ref_embeddings))      
            #     distances = np.concatenate([distances, np.mean(np.square(qry_embed - ref_embeddings[id_start:id_end]), axis=1)], axis=0)
            embed_distances.append(self._reduce_func(distances))

        return embed_distances



    def _struct_msd(self, qry_data, ref_data):
        # Implement structural mean squared deviation calculation
        return

    def _sequ_embed_dist(self, qry_data, ref_data):
        # Implement sequence embedding distance calculation
        return

    def _precision_recall_curve(self, qry_data):
        precision, recall, _ = precision_recall_curve(qry_data['label'], qry_data['pred'])
        return precision, recall

    def _compute_ref_profile_for_chain(self, chain_key):
        """Lazily compute and cache the within-training reference distance profile.

        Uses the same subsample approach as compute_entropy_weights() (RNG seed=42).
        """
        if chain_key in self._chain_ref_profiles:
            return self._chain_ref_profiles[chain_key]

        ref_seqs = self.ref_data[chain_key].tolist()
        n_train = len(ref_seqs)
        n_sub = min(self.entropy_subsample, n_train)
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(n_train, size=n_sub, replace=False)
        sub_seqs = [ref_seqs[i] for i in sub_idx]

        # Pairwise ratios: subsample vs full training
        _, _, train_rmap = compute_pairwise_ratios(sub_seqs, ref_seqs)

        P_ref, bin_edges = compute_ref_distance_profile(
            train_rmap, sub_seqs, self.logdist_k, self.logdist_b,
            n_hist_bins=self.logdist_n_hist_bins)

        self._chain_ref_profiles[chain_key] = (P_ref, bin_edges)
        return P_ref, bin_edges

    def _compute_tau_for_chain(self, chain_key):
        """Lazily compute and cache softmax temperature for info_gain.

        tau = std of all per-pair log-distances within the training subsample.
        """
        if chain_key in self._chain_taus:
            return self._chain_taus[chain_key]

        ref_seqs = self.ref_data[chain_key].tolist()
        n_train = len(ref_seqs)
        n_sub = min(self.entropy_subsample, n_train)
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(n_train, size=n_sub, replace=False)
        sub_seqs = [ref_seqs[i] for i in sub_idx]

        _, _, train_rmap = compute_pairwise_ratios(sub_seqs, ref_seqs)
        tau = compute_tau_from_training(train_rmap, sub_seqs,
                                        self.logdist_k, self.logdist_b)
        self._chain_taus[chain_key] = tau
        return tau

    def _logdist(self, qry_data, ref_data):
        """Multi-chain log-transformed distance.

        Computes per-chain LogDist for each chain in self.chain_cols (or 'sequ'
        if chain_cols is None), then combines via weighted average.

        Per-pair distance:  d(q, r) = log(k * (1 - Levenshtein.ratio(q, r) + b))

        Reduction (controlled by self.logdist_reduction):
          'topk':          D(q) = mean of top-K smallest d(q, r) values
          'freqw_topk':    score(q,r) = d(q,r) * P(r), then mean of top-K
                           most negative scores. Selects neighbors that are
                           both close AND frequent in training.
          'prob_weighted':  D(q) = sum_s P(s) * d(q, s)  (entropy format)
                           where P(s) = freq(s) / N_train
          'jsd':           Jensen-Shannon divergence vs reference profile
          'kl':            KL divergence vs reference profile
          'wasserstein':   Wasserstein-1 distance vs reference profile
          'info_gain':     Posterior entropy H(softmax(-d/tau))
          'mutual_info':   -MI(distance, ref_frequency)
          'blend':         alpha * z(topK_combined) + (1-alpha) * z(freqW→topK_combined)
                           Combines topK local distance with frequency-weighted
                           selection (neighbors that are both close AND common).
                           alpha controlled by self.logdist_blend_alpha (default 0.5).

        Multi-chain:  D_multi(q) = sum(w_i * D_chain_i(q))

        Combine order (controlled by self.logdist_combine_order):
          'reduce_first' (default): per-chain reduce then combine
          'combine_first': combine per-sample then reduce
        """
        # Dispatch to combine-first if requested (multi-chain only)
        if (self.logdist_combine_order == 'combine_first'
                and self.chain_cols is not None
                and len(self.chain_cols) > 1):
            return self._logdist_combine_first(qry_data, ref_data)

        divergence_methods = ('jsd', 'kl', 'wasserstein')

        if self.chain_cols is not None:
            chain_keys = [f'_chain_{c}' for c in self.chain_cols]
        else:
            chain_keys = ['sequ']

        # For 'blend', compute both topK and freqW→topK per chain, then combine
        if self.logdist_reduction == 'blend':
            return self._logdist_blend(qry_data, ref_data, chain_keys)

        all_chain_dists = []
        for chain_key in chain_keys:
            qry_seqs = qry_data[chain_key].tolist()
            ref_seqs = ref_data[chain_key].tolist()

            # Compute and cache pairwise ratios (and unique refs)
            if chain_key not in self._chain_ratio_maps:
                _, unique_ref, ratio_map = compute_pairwise_ratios(qry_seqs, ref_seqs)
                self._chain_ratio_maps[chain_key] = ratio_map
                self._chain_unique_refs[chain_key] = unique_ref

            ratio_map = self._chain_ratio_maps[chain_key]

            if self.logdist_reduction == 'freqw_topk':
                # Frequency-weighted top-K: selects neighbors both close AND common
                if chain_key not in self._chain_ref_probs:
                    self._chain_ref_probs[chain_key] = \
                        compute_ref_probs(ref_seqs)
                ref_probs = self._chain_ref_probs[chain_key]
                dists = distances_for_kb_freqw_topk(
                    qry_seqs, ratio_map,
                    self.logdist_k, self.logdist_b, self.logdist_K,
                    ref_probs)
            elif self.logdist_reduction == 'prob_weighted':
                # Compute and cache reference sequence probabilities
                if chain_key not in self._chain_ref_probs:
                    self._chain_ref_probs[chain_key] = \
                        compute_ref_probs(ref_seqs)
                ref_probs = self._chain_ref_probs[chain_key]
                dists = distances_for_kb_prob(qry_seqs, ratio_map,
                                              self.logdist_k, self.logdist_b,
                                              ref_probs)
            elif self.logdist_reduction in divergence_methods:
                P_ref, bin_edges = self._compute_ref_profile_for_chain(
                    chain_key)
                dists = distances_for_kb_divergence(
                    qry_seqs, ratio_map,
                    self.logdist_k, self.logdist_b,
                    P_ref, bin_edges,
                    method=self.logdist_reduction)
            elif self.logdist_reduction == 'info_gain':
                tau = self._compute_tau_for_chain(chain_key)
                dists = distances_for_kb_info_gain(
                    qry_seqs, ratio_map,
                    self.logdist_k, self.logdist_b, tau)
            elif self.logdist_reduction == 'mutual_info':
                if chain_key not in self._chain_ref_freqs:
                    self._chain_ref_freqs[chain_key] = \
                        compute_ref_freqs(ref_seqs)
                ref_freqs = self._chain_ref_freqs[chain_key]
                dists = distances_for_kb_mutual_info(
                    qry_seqs, ratio_map,
                    self.logdist_k, self.logdist_b, ref_freqs)
            else:
                dists = distances_for_kb(qry_seqs, ratio_map,
                                         self.logdist_k, self.logdist_b,
                                         self.logdist_K)
            all_chain_dists.append(dists)

        # Weighted average across chains
        w = None if self.chain_weights is None or isinstance(self.chain_weights, str) \
            else self.chain_weights
        return np.average(all_chain_dists, axis=0, weights=w).tolist()

    def _logdist_blend(self, qry_data, ref_data, chain_keys):
        """Blend reduction: alpha * z(topK) + (1-alpha) * z(freqW→topK).

        Computes both topK and freqW→topK per chain using shared pairwise
        ratios, combines each with chain weights, z-normalizes, then blends.

        topK captures local neighborhood distance (how far from training).
        freqW→topK selects neighbors that are both close AND common — exactly
        the training sequences the model learned best.
        """
        all_topk = []
        all_freqw_topk = []

        for chain_key in chain_keys:
            qry_seqs = qry_data[chain_key].tolist()
            ref_seqs = ref_data[chain_key].tolist()

            # Compute and cache pairwise ratios (shared by both reductions)
            if chain_key not in self._chain_ratio_maps:
                _, unique_ref, ratio_map = compute_pairwise_ratios(qry_seqs, ref_seqs)
                self._chain_ratio_maps[chain_key] = ratio_map
                self._chain_unique_refs[chain_key] = unique_ref
            ratio_map = self._chain_ratio_maps[chain_key]

            # topK distances
            topk_dists = distances_for_kb(qry_seqs, ratio_map,
                                          self.logdist_k, self.logdist_b,
                                          self.logdist_K)
            all_topk.append(topk_dists)

            # freqW→topK distances
            if chain_key not in self._chain_ref_probs:
                self._chain_ref_probs[chain_key] = compute_ref_probs(ref_seqs)
            ref_probs = self._chain_ref_probs[chain_key]
            freqw_topk_dists = distances_for_kb_freqw_topk(
                qry_seqs, ratio_map,
                self.logdist_k, self.logdist_b, self.logdist_K,
                ref_probs)
            all_freqw_topk.append(freqw_topk_dists)

        # Weighted combination per reduction
        w = None if self.chain_weights is None or isinstance(self.chain_weights, str) \
            else self.chain_weights
        topk_combined = np.average(all_topk, axis=0, weights=w)
        freqw_topk_combined = np.average(all_freqw_topk, axis=0, weights=w)

        # Z-normalize each combined signal
        topk_std = topk_combined.std()
        freqw_topk_std = freqw_topk_combined.std()
        z_topk = (topk_combined - topk_combined.mean()) / max(topk_std, 1e-10)
        z_freqw_topk = (freqw_topk_combined - freqw_topk_combined.mean()) / \
            max(freqw_topk_std, 1e-10)

        # Blend
        alpha = self.logdist_blend_alpha
        blended = alpha * z_topk + (1.0 - alpha) * z_freqw_topk

        return blended.tolist()

    def _logdist_combine_first(self, qry_data, ref_data):
        """Combine-first multi-chain LogDist.

        Instead of reducing per-chain then combining:
          1. Compute per-chain pairwise distances (same as reduce_first)
          2. Combine chains per training sample: D(q,r) = sum_i w_i * d_i(q,r)
          3. Reduce across training samples: D(q) = mean(top-K smallest D(q,r))

        This preserves cross-chain correlations at the sample level.
        Only supports 'topk' and 'freqw_topk' reductions.
        """
        chain_keys = [f'_chain_{c}' for c in self.chain_cols]

        # Compute and cache pairwise ratios per chain
        ratio_maps_list = []
        unique_refs_list = []
        ref_seqs_list = []
        for chain_key in chain_keys:
            qry_seqs = qry_data[chain_key].tolist()
            ref_seqs = ref_data[chain_key].tolist()
            ref_seqs_list.append(ref_seqs)

            if chain_key not in self._chain_ratio_maps:
                _, unique_ref, ratio_map = compute_pairwise_ratios(
                    qry_seqs, ref_seqs)
                self._chain_ratio_maps[chain_key] = ratio_map
                self._chain_unique_refs[chain_key] = unique_ref

            ratio_maps_list.append(self._chain_ratio_maps[chain_key])
            unique_refs_list.append(self._chain_unique_refs[chain_key])

        # Get chain weights
        w = self.chain_weights
        if w is None or isinstance(w, str):
            w = np.ones(len(chain_keys)) / len(chain_keys)
        w = np.asarray(w, dtype=float)

        # Get query sequences per chain
        chain_seqs_list_qry = [qry_data[ck].tolist() for ck in chain_keys]

        # Determine reduction
        reduction = self.logdist_reduction
        if reduction not in ('topk', 'freqw_topk'):
            print(f"  WARNING: combine_first only supports 'topk' and "
                  f"'freqw_topk', got '{reduction}'. Falling back to 'topk'.")
            reduction = 'topk'

        # Compute sample probs if needed
        sample_probs = None
        if reduction == 'freqw_topk':
            sample_probs = compute_sample_probs(ref_seqs_list)

        print(f"  Computing combine-first LogDist "
              f"(reduction={reduction}, K={self.logdist_K})...")
        result = distances_combine_first(
            chain_seqs_list_qry, ratio_maps_list, unique_refs_list,
            ref_seqs_list, w, self.logdist_k, self.logdist_b,
            self.logdist_K, reduction=reduction,
            sample_probs=sample_probs)

        return result.tolist()


# ===========================================================================
# Module-level logdist functions (usable independently of General_Evaluator)
# ===========================================================================

def compute_pairwise_ratios(qry_seqs, ref_seqs):
    """Compute pairwise Levenshtein.ratio matrix between unique query and ref sequences.

    Returns:
        (unique_qry, unique_ref, ratio_map) where ratio_map[seq] is a 1-D
        numpy array of ratios against all unique ref sequences.
    """
    unique_qry = list(dict.fromkeys(qry_seqs))
    unique_ref = list(dict.fromkeys(ref_seqs))

    print(f"  Computing pairwise Levenshtein.ratio matrix: "
          f"{len(unique_qry)} x {len(unique_ref)} = "
          f"{len(unique_qry) * len(unique_ref):,}")

    ratio_map = {}
    t0 = time.time()
    for i, qs in enumerate(unique_qry):
        if (i + 1) % 200 == 0 or i == 0 or i == len(unique_qry) - 1:
            print(f"    [{i+1:>5}/{len(unique_qry)}] "
                  f"{100*(i+1)/len(unique_qry):.1f}%")
        row = np.array([Levenshtein.ratio(qs, rs) for rs in unique_ref])
        ratio_map[qs] = row

    print(f"  Done in {time.time()-t0:.1f}s")
    return unique_qry, unique_ref, ratio_map


def logdist_from_ratios(ratio_row, k, b, K=None, use_log=True):
    """Compute log-transformed distance from a precomputed ratio row.

    Per-pair:   d(q, r) = log(k * (1 - ratio + b))   [use_log=True]
                d(q, r) = k * (1 - ratio + b)         [use_log=False]
    Reduced:    D(q) = mean of top-K smallest d values
                If K is None or >= len(ratio_row), mean over all references.
    """
    edit_dists = 1.0 - ratio_row
    args_inside = k * (edit_dists + b)
    args_inside = np.clip(args_inside, 1e-15, None)
    per_pair = np.log(args_inside) if use_log else args_inside
    if K is not None and K < len(per_pair):
        topk = np.sort(per_pair)[:K]
        return float(np.mean(topk))
    return float(np.mean(per_pair))


def distances_for_kb(qry_seqs, ratio_map, k, b, K=None):
    """Compute log-transformed distances for all query sequences."""
    return np.array([logdist_from_ratios(ratio_map[qs], k, b, K) for qs in qry_seqs])


def compute_ref_probs(ref_seqs):
    """Compute probability P(s) = freq(s)/N for each unique reference sequence.

    Returns:
        numpy array of probabilities aligned with the unique_ref order
        (same order as used in compute_pairwise_ratios).
    """
    from collections import Counter
    unique_ref = list(dict.fromkeys(ref_seqs))  # preserves order
    freq = Counter(ref_seqs)
    n_total = len(ref_seqs)
    probs = np.array([freq[s] / n_total for s in unique_ref])
    return probs


def logdist_from_ratios_prob(ratio_row, k, b, ref_probs):
    """Probability-weighted LogDist (entropy format).

    D(q) = sum_s P(s) * log(k * (1 - ratio(q,s) + b))

    This is a proper expected self-information — each reference
    contributes proportionally to its training frequency.
    No top-K cutoff needed.
    """
    edit_dists = 1.0 - ratio_row
    args_inside = k * (edit_dists + b)
    args_inside = np.clip(args_inside, 1e-15, None)
    per_pair = np.log(args_inside)
    return float(np.sum(ref_probs * per_pair))


def distances_for_kb_prob(qry_seqs, ratio_map, k, b, ref_probs):
    """Compute probability-weighted LogDist for all query sequences."""
    return np.array([logdist_from_ratios_prob(ratio_map[qs], k, b, ref_probs)
                     for qs in qry_seqs])


def logdist_freqw_then_topk(ratio_row, k, b, K, ref_probs):
    """Frequency-weighted distance selection with top-K.

    score(q, r) = d(q, r) * P(r): selects neighbors that are both closest
    AND most frequent in training.  Most negative score = best match.

    Returns mean of top-K scores (most negative).
    """
    edit_dists = 1.0 - ratio_row
    args_inside = k * (edit_dists + b)
    args_inside = np.clip(args_inside, 1e-15, None)
    dists = np.log(args_inside)
    scores = dists * ref_probs  # d<0, P>0 → score<0; more negative = better
    if K is not None and K < len(scores):
        topk_idx = np.argpartition(scores, K)[:K]
    else:
        topk_idx = np.arange(len(scores))
    return float(scores[topk_idx].mean())


def distances_for_kb_freqw_topk(qry_seqs, ratio_map, k, b, K, ref_probs):
    """Batch freqW→topK computation for all query sequences."""
    return np.array([logdist_freqw_then_topk(ratio_map[qs], k, b, K, ref_probs)
                     for qs in qry_seqs])


# ===========================================================================
# Divergence-based reductions
# ===========================================================================

def compute_ref_distance_profile(ratio_map, sub_seqs, k, b,
                                  n_hist_bins=50, epsilon=1e-10):
    """Precompute average within-training LogDist histogram profile.

    For each subsampled training sequence, compute per-pair log-distances to
    all training refs, histogram them with shared bin edges, and average.

    Args:
        ratio_map: dict mapping subsampled seq -> 1-D array of Levenshtein
            ratios against all training refs (from compute_pairwise_ratios).
        sub_seqs: list of subsampled training sequences (keys into ratio_map).
        k, b: LogDist transform parameters.
        n_hist_bins: number of histogram bins for distance profiles.
        epsilon: smoothing constant to avoid log(0) in KL/JSD.

    Returns:
        (P_ref, bin_edges): P_ref is normalized histogram (length n_hist_bins),
            bin_edges has length n_hist_bins+1.
    """
    # Compute all per-pair log-distances to determine shared bin edges
    all_dists = []
    for s in sub_seqs:
        row = ratio_map[s]
        edit_dists = 1.0 - row
        args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
        per_pair = np.log(args_inside)
        all_dists.append(per_pair)

    pooled = np.concatenate(all_dists)
    _, bin_edges = np.histogram(pooled, bins=n_hist_bins)

    # Histogram each subsample's distances with shared edges, then average
    histograms = np.zeros((len(sub_seqs), n_hist_bins))
    for i, s in enumerate(sub_seqs):
        row = ratio_map[s]
        edit_dists = 1.0 - row
        args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
        per_pair = np.log(args_inside)
        counts, _ = np.histogram(per_pair, bins=bin_edges)
        total = counts.sum()
        if total > 0:
            histograms[i] = counts / total
        else:
            histograms[i] = 1.0 / n_hist_bins

    P_ref = histograms.mean(axis=0)
    P_ref += epsilon
    P_ref /= P_ref.sum()

    return P_ref, bin_edges


def logdist_divergence_jsd(ratio_row, k, b, ref_profile, bin_edges,
                            epsilon=1e-10):
    """Jensen-Shannon divergence of query distance profile vs reference.

    Returns float in [0, ln2].
    """
    edit_dists = 1.0 - ratio_row
    args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
    per_pair = np.log(args_inside)

    P_q, _ = np.histogram(per_pair, bins=bin_edges)
    total = P_q.sum()
    if total > 0:
        P_q = P_q.astype(float) / total
    else:
        P_q = np.ones(len(ref_profile)) / len(ref_profile)
    P_q += epsilon
    P_q /= P_q.sum()

    M = 0.5 * (P_q + ref_profile)
    jsd = 0.5 * np.sum(P_q * np.log(P_q / M)) + \
          0.5 * np.sum(ref_profile * np.log(ref_profile / M))
    return float(jsd)


def logdist_divergence_kl(ratio_row, k, b, ref_profile, bin_edges,
                           epsilon=1e-10):
    """KL divergence: KL(P_q || P_ref). Returns float >= 0."""
    edit_dists = 1.0 - ratio_row
    args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
    per_pair = np.log(args_inside)

    P_q, _ = np.histogram(per_pair, bins=bin_edges)
    total = P_q.sum()
    if total > 0:
        P_q = P_q.astype(float) / total
    else:
        P_q = np.ones(len(ref_profile)) / len(ref_profile)
    P_q += epsilon
    P_q /= P_q.sum()

    kl = float(np.sum(P_q * np.log(P_q / ref_profile)))
    return kl


def logdist_divergence_wasserstein(ratio_row, k, b, ref_profile, bin_edges):
    """Wasserstein-1 distance via CDF comparison. Returns float >= 0."""
    edit_dists = 1.0 - ratio_row
    args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
    per_pair = np.log(args_inside)

    P_q, _ = np.histogram(per_pair, bins=bin_edges)
    total = P_q.sum()
    if total > 0:
        P_q = P_q.astype(float) / total
    else:
        P_q = np.ones(len(ref_profile)) / len(ref_profile)

    CDF_q = np.cumsum(P_q)
    CDF_ref = np.cumsum(ref_profile)
    bin_widths = np.diff(bin_edges)

    return float(np.sum(np.abs(CDF_q - CDF_ref) * bin_widths))


def distances_for_kb_divergence(qry_seqs, ratio_map, k, b, ref_profile,
                                 bin_edges, method='jsd', epsilon=1e-10):
    """Batch divergence computation for all query sequences.

    Args:
        qry_seqs: list of query sequences (keys into ratio_map).
        ratio_map: dict mapping seq -> ratio array.
        k, b: LogDist transform parameters.
        ref_profile: reference histogram (from compute_ref_distance_profile).
        bin_edges: shared histogram bin edges.
        method: 'jsd', 'kl', or 'wasserstein'.
        epsilon: smoothing constant for JSD/KL.

    Returns:
        1-D numpy array of divergence values.
    """
    dispatch = {
        'jsd': logdist_divergence_jsd,
        'kl': logdist_divergence_kl,
        'wasserstein': logdist_divergence_wasserstein,
    }
    func = dispatch[method]

    if method == 'wasserstein':
        return np.array([func(ratio_map[qs], k, b, ref_profile, bin_edges)
                         for qs in qry_seqs])
    else:
        return np.array([func(ratio_map[qs], k, b, ref_profile, bin_edges,
                               epsilon)
                         for qs in qry_seqs])


# ===========================================================================
# Information-theoretic reductions (info_gain, mutual_info)
# ===========================================================================

def compute_ref_freqs(ref_seqs):
    """Compute frequency count for each unique reference sequence.

    Returns:
        numpy array of integer counts, aligned with unique_ref order
        (same order as used in compute_pairwise_ratios).
    """
    from collections import Counter
    unique_ref = list(dict.fromkeys(ref_seqs))
    freq = Counter(ref_seqs)
    return np.array([freq[s] for s in unique_ref], dtype=float)


def compute_ref_probs_from_freqs(ref_freqs):
    """Convert raw frequency counts to probabilities.

    Args:
        ref_freqs: numpy array of frequency counts (from compute_ref_freqs).

    Returns:
        numpy array of probabilities (same length as ref_freqs).
    """
    total = ref_freqs.sum()
    return ref_freqs / total if total > 0 else np.ones_like(ref_freqs) / len(ref_freqs)


def compute_tau_from_training(ratio_map, sub_seqs, k, b):
    """Compute softmax temperature from within-training distance distribution.

    Uses the standard deviation of ALL per-pair log-distances pooled across
    the training subsample.  This gives a natural scale for the softmax.

    Returns:
        float tau (std of per-pair distances).
    """
    all_dists = []
    for s in sub_seqs:
        row = ratio_map[s]
        edit_dists = 1.0 - row
        args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
        per_pair = np.log(args_inside)
        all_dists.append(per_pair)
    pooled = np.concatenate(all_dists)
    return float(np.std(pooled))


def logdist_info_gain(ratio_row, k, b, tau):
    """Posterior entropy of softmax distribution over training references.

    P(r|q) = softmax(-d(q, r) / tau)
    Returns H(P(r|q)) = -sum P(r|q) * log P(r|q)

    Higher entropy means the query doesn't concentrate on any specific
    training sequence (unseen-like); lower entropy means it identifies
    a specific match (seen-like).

    Range: [0, log(N_refs)].
    """
    edit_dists = 1.0 - ratio_row
    args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
    per_pair = np.log(args_inside)

    # softmax(-d/tau): since d is negative, -d is positive and
    # larger for closer matches → higher probability
    logits = -per_pair / tau
    logits -= logits.max()  # numerical stability
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum()

    H = -np.sum(probs * np.log(probs + 1e-15))
    return float(H)


def logdist_mutual_info(ratio_row, k, b, ref_freqs,
                         n_dist_bins=10, n_freq_bins=5):
    """Mutual information between per-pair log-distances and ref frequencies.

    Higher MI means closeness-to-query correlates with training frequency
    (seen-like: exact matches tend to be common epitopes).

    Returns -MI (negated so higher value = further from training),
    consistent with the distance convention.

    Range: [-MI_max, 0].
    """
    edit_dists = 1.0 - ratio_row
    args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
    per_pair = np.log(args_inside)

    n = len(per_pair)

    # Bin distances using percentile edges
    d_pcts = np.linspace(0, 100, n_dist_bins + 1)
    d_edges = np.percentile(per_pair, d_pcts)
    d_edges[-1] += 1e-10
    d_idx = np.clip(np.digitize(per_pair, d_edges[1:-1]), 0, n_dist_bins - 1)

    # Bin frequencies using percentile edges
    f_pcts = np.linspace(0, 100, n_freq_bins + 1)
    f_edges = np.percentile(ref_freqs, f_pcts)
    f_edges[-1] += 1e-10
    f_idx = np.clip(np.digitize(ref_freqs, f_edges[1:-1]), 0, n_freq_bins - 1)

    # Joint distribution
    joint = np.zeros((n_dist_bins, n_freq_bins))
    for i in range(n):
        joint[d_idx[i], f_idx[i]] += 1.0
    joint /= n

    # Marginals
    p_d = joint.sum(axis=1)
    p_f = joint.sum(axis=0)

    # MI = sum P(d,f) * log(P(d,f) / (P(d)*P(f)))
    mi = 0.0
    for i in range(n_dist_bins):
        for j in range(n_freq_bins):
            if joint[i, j] > 0 and p_d[i] > 0 and p_f[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (p_d[i] * p_f[j]))
    return float(-mi)


def logdist_mutual_info_selfinfo(ratio_row, k, b, ref_selfinfo,
                                  n_dist_bins=10, n_si_bins=5):
    """MI between per-pair log-distances and reference self-information.

    Like logdist_mutual_info but uses I(r) = -log(P(r)) instead of raw
    frequency.  Self-information is the proper information-theoretic quantity;
    it measures "surprise" rather than count.

    Returns -MI (negated: higher = further from training).
    """
    edit_dists = 1.0 - ratio_row
    args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
    per_pair = np.log(args_inside)

    n = len(per_pair)

    d_pcts = np.linspace(0, 100, n_dist_bins + 1)
    d_edges = np.percentile(per_pair, d_pcts)
    d_edges[-1] += 1e-10
    d_idx = np.clip(np.digitize(per_pair, d_edges[1:-1]), 0, n_dist_bins - 1)

    si_pcts = np.linspace(0, 100, n_si_bins + 1)
    si_edges = np.percentile(ref_selfinfo, si_pcts)
    si_edges[-1] += 1e-10
    si_idx = np.clip(np.digitize(ref_selfinfo, si_edges[1:-1]), 0, n_si_bins - 1)

    joint = np.zeros((n_dist_bins, n_si_bins))
    for i in range(n):
        joint[d_idx[i], si_idx[i]] += 1.0
    joint /= n

    p_d = joint.sum(axis=1)
    p_si = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_dist_bins):
        for j in range(n_si_bins):
            if joint[i, j] > 0 and p_d[i] > 0 and p_si[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (p_d[i] * p_si[j]))
    return float(-mi)


def compute_ref_selfinfo(ref_seqs):
    """Compute self-information I(r) = -log(P(r)) for each unique ref.

    Returns array aligned with unique_ref order.
    """
    from collections import Counter
    unique_ref = list(dict.fromkeys(ref_seqs))
    freq = Counter(ref_seqs)
    n_total = len(ref_seqs)
    return np.array([-np.log(freq[s] / n_total) for s in unique_ref])


def distances_for_kb_mutual_info_selfinfo(qry_seqs, ratio_map, k, b,
                                           ref_selfinfo,
                                           n_dist_bins=10, n_si_bins=5):
    """Batch MI(distance, self-information) for all query sequences."""
    return np.array([logdist_mutual_info_selfinfo(
        ratio_map[qs], k, b, ref_selfinfo, n_dist_bins, n_si_bins)
        for qs in qry_seqs])


def distances_for_kb_info_gain(qry_seqs, ratio_map, k, b, tau):
    """Batch info_gain (posterior entropy) for all query sequences."""
    return np.array([logdist_info_gain(ratio_map[qs], k, b, tau)
                     for qs in qry_seqs])


def distances_for_kb_mutual_info(qry_seqs, ratio_map, k, b, ref_freqs,
                                  n_dist_bins=10, n_freq_bins=5):
    """Batch mutual_info for all query sequences."""
    return np.array([logdist_mutual_info(ratio_map[qs], k, b, ref_freqs,
                                          n_dist_bins, n_freq_bins)
                     for qs in qry_seqs])


def compute_multichain_distances(chain_seqs_list, ratio_maps, k, b, K,
                                  weights=None, combine_order='reduce_first',
                                  unique_refs_list=None, ref_seqs_list=None,
                                  reduction='topk'):
    """Compute per-sample multi-chain weighted LogDist (generalized N-chain).

    Parameters:
        chain_seqs_list: list of lists/arrays, each containing sequences for one chain
            e.g., [peptide_seqs, cdr3a_seqs, cdr3b_seqs] for TCR
            or [heavy_seqs, light_seqs] for BCR
        ratio_maps: list of dicts, one per chain (from compute_pairwise_ratios)
        k, b, K: LogDist parameters
        weights: optional array of per-chain weights (must sum to 1).
            If None, uses uniform 1/n_chains weighting.
        combine_order: 'reduce_first' (default) or 'combine_first'
        unique_refs_list: required for combine_first — list of unique ref lists per chain
        ref_seqs_list: required for combine_first — list of raw ref seq lists per chain
        reduction: 'topk' or 'freqw_topk' (only used with combine_first)

    Returns:
        1-D numpy array of per-sample distances (weighted mean across chains)
    """
    n_chains = len(chain_seqs_list)
    n_samples = len(chain_seqs_list[0])

    if weights is None:
        weights = np.ones(n_chains) / n_chains
    else:
        weights = np.asarray(weights, dtype=float)

    if combine_order == 'combine_first':
        if unique_refs_list is None or ref_seqs_list is None:
            raise ValueError(
                "combine_first requires unique_refs_list and ref_seqs_list")
        sample_probs = None
        if reduction == 'freqw_topk':
            sample_probs = compute_sample_probs(ref_seqs_list)
        return distances_combine_first(
            chain_seqs_list, ratio_maps, unique_refs_list, ref_seqs_list,
            weights, k, b, K, reduction=reduction,
            sample_probs=sample_probs)

    # reduce_first (original behavior)
    dists = np.zeros(n_samples)
    for chain_idx in range(n_chains):
        seqs = chain_seqs_list[chain_idx]
        rmap = ratio_maps[chain_idx]
        w = weights[chain_idx]
        for i in range(n_samples):
            dists[i] += w * logdist_from_ratios(rmap[seqs[i]], k, b, K)

    return dists


def precompute_chain_distances_K(ratio_map, k_values, b_values, K_values):
    """Precompute logdist for all (K, k, b) combos and all unique sequences.

    Returns dict: (K, k, b) -> {seq: distance_value}
    """
    unique_seqs = list(ratio_map.keys())
    result = {}
    t0 = time.time()
    combos = [(Kv, kv, bv) for Kv in K_values for kv in k_values for bv in b_values]

    for idx, (Kv, kv, bv) in enumerate(combos):
        dist_map = {}
        for seq in unique_seqs:
            dist_map[seq] = logdist_from_ratios(ratio_map[seq], kv, bv, K=Kv)
        result[(Kv, kv, bv)] = dist_map

        if (idx + 1) % 50 == 0 or idx == 0 or idx == len(combos) - 1:
            print(f"      [{idx+1}/{len(combos)}] "
                  f"{100*(idx+1)/len(combos):.0f}%  "
                  f"({time.time()-t0:.1f}s)")

    return result


def avg_dist_from_ratios(ratio_row):
    """Average Levenshtein edit distance: mean(1 - ratio)."""
    return float(np.mean(1.0 - ratio_row))


def min_dist_from_ratios(ratio_row):
    """Min Levenshtein edit distance: min(1 - ratio)."""
    return float(np.min(1.0 - ratio_row))


# ===========================================================================
# Combine-first LogDist functions
# ===========================================================================

def build_sample_to_unique_index(ref_seqs_list, unique_refs_list):
    """Map training sample index -> unique_ref index for each chain.

    Parameters:
        ref_seqs_list: list of lists, each containing raw ref sequences for one chain
            (length N_train each, may contain duplicates)
        unique_refs_list: list of lists, each containing unique ref sequences for one chain
            (from compute_pairwise_ratios)

    Returns:
        list of int arrays, each shape (N_train,), mapping sample -> unique index
    """
    indices_per_chain = []
    for ref_seqs, unique_refs in zip(ref_seqs_list, unique_refs_list):
        seq_to_idx = {s: i for i, s in enumerate(unique_refs)}
        idx_arr = np.array([seq_to_idx[s] for s in ref_seqs], dtype=np.intp)
        indices_per_chain.append(idx_arr)
    return indices_per_chain


def compute_combined_pairwise_logdist(ratio_rows_per_chain, sample_indices,
                                       weights, k, b):
    """Compute combined per-sample LogDist for one query against all training samples.

    For each training sample r:
        D(q, r) = sum_i w_i * log(k * (1 - ratio_i(q, unique_ref[idx_i[r]]) + b))

    Parameters:
        ratio_rows_per_chain: list of 1-D arrays, each containing ratios for one chain
            (shape N_unique_refs_i each, from ratio_map[qry_seq])
        sample_indices: list of int arrays, each shape (N_train,),
            mapping training sample -> unique ref index per chain
        weights: 1-D array of chain weights (length N_chains, sums to 1)
        k, b: LogDist transform parameters

    Returns:
        1-D array shape (N_train,) of combined distances
    """
    n_train = len(sample_indices[0])
    combined = np.zeros(n_train)
    for ch_idx, (ratio_row, s_idx, w) in enumerate(
            zip(ratio_rows_per_chain, sample_indices, weights)):
        # Gather ratios for this chain via fancy indexing
        ratios_for_samples = ratio_row[s_idx]  # shape (N_train,)
        edit_dists = 1.0 - ratios_for_samples
        args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
        per_pair = np.log(args_inside)
        combined += w * per_pair
    return combined


def logdist_combine_first_topk(ratio_rows_per_chain, sample_indices,
                                weights, k, b, K):
    """Combine-first LogDist with top-K reduction.

    1. Compute combined distance D(q, r) for all training samples
    2. Select top-K smallest combined distances
    3. Return mean of top-K

    Returns:
        float: combine-first top-K distance for one query
    """
    combined = compute_combined_pairwise_logdist(
        ratio_rows_per_chain, sample_indices, weights, k, b)
    if K is not None and K < len(combined):
        topk = np.partition(combined, K)[:K]
        return float(np.mean(topk))
    return float(np.mean(combined))


def logdist_combine_first_freqw_topk(ratio_rows_per_chain, sample_indices,
                                      weights, k, b, K, sample_probs):
    """Combine-first LogDist with frequency-weighted top-K reduction.

    1. Compute combined distance D(q, r) for all training samples
    2. Score each sample: score(q, r) = D(q, r) * P(r)
    3. Select top-K most negative scores
    4. Return mean of top-K scores

    Parameters:
        sample_probs: 1-D array of P(sample) for each training sample (length N_train)

    Returns:
        float: combine-first freqw_topk distance for one query
    """
    combined = compute_combined_pairwise_logdist(
        ratio_rows_per_chain, sample_indices, weights, k, b)
    scores = combined * sample_probs  # D<0, P>0 -> score<0; more negative = better
    if K is not None and K < len(scores):
        topk_idx = np.argpartition(scores, K)[:K]
    else:
        topk_idx = np.arange(len(scores))
    return float(scores[topk_idx].mean())


def compute_sample_probs(ref_seqs_list):
    """Compute P(sample) based on full chain tuple frequency.

    P(sample) = freq(tuple of all chains) / N_train

    Parameters:
        ref_seqs_list: list of lists, each containing ref sequences for one chain

    Returns:
        1-D array of probabilities, shape (N_train,)
    """
    from collections import Counter
    n_train = len(ref_seqs_list[0])
    # Build tuples of all chains for each training sample
    tuples = [tuple(ref_seqs_list[ch][i] for ch in range(len(ref_seqs_list)))
              for i in range(n_train)]
    freq = Counter(tuples)
    probs = np.array([freq[t] / n_train for t in tuples])
    return probs


def distances_combine_first(chain_seqs_list_qry, ratio_maps_list,
                             unique_refs_list, ref_seqs_list,
                             weights, k, b, K, reduction='topk',
                             sample_probs=None):
    """Batch combine-first LogDist for all query samples.

    Parameters:
        chain_seqs_list_qry: list of lists, query sequences per chain
            e.g., [peptide_qry, cdr3a_qry, cdr3b_qry]
        ratio_maps_list: list of dicts, ratio_map per chain
            (from compute_pairwise_ratios)
        unique_refs_list: list of lists, unique ref sequences per chain
        ref_seqs_list: list of lists, raw ref sequences per chain (with duplicates)
        weights: 1-D array of chain weights
        k, b, K: LogDist parameters
        reduction: 'topk' or 'freqw_topk'
        sample_probs: 1-D array of P(sample) for freqw_topk (from compute_sample_probs)

    Returns:
        1-D numpy array of per-query combined distances
    """
    n_chains = len(chain_seqs_list_qry)
    n_qry = len(chain_seqs_list_qry[0])
    weights = np.asarray(weights, dtype=float)

    # Build sample -> unique ref index mapping (once)
    sample_indices = build_sample_to_unique_index(ref_seqs_list, unique_refs_list)

    # Precompute sample_probs if needed
    if reduction == 'freqw_topk' and sample_probs is None:
        sample_probs = compute_sample_probs(ref_seqs_list)

    # Cache: query combo tuple -> distance (avoid recomputing for duplicate queries)
    combo_cache = {}
    result = np.zeros(n_qry)

    for i in range(n_qry):
        # Build query combo key
        combo_key = tuple(chain_seqs_list_qry[ch][i] for ch in range(n_chains))
        if combo_key in combo_cache:
            result[i] = combo_cache[combo_key]
            continue

        # Gather ratio rows for this query
        ratio_rows = [ratio_maps_list[ch][combo_key[ch]]
                      for ch in range(n_chains)]

        if reduction == 'freqw_topk':
            d = logdist_combine_first_freqw_topk(
                ratio_rows, sample_indices, weights, k, b, K, sample_probs)
        else:
            d = logdist_combine_first_topk(
                ratio_rows, sample_indices, weights, k, b, K)

        combo_cache[combo_key] = d
        result[i] = d

        if (i + 1) % 500 == 0 or i == 0 or i == n_qry - 1:
            print(f"    combine-first [{i+1:>5}/{n_qry}] "
                  f"{100*(i+1)/n_qry:.1f}%")

    return result


# ===========================================================================
# Combine-first: per-chain raw distance helpers
# ===========================================================================

def compute_per_chain_raw_logdist(ratio_rows_per_chain, sample_indices, k, b,
                                   use_log=True):
    """Compute per-chain raw log-distances for one query against all training samples.

    Extracts the per-chain distance computation so it can be computed once and
    reused across all combine methods.

    Parameters:
        ratio_rows_per_chain: list of 1-D arrays, ratios for each chain
        sample_indices: list of int arrays, mapping training sample -> unique ref index
        k, b: LogDist transform parameters
        use_log: if True, apply log transform; if False, use raw k*(1-ratio+b)

    Returns:
        list of 1-D arrays (one per chain), each shape (N_train,)
    """
    per_chain = []
    for ratio_row, s_idx in zip(ratio_rows_per_chain, sample_indices):
        ratios_for_samples = ratio_row[s_idx]
        edit_dists = 1.0 - ratios_for_samples
        args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
        per_chain.append(np.log(args_inside) if use_log else args_inside)
    return per_chain


def compute_pairwise_chain_stats(ratio_maps_list, unique_refs_list,
                                  ref_seqs_list, k, b, subsample=500,
                                  seed=42, use_log=True):
    """Pre-compute mean/std of pairwise log-distances from training subsample.

    Used by znorm_sum and max_znorm to put chains on comparable scales.

    Parameters:
        ratio_maps_list: list of ratio_map dicts (one per chain)
        unique_refs_list: list of unique ref sequence lists (one per chain)
        ref_seqs_list: list of raw ref sequence lists (one per chain)
        k, b: LogDist transform parameters
        subsample: number of training samples to subsample
        seed: random seed
        use_log: if True, apply log transform; if False, use raw k*(1-ratio+b)

    Returns:
        list of (mean, std) tuples, one per chain
    """
    n_chains = len(ratio_maps_list)
    n_train = len(ref_seqs_list[0])
    rng = np.random.RandomState(seed)
    n_sub = min(subsample, n_train)
    sub_idx = rng.choice(n_train, size=n_sub, replace=False)

    sample_indices = build_sample_to_unique_index(ref_seqs_list, unique_refs_list)

    chain_stats = []
    for ch in range(n_chains):
        all_dists = []
        for i in sub_idx:
            seq = ref_seqs_list[ch][i]
            if seq not in ratio_maps_list[ch]:
                continue
            ratio_row = ratio_maps_list[ch][seq]
            ratios_for_samples = ratio_row[sample_indices[ch]]
            edit_dists = 1.0 - ratios_for_samples
            args_inside = np.clip(k * (edit_dists + b), 1e-15, None)
            per_pair = np.log(args_inside) if use_log else args_inside
            all_dists.append(per_pair)

        if all_dists:
            all_flat = np.concatenate(all_dists)
            mu = float(np.mean(all_flat))
            sigma = float(np.std(all_flat))
            if sigma < 1e-10:
                sigma = 1e-10
        else:
            mu, sigma = 0.0, 1.0
        chain_stats.append((mu, sigma))

    return chain_stats


# ===========================================================================
# Combine-first: combination functions
# ===========================================================================

def combine_weighted_sum(per_chain_dists, weights, chain_stats=None):
    """Weighted sum: D(q,r) = sum(w_i * d_i(q,r)).

    This is the existing baseline combination. chain_stats is unused.
    """
    n_train = len(per_chain_dists[0])
    combined = np.zeros(n_train)
    for d_ch, w in zip(per_chain_dists, weights):
        combined += w * d_ch
    return combined


def combine_znorm_sum(per_chain_dists, weights, chain_stats):
    """Z-normalized weighted sum: D(q,r) = sum(w_i * (d_i - mu_i) / sigma_i).

    Equalizes scale across chains before combining.
    """
    n_train = len(per_chain_dists[0])
    combined = np.zeros(n_train)
    for d_ch, w, (mu, sigma) in zip(per_chain_dists, weights, chain_stats):
        z = (d_ch - mu) / sigma
        combined += w * z
    return combined


def combine_rank_sum(per_chain_dists, weights, chain_stats=None):
    """Rank-based weighted sum: D(q,r) = sum(w_i * rank_i(r) / N).

    Nonparametric, robust to scale and distribution differences.
    """
    n_train = len(per_chain_dists[0])
    combined = np.zeros(n_train)
    for d_ch, w in zip(per_chain_dists, weights):
        ranks = np.argsort(np.argsort(d_ch)).astype(float) / n_train
        combined += w * ranks
    return combined


def combine_geometric(per_chain_dists, weights, chain_stats=None):
    """Weighted geometric mean: D(q,r) = -prod(|d_i|^w_i).

    Multiplicative — all chains must be close for the result to be close.
    Negated so that more negative = closer (consistent with other methods).
    """
    n_train = len(per_chain_dists[0])
    log_combined = np.zeros(n_train)
    for d_ch, w in zip(per_chain_dists, weights):
        abs_d = np.maximum(np.abs(d_ch), 1e-10)
        log_combined += w * np.log(abs_d)
    return -np.exp(log_combined)


def combine_max_znorm(per_chain_dists, weights, chain_stats):
    """Chebyshev (max z-score): D(q,r) = max_i((d_i - mu_i) / sigma_i).

    Worst chain dominates — if ANY chain is far, combined distance is high.
    Weights are ignored (pure max).
    """
    n_train = len(per_chain_dists[0])
    z_scores = []
    for d_ch, (mu, sigma) in zip(per_chain_dists, chain_stats):
        z = (d_ch - mu) / sigma
        z_scores.append(z)
    return np.max(z_scores, axis=0)


def combine_weighted_max_znorm(per_chain_dists, weights, chain_stats):
    """Weighted Chebyshev: select chain by max(w_i * z_i), return z of winner.

    Weights affect only chain SELECTION (which chain's z-score wins),
    not the output magnitude. This avoids biasing distance values by weight scale.

    Rescales weights from sum=1 to sum=n_chains (mean=1) so that:
    - uniform weights → w_scaled=[1,1,1] → recovers unweighted max_znorm
    - sigma_H weights → CDR3 more likely selected, but output is pure z-score
    """
    n_chains = len(per_chain_dists)
    w_scaled = np.array(weights) * n_chains
    z_arr = []
    wz_arr = []
    for d_ch, ws, (mu, sigma) in zip(per_chain_dists, w_scaled, chain_stats):
        z = (d_ch - mu) / sigma
        z_arr.append(z)
        wz_arr.append(ws * z)
    z_arr = np.array(z_arr)
    wz_arr = np.array(wz_arr)
    # Select chain with largest w*z, return its unscaled z-score
    winner = np.argmax(wz_arr, axis=0)
    return z_arr[winner, np.arange(z_arr.shape[1])]


def combine_weighted_min_znorm(per_chain_dists, weights, chain_stats):
    """Weighted min z-norm: select chain by min(w_i * z_i), return z of winner.

    Like weighted_max_znorm but picks the chain with the SMALLEST weighted
    z-score (most similar to training) as the representative distance.
    """
    n_chains = len(per_chain_dists)
    w_scaled = np.array(weights) * n_chains
    z_arr = []
    wz_arr = []
    for d_ch, ws, (mu, sigma) in zip(per_chain_dists, w_scaled, chain_stats):
        z = (d_ch - mu) / sigma
        z_arr.append(z)
        wz_arr.append(ws * z)
    z_arr = np.array(z_arr)
    wz_arr = np.array(wz_arr)
    # Select chain with smallest w*z, return its unscaled z-score
    winner = np.argmin(wz_arr, axis=0)
    return z_arr[winner, np.arange(z_arr.shape[1])]


def combine_harmonic(per_chain_dists, weights, chain_stats=None):
    """Weighted harmonic mean: D(q,r) = 1 / sum(w_i / d_i).

    Emphasizes the closest (most negative) chain — dominated by small |d|.
    Clamped to avoid division by zero.
    """
    n_train = len(per_chain_dists[0])
    inv_sum = np.zeros(n_train)
    for d_ch, w in zip(per_chain_dists, weights):
        d_clamped = np.minimum(d_ch, -1e-10)
        inv_sum += w / d_clamped
    # Avoid division by zero in the inverse
    inv_sum = np.where(np.abs(inv_sum) < 1e-15, -1e-15, inv_sum)
    return 1.0 / inv_sum


def combine_l1_znorm(per_chain_dists, weights, chain_stats):
    """Weighted L1 z-norm: D = sum(w_i * |z_i|)."""
    combined = np.zeros(len(per_chain_dists[0]))
    for d_ch, w, (mu, sigma) in zip(per_chain_dists, weights, chain_stats):
        combined += w * np.abs((d_ch - mu) / sigma)
    return combined


def combine_l2_znorm(per_chain_dists, weights, chain_stats):
    """Weighted L2 z-norm: D = sqrt(sum(w_i * z_i^2))."""
    combined = np.zeros(len(per_chain_dists[0]))
    for d_ch, w, (mu, sigma) in zip(per_chain_dists, weights, chain_stats):
        z = (d_ch - mu) / sigma
        combined += w * z ** 2
    return np.sqrt(combined)


def combine_l3_znorm(per_chain_dists, weights, chain_stats):
    """Weighted L3 z-norm: D = (sum(w_i * |z_i|^3))^(1/3)."""
    combined = np.zeros(len(per_chain_dists[0]))
    for d_ch, w, (mu, sigma) in zip(per_chain_dists, weights, chain_stats):
        combined += w * np.abs((d_ch - mu) / sigma) ** 3
    return combined ** (1.0 / 3.0)


def combine_naive_avg(per_chain_dists, weights, chain_stats=None):
    """Uniform mean across chains. No z-norm, no weights."""
    return np.mean(per_chain_dists, axis=0)


def combine_naive_min(per_chain_dists, weights, chain_stats=None):
    """Per-sample minimum across chains. No z-norm, no weights."""
    return np.min(per_chain_dists, axis=0)


def make_combine_weighted_max_anchor(k, b):
    """Factory: create anchor-normalized combine function for given k, b.

    Returns a combine function with the standard (per_chain_dists, weights,
    chain_stats) signature so it plugs into the existing pipeline.

    Instead of z-normalizing with per-fold (mu, sigma), maps each chain's
    distance to [0, 1] using the theoretical min/max of LogDist:
        d_min = log(k * b)       (identical sequences, ratio=1)
        d_max = log(k * (1 + b)) (maximally different, ratio=0)
        d_norm = (d - d_min) / (d_max - d_min)

    This normalization is fold-independent (no training statistics needed),
    so distances are directly comparable across different training sets.

    Then selects the chain with max(w_i * d_norm_i) and returns d_norm of winner.
    """
    d_min = np.log(k * b)
    d_max = np.log(k * (1 + b))
    d_range = d_max - d_min

    def combine_fn(per_chain_dists, weights, chain_stats=None):
        n_chains = len(per_chain_dists)
        w_scaled = np.array(weights) * n_chains
        norm_arr = []
        wnorm_arr = []
        for d_ch, ws in zip(per_chain_dists, w_scaled):
            d_norm = (d_ch - d_min) / d_range
            norm_arr.append(d_norm)
            wnorm_arr.append(ws * d_norm)
        norm_arr = np.array(norm_arr)
        wnorm_arr = np.array(wnorm_arr)
        # Select chain with largest weighted anchor-normalized distance
        winner = np.argmax(wnorm_arr, axis=0)
        return norm_arr[winner, np.arange(norm_arr.shape[1])]

    return combine_fn


def make_combine_zscore_select_anchor_output(k, b):
    """Factory: z-score selection, anchor [0,1] output.

    Uses z-scores (with sigma_C weights) to select which chain is most
    anomalous (best ranking), but returns the anchor-normalized [0,1]
    distance of the winner (fold-independent scale for prediction).
    """
    d_min = np.log(k * b)
    d_max = np.log(k * (1 + b))
    d_range = d_max - d_min

    def combine_fn(per_chain_dists, weights, chain_stats):
        n_chains = len(per_chain_dists)
        w_scaled = np.array(weights) * n_chains
        # Z-score selection
        wz_arr = []
        for d_ch, ws, (mu, sigma) in zip(per_chain_dists, w_scaled, chain_stats):
            z = (d_ch - mu) / sigma
            wz_arr.append(ws * z)
        wz_arr = np.array(wz_arr)
        # Anchor-normalized output
        anchor_arr = np.array([(d_ch - d_min) / d_range for d_ch in per_chain_dists])
        winner = np.argmax(wz_arr, axis=0)
        return anchor_arr[winner, np.arange(anchor_arr.shape[1])]

    return combine_fn


def combine_weighted_max_znorm_raw(per_chain_dists, weights, chain_stats):
    """Z-norm for selection, raw distance for output.

    Uses z-scores (with sigma_C weights) to select which chain is most
    anomalous, but returns the RAW per-chain distance of the winner.
    This keeps the output in natural (positive) distance units, avoiding
    the compressed negative z-score range that hurts exponential fitting.
    """
    n_chains = len(per_chain_dists)
    w_scaled = np.array(weights) * n_chains
    raw_arr = np.array(per_chain_dists)   # shape (n_chains, n_samples)
    wz_arr = []
    for d_ch, ws, (mu, sigma) in zip(per_chain_dists, w_scaled, chain_stats):
        z = (d_ch - mu) / sigma
        wz_arr.append(ws * z)
    wz_arr = np.array(wz_arr)
    # Select chain with largest w*z, but return its RAW distance
    winner = np.argmax(wz_arr, axis=0)
    return raw_arr[winner, np.arange(raw_arr.shape[1])]


def make_combine_softmax_znorm_raw(tau=1.0):
    """Factory: softmax of w*z as weights, applied to RAW per-chain distances.

    Instead of hard argmax, uses softmax(tau * w_i * z_i) to produce soft
    weights, then returns weighted sum of raw distances. This preserves
    cross-dataset distance scale while still upweighting anomalous chains.

    tau controls sharpness: tau→∞ = hard max, tau=1 = standard softmax,
    tau→0 = uniform average.
    """
    def combine_fn(per_chain_dists, weights, chain_stats):
        n_chains = len(per_chain_dists)
        w_scaled = np.array(weights) * n_chains
        raw_arr = np.array(per_chain_dists)   # shape (n_chains, n_samples)
        wz_arr = np.zeros_like(raw_arr)
        for i, (d_ch, ws, (mu, sigma)) in enumerate(
                zip(per_chain_dists, w_scaled, chain_stats)):
            z = (d_ch - mu) / sigma
            wz_arr[i] = ws * z
        # Softmax along chain axis (axis=0)
        wz_scaled = tau * wz_arr
        wz_scaled -= wz_scaled.max(axis=0, keepdims=True)  # numerical stability
        exp_wz = np.exp(wz_scaled)
        alpha = exp_wz / exp_wz.sum(axis=0, keepdims=True)  # (n_chains, n_samples)
        return (alpha * raw_arr).sum(axis=0)
    combine_fn.__name__ = f'softmax_znorm_raw_tau{tau}'
    return combine_fn


def make_combine_softmax_znorm(tau=1.0):
    """Factory: softmax of w*z as weights, applied to Z-NORMALIZED distances.

    Same softmax weighting but outputs weighted sum of z-scores instead of
    raw distances. Tests whether soft selection + normalized signal helps.
    """
    def combine_fn(per_chain_dists, weights, chain_stats):
        n_chains = len(per_chain_dists)
        w_scaled = np.array(weights) * n_chains
        z_arr = np.zeros((n_chains, len(per_chain_dists[0])))
        wz_arr = np.zeros_like(z_arr)
        for i, (d_ch, ws, (mu, sigma)) in enumerate(
                zip(per_chain_dists, w_scaled, chain_stats)):
            z = (d_ch - mu) / sigma
            z_arr[i] = z
            wz_arr[i] = ws * z
        # Softmax along chain axis
        wz_scaled = tau * wz_arr
        wz_scaled -= wz_scaled.max(axis=0, keepdims=True)
        exp_wz = np.exp(wz_scaled)
        alpha = exp_wz / exp_wz.sum(axis=0, keepdims=True)
        return (alpha * z_arr).sum(axis=0)
    combine_fn.__name__ = f'softmax_znorm_tau{tau}'
    return combine_fn


# Pre-build softmax variants at common temperatures
_softmax_raw_t05 = make_combine_softmax_znorm_raw(tau=0.5)
_softmax_raw_t1 = make_combine_softmax_znorm_raw(tau=1.0)
_softmax_raw_t2 = make_combine_softmax_znorm_raw(tau=2.0)
_softmax_raw_t5 = make_combine_softmax_znorm_raw(tau=5.0)
_softmax_znorm_t05 = make_combine_softmax_znorm(tau=0.5)
_softmax_znorm_t1 = make_combine_softmax_znorm(tau=1.0)
_softmax_znorm_t2 = make_combine_softmax_znorm(tau=2.0)
_softmax_znorm_t5 = make_combine_softmax_znorm(tau=5.0)


# Map of combine method names to functions
COMBINE_FUNCTIONS = {
    'weighted_sum': combine_weighted_sum,
    'znorm_sum': combine_znorm_sum,
    'rank_sum': combine_rank_sum,
    'geometric': combine_geometric,
    'max_znorm': combine_max_znorm,
    'weighted_max_znorm': combine_weighted_max_znorm,
    'weighted_max_znorm_raw': combine_weighted_max_znorm_raw,
    'weighted_min_znorm': combine_weighted_min_znorm,
    'harmonic': combine_harmonic,
    'l1_znorm': combine_l1_znorm,
    'l2_znorm': combine_l2_znorm,
    'l3_znorm': combine_l3_znorm,
    'naive_avg': combine_naive_avg,
    'naive_min': combine_naive_min,
    'softmax_raw_t05': _softmax_raw_t05,
    'softmax_raw_t1': _softmax_raw_t1,
    'softmax_raw_t2': _softmax_raw_t2,
    'softmax_raw_t5': _softmax_raw_t5,
    'softmax_znorm_t05': _softmax_znorm_t05,
    'softmax_znorm_t1': _softmax_znorm_t1,
    'softmax_znorm_t2': _softmax_znorm_t2,
    'softmax_znorm_t5': _softmax_znorm_t5,
}

# Methods that require chain_stats (z-normalization)
ZNORM_METHODS = {'znorm_sum', 'max_znorm', 'weighted_max_znorm', 'weighted_max_znorm_raw', 'weighted_min_znorm', 'l1_znorm', 'l2_znorm', 'l3_znorm',
                 'softmax_raw_t05', 'softmax_raw_t1', 'softmax_raw_t2', 'softmax_raw_t5',
                 'softmax_znorm_t05', 'softmax_znorm_t1', 'softmax_znorm_t2', 'softmax_znorm_t5'}


def distances_combine_first_multi(chain_seqs_list_qry, ratio_maps_list,
                                    unique_refs_list, ref_seqs_list,
                                    weights, k, b, K, reduction='topk',
                                    sample_probs=None,
                                    combine_method='weighted_sum',
                                    chain_stats=None,
                                    use_log=True):
    """Batch combine-first LogDist with selectable combination function.

    Like distances_combine_first() but supports multiple combine methods.

    Parameters:
        chain_seqs_list_qry: list of lists, query sequences per chain
        ratio_maps_list: list of dicts, ratio_map per chain
        unique_refs_list: list of lists, unique ref sequences per chain
        ref_seqs_list: list of lists, raw ref sequences per chain
        weights: 1-D array of chain weights
        k, b, K: LogDist parameters
        reduction: 'topk', 'freqw_topk', 'avg', or 'min'
        sample_probs: 1-D array of P(sample) for freqw_topk
        combine_method: one of COMBINE_FUNCTIONS keys
        chain_stats: list of (mu, sigma) per chain (required for znorm/max methods)
        use_log: if True, apply log transform; if False, use raw k*(1-ratio+b)

    Returns:
        1-D numpy array of per-query combined distances
    """
    combine_fn = COMBINE_FUNCTIONS[combine_method]
    n_chains = len(chain_seqs_list_qry)
    n_qry = len(chain_seqs_list_qry[0])
    weights = np.asarray(weights, dtype=float)

    sample_indices = build_sample_to_unique_index(ref_seqs_list, unique_refs_list)

    if reduction == 'freqw_topk' and sample_probs is None:
        sample_probs = compute_sample_probs(ref_seqs_list)

    combo_cache = {}
    result = np.zeros(n_qry)

    for i in range(n_qry):
        combo_key = tuple(chain_seqs_list_qry[ch][i] for ch in range(n_chains))
        if combo_key in combo_cache:
            result[i] = combo_cache[combo_key]
            continue

        ratio_rows = [ratio_maps_list[ch][combo_key[ch]]
                      for ch in range(n_chains)]

        # Compute per-chain raw distances once
        per_chain_dists = compute_per_chain_raw_logdist(
            ratio_rows, sample_indices, k, b, use_log=use_log)

        # Apply combination function
        combined = combine_fn(per_chain_dists, weights, chain_stats)

        # Apply reduction
        if reduction == 'freqw_topk':
            scores = combined * sample_probs
            if K is not None and K < len(scores):
                topk_idx = np.argpartition(scores, K)[:K]
            else:
                topk_idx = np.arange(len(scores))
            d = float(scores[topk_idx].mean())
        elif reduction == 'avg':
            d = float(np.mean(combined))
        elif reduction == 'min':
            d = float(np.min(combined))
        else:
            # topk
            if K is not None and K < len(combined):
                topk = np.partition(combined, K)[:K]
                d = float(np.mean(topk))
            else:
                d = float(np.mean(combined))

        combo_cache[combo_key] = d
        result[i] = d

        if (i + 1) % 500 == 0 or i == 0 or i == n_qry - 1:
            print(f"    {combine_method} [{i+1:>5}/{n_qry}] "
                  f"{100*(i+1)/n_qry:.1f}%")

    return result


# ===========================================================================
# Evaluation helper functions
# ===========================================================================

def safe_metric(name, y_true, y_prob):
    """Compute a performance metric safely, returning np.nan on failure.

    Supports: aucroc, ap, acc, f1, prec, recall, mcc, brier, bss, ppv, npv.
    """
    y_bin = (y_prob >= 0.5).astype(int)
    n_pos, n_neg = int(y_true.sum()), len(y_true) - int(y_true.sum())
    try:
        if name == 'aucroc':
            return np.nan if (n_pos == 0 or n_neg == 0) else float(roc_auc_score(y_true, y_prob))
        elif name == 'ap':
            return np.nan if (n_pos == 0 or n_neg == 0) else float(average_precision_score(y_true, y_prob))
        elif name == 'acc':
            return float(accuracy_score(y_true, y_bin))
        elif name == 'f1':
            return float(f1_score(y_true, y_bin, zero_division=0))
        elif name == 'prec' or name == 'ppv':
            return float(precision_score(y_true, y_bin, zero_division=0))
        elif name == 'recall':
            return float(recall_score(y_true, y_bin, zero_division=0))
        elif name == 'npv':
            tn = int(((y_bin == 0) & (y_true == 0)).sum())
            fn = int(((y_bin == 0) & (y_true == 1)).sum())
            return float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan
        elif name == 'mcc':
            return np.nan if (n_pos == 0 or n_neg == 0) else float(matthews_corrcoef(y_true, y_bin))
        elif name == 'brier':
            return float(np.mean((y_prob - y_true) ** 2))
        elif name == 'bss':
            brier = float(np.mean((y_prob - y_true) ** 2))
            prev = float(y_true.mean())
            brier_clim = prev * (1 - prev)
            return float(1 - brier / brier_clim) if brier_clim > 0 else np.nan
    except Exception:
        pass
    return np.nan


def bayesian_calibrate(y_prob, ppv, npv, prevalence,
                       p_pos=0.75, p_neg=0.25,
                       shrinkage=0.0, blend_alpha=1.0, b_clamp=None,
                       adaptive_blend=False, adaptive_scale=5.0,
                       blend_mode=None, auto_b_threshold=0.0,
                       global_ab=False):
    """Bayesian calibration of model predictions via PPV, NPV, and prevalence.

    Analytically derives Platt-scaling parameters (a, b) from PPV and NPV,
    then maps every raw model probability to a calibrated posterior:

        P(Y=1 | PP=p) = sigmoid(logit(pi) + a + b * logit(p))

    Stabilisation knobs (can be combined):

    1. **shrinkage** ∈ [0, 1]: pulls (a, b) toward the identity map (a=0, b=1).
       ``a_eff = (1 - s)*a``,  ``b_eff = (1 - s)*b + s``.
       At s=0 → full Bayesian; s=1 → original model output.

    2. **blend_alpha** ∈ [0, 1]: blends in probability space.
       ``final = alpha * calibrated + (1 - alpha) * original``.
       At alpha=1 → full Bayesian; alpha=0 → original model output.

    3. **b_clamp** (float or tuple): clamps the slope *b*.
       Scalar ``c`` → b ∈ [1/c, c].  Tuple ``(lo, hi)`` → b ∈ [lo, hi].
       Prevents extreme recalibration from noisy PPV/NPV.

    4. **adaptive_blend** (bool): per-sample blending based on PPV+NPV.
       ``alpha_i = sigmoid(scale * (PPV_i + NPV_i - 1))``.
       When PPV+NPV > 1 (model discriminates): alpha → 1 (trust calibration).
       When PPV+NPV ≤ 1 (near chance): alpha → 0 (keep original).
       Overrides ``blend_alpha`` with per-sample values.

    5. **blend_mode** (str or None): advanced blending mode. Overrides
       ``adaptive_blend`` and ``blend_alpha`` when set.
       - ``'min_ppv_npv'``: ``alpha_i = sigmoid(scale * (min(PPV_i, NPV_i) - 0.5))``.
         Only trusts calibration when BOTH PPV and NPV are high.
       - ``'auto'``: hard auto-select based on mean(b):
         * mean(b) > ``auto_b_threshold`` → full Bayesian (no blend).
         * mean(b) ≤ ``auto_b_threshold`` → b_clamp ≥ 0 + min(PPV,NPV) blend.
       - ``'auto_smooth'``: smooth auto-blend. Always uses b_clamp ≥ 0 +
         min(PPV,NPV), but shifts the sigmoid center by mean(b):
         ``alpha_i = sigmoid(scale * (min(PPV_i, NPV_i) - center))``
         where ``center = 1.0 - sigmoid(3 * mean(b))``
         High mean(b) → center → 0 → most samples get high alpha.
         Low mean(b)  → center → 1 → most samples get low alpha.

    6. **global_ab** (bool): use GLOBAL (a, b) instead of per-sample.
       Per-sample (a, b) from per-sample PPV/NPV can have wildly varying
       slopes (some positive, some negative), which destroys rank ordering
       and hurts AUROC. When ``global_ab=True``, a single (a, b) is computed
       from mean(PPV), mean(NPV), mean(prevalence), ensuring the calibration
       is a monotonic function of y_prob → AUROC is preserved when b > 0.
       Per-sample PPV/NPV are still used for blending alpha.

    Parameters
    ----------
    y_prob : array-like, shape (n,)
        Raw model predicted probabilities in (0, 1).
    ppv : float or array-like
        Positive predictive value(s).  Scalar or per-sample array.
    npv : float or array-like
        Negative predictive value(s).
    prevalence : float or array-like
        Class prevalence pi = P(Y=1).
    p_pos : float or array-like, default 0.75
        Representative score of the positive bin (pred >= threshold).
    p_neg : float or array-like, default 0.25
        Representative score of the negative bin (pred < threshold).
    shrinkage : float, default 0.0
        Shrink (a, b) toward identity (a=0, b=1).  0 = no shrinkage.
    blend_alpha : float, default 1.0
        Blend calibrated with original in probability space. 1 = fully
        calibrated, 0 = original.
    b_clamp : float or tuple or None, default None
        Clamp slope b.  Scalar c → [1/c, c]; tuple → [lo, hi].
    adaptive_blend : bool, default False
        Per-sample blending using PPV+NPV as confidence signal.
        Overrides ``blend_alpha``.
    adaptive_scale : float, default 5.0
        Steepness of the sigmoid for adaptive/min_ppv_npv blending.
        Higher = sharper transition.
    blend_mode : str or None, default None
        ``'min_ppv_npv'``, ``'auto'``, ``'auto_smooth'``, or None
        (use legacy parameters).
    auto_b_threshold : float, default 0.0
        Threshold on mean(b) for ``'auto'`` mode. When mean(b) > threshold,
        use full Bayesian; otherwise use conservative min_ppv_npv + b_clamp.
    global_ab : bool, default False
        Compute a single (a, b) from mean(PPV), mean(NPV), mean(prevalence)
        instead of per-sample values.  Prevents rank-ordering destruction
        when per-sample b values have mixed signs.  Per-sample PPV/NPV are
        still used for blending alpha when adaptive blending is enabled.

    Returns
    -------
    calibrated : ndarray, shape (n,)
        Calibrated probabilities in [0, 1].
    params : dict
        {'a': ..., 'b': ..., 'a_eff': ..., 'b_eff': ..., 'alpha': ...,
         'blend_mode_used': ...}.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    ppv = np.asarray(ppv, dtype=float)
    npv = np.asarray(npv, dtype=float)
    prevalence = np.asarray(prevalence, dtype=float)

    eps = 1e-7
    _logit = lambda x: np.log(np.clip(x, eps, 1 - eps) /
                               np.clip(1 - x, eps, 1 - eps))
    _sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    logit_p_pos = _logit(p_pos)
    logit_p_neg = _logit(p_neg)
    denom = logit_p_pos - logit_p_neg

    b = (_logit(ppv) - _logit(1.0 - npv)) / np.where(
        np.abs(denom) > 1e-12, denom, 1e-12)
    a = _logit(ppv) - _logit(prevalence) - b * logit_p_pos

    # ── Global (a, b): single pair from mean PPV/NPV/prev ────────
    # Per-sample ppv/npv are preserved for blending alpha computation.
    if global_ab:
        ppv_mean = float(np.mean(ppv))
        npv_mean = float(np.mean(npv))
        prev_mean = float(np.mean(prevalence))
        b_global = (_logit(ppv_mean) - _logit(1.0 - npv_mean)) / (
            denom if np.abs(denom) > 1e-12 else 1e-12)
        a_global = _logit(ppv_mean) - _logit(prev_mean) - b_global * logit_p_pos
        # Replace per-sample with scalar
        b = np.full_like(y_prob, float(b_global))
        a = np.full_like(y_prob, float(a_global))

    # ── Auto mode: decide strategy based on mean(b) ───────────────
    mode_used = blend_mode
    if blend_mode == 'auto':
        mean_b = float(np.mean(b))
        if mean_b > auto_b_threshold:
            mode_used = 'full'
        else:
            mode_used = 'min_ppv_npv'
            b_clamp = (0.0, 100.0)
    elif blend_mode == 'auto_smooth':
        mean_b = float(np.mean(b))
        # Shift sigmoid center based on mean(b):
        #   center = 1.0 - sigmoid(3 * mean_b)
        #   mean_b= 1.0 → sigmoid(3)=0.95 → center=0.05 → high alpha
        #   mean_b= 0.0 → sigmoid(0)=0.50 → center=0.50 → moderate
        #   mean_b=-1.0 → sigmoid(-3)=0.05 → center=0.95 → low alpha
        # Store center for use in blending step
        _auto_smooth_center = 1.0 - 1.0 / (1.0 + np.exp(-3.0 * mean_b))
        mode_used = 'auto_smooth'
        b_clamp = (0.0, 100.0)

    # ── Stabilisation ──────────────────────────────────────────────
    a_eff, b_eff = a.copy() if hasattr(a, 'copy') else a, b.copy() if hasattr(b, 'copy') else b

    # 1. Clamp b
    if b_clamp is not None:
        if isinstance(b_clamp, (list, tuple)):
            lo, hi = b_clamp
        else:
            lo, hi = 1.0 / b_clamp, float(b_clamp)
        b_eff = np.clip(b_eff, lo, hi)
        # Recompute a to stay consistent with PPV constraint
        a_eff = _logit(ppv) - _logit(prevalence) - b_eff * logit_p_pos

    # 2. Shrinkage toward identity (a=0, b=1)
    if shrinkage > 0:
        a_eff = (1.0 - shrinkage) * a_eff
        b_eff = (1.0 - shrinkage) * b_eff + shrinkage

    calibrated = _sigmoid(_logit(prevalence) + a_eff + b_eff * _logit(y_prob))
    calibrated = np.where(np.isfinite(calibrated), calibrated, 0.5)

    # 3. Blend with original
    alpha = blend_alpha  # default
    if mode_used == 'auto_smooth':
        # Smooth auto: shift sigmoid center by mean(b)
        confidence = np.minimum(ppv, npv)
        alpha = _sigmoid(adaptive_scale * (confidence - _auto_smooth_center))
        calibrated = alpha * calibrated + (1.0 - alpha) * y_prob
    elif mode_used == 'min_ppv_npv':
        # Per-sample alpha based on min(PPV, NPV) — conservative: requires
        # BOTH to be high before trusting calibration
        confidence = np.minimum(ppv, npv)
        alpha = _sigmoid(adaptive_scale * (confidence - 0.5))
        calibrated = alpha * calibrated + (1.0 - alpha) * y_prob
    elif mode_used == 'full':
        # Full Bayesian — no blending (auto mode selected this)
        alpha = np.ones_like(y_prob) if y_prob.ndim > 0 else 1.0
    elif adaptive_blend:
        # Legacy: per-sample alpha based on PPV + NPV
        ppv_plus_npv = ppv + npv
        alpha = _sigmoid(adaptive_scale * (ppv_plus_npv - 1.0))
        calibrated = alpha * calibrated + (1.0 - alpha) * y_prob
    elif blend_alpha < 1.0:
        calibrated = blend_alpha * calibrated + (1.0 - blend_alpha) * y_prob

    return calibrated, {'a': a, 'b': b, 'a_eff': a_eff, 'b_eff': b_eff,
                        'alpha': alpha, 'blend_mode_used': mode_used}


def binned_correlations(data, dist_col, perf_metrics, bin_num):
    """Bin data by dist_col and return Pearson/Spearman correlations per metric.

    Parameters:
        data: DataFrame with 'label', 'pred', and dist_col columns
        dist_col: column name for distances
        perf_metrics: list of metric names (e.g., ['aucroc', 'ap', 'acc'])
        bin_num: number of equal-sized bins

    Returns:
        dict[metric] -> {pearson_r, pearson_p, spearman_r, spearman_p,
                         bin_dists, bin_perfs}
    """
    sorted_d = data.sort_values(by=dist_col).reset_index(drop=True)
    bs = len(sorted_d) // bin_num
    bin_dists, bin_perfs = [], {m: [] for m in perf_metrics}

    for i in range(bin_num):
        s = i * bs
        e = len(sorted_d) if i == bin_num - 1 else (i + 1) * bs
        bd = sorted_d.iloc[s:e]
        bin_dists.append(bd[dist_col].mean())
        for m in perf_metrics:
            bin_perfs[m].append(safe_metric(m, bd['label'].values, bd['pred'].values))

    results = {}
    bd_arr = np.array(bin_dists)
    for m in perf_metrics:
        bp = np.array(bin_perfs[m])
        valid = ~np.isnan(bp)
        if valid.sum() >= 3 and np.ptp(bd_arr[valid]) > 0 and np.ptp(bp[valid]) > 0:
            rp, pp = pearsonr(bd_arr[valid], bp[valid])
            rs, ps = spearmanr(bd_arr[valid], bp[valid])
            try:
                lr = linregress(bd_arr[valid], bp[valid])
                slope = lr.slope
            except ValueError:
                slope = np.nan
            results[m] = {'pearson_r': rp, 'pearson_p': pp,
                          'spearman_r': rs, 'spearman_p': ps,
                          'slope': slope,
                          'bin_dists': bd_arr.tolist(),
                          'bin_perfs': bp.tolist()}
        else:
            results[m] = {'pearson_r': np.nan, 'pearson_p': np.nan,
                          'spearman_r': np.nan, 'spearman_p': np.nan,
                          'slope': np.nan,
                          'bin_dists': bd_arr.tolist(),
                          'bin_perfs': bp.tolist()}
    return results


def per_epitope_correlations(ep_df, dist_col, perf_metrics):
    """Compute correlations across individual epitopes.

    Parameters:
        ep_df: DataFrame with per-epitope metrics and a distance column
        dist_col: column name for distances
        perf_metrics: list of metric names

    Returns:
        dict[metric] -> {pearson_r, pearson_p, spearman_r, spearman_p}
    """
    results = {}
    for m in perf_metrics:
        v = ep_df[ep_df[m].notna()]
        if len(v) >= 3:
            rp, pp = pearsonr(v[dist_col], v[m])
            rs, ps = spearmanr(v[dist_col], v[m])
            results[m] = {'pearson_r': rp, 'pearson_p': pp,
                          'spearman_r': rs, 'spearman_p': ps}
        else:
            results[m] = {'pearson_r': np.nan, 'pearson_p': np.nan,
                          'spearman_r': np.nan, 'spearman_p': np.nan}
    return results


def plot_binned_curves(seen_binned, unseen_binned, eval_metrics, title, fname,
                       combined_binned=None):
    """Plot generalization curves (performance vs distance) for all metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_panels = 3 if combined_binned else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))

    ax = axes[0]
    ax.set_title('Seen Epitopes (Binned)', fontsize=13, fontweight='bold')
    for m in eval_metrics:
        bd = seen_binned[m].get('bin_dists', [])
        bp = seen_binned[m].get('bin_perfs', [])
        if bd and bp:
            ax.plot(bd, bp, 'o-', label=m, markersize=6)
    ax.set_xlabel('Log-transformed Distance', fontsize=11)
    ax.set_ylabel('Performance', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    ax = axes[1]
    ax.set_title('Unseen Epitopes (Binned)', fontsize=13, fontweight='bold')
    for m in eval_metrics:
        bd = unseen_binned[m].get('bin_dists', [])
        bp = unseen_binned[m].get('bin_perfs', [])
        if bd and bp:
            ax.plot(bd, bp, 'o-', label=m, markersize=6)
    ax.set_xlabel('Log-transformed Distance', fontsize=11)
    ax.set_ylabel('Performance', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    if combined_binned:
        ax = axes[2]
        ax.set_title('Combined Seen+Unseen (Binned)', fontsize=13, fontweight='bold')
        for m in eval_metrics:
            bd = combined_binned[m].get('bin_dists', [])
            bp = combined_binned[m].get('bin_perfs', [])
            if bd and bp:
                ax.plot(bd, bp, 'o-', label=m, markersize=6)
        ax.set_xlabel('Log-transformed Distance', fontsize=11)
        ax.set_ylabel('Performance', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


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

    ref_data = read_table(args.ref)
    qry_data = read_table(args.qry)

        
    # print(qry_data['Heavy'])
    ref_data=ref_data.iloc[0:50]
    qry_data=qry_data.iloc[50:100]

    general_eval = General_Evaluator(ref_data=ref_data, qry_data=qry_data,header_dict=header_dict)
    eval_data = general_eval.eval_generalization(split_by="seq_edit_dist",eval_by="acc")
    print("Evaluation Data:", eval_data)
