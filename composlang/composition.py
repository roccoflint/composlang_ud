
import typing
from composlang.corpus import Corpus
import pandas as pd
from functools import partial
from collections import defaultdict, Counter#, OrderedDict
from tqdm.auto import tqdm
from composlang.utils import log
import numpy as np

# def entropy_np(pk: np.typing.ArrayLike,
#                base: typing.Optional[float] = 2,
#                axis: int = 0):
#     pk_norm = pk / pk.sum(axis=axis)
#     base = np.e if base is None else base
#     return -(pk_norm * np.log(pk_norm) / np.log(base)).sum(axis=axis)


class CompositionAnalysis:

    def __init__(self, corpus: Corpus, 
                 child_upos: str = 'ADJ', parent_upos: str = 'NOUN',
                 run_analyses = True):
        
        self.child_upos: str = child_upos
        self.parent_upos: str = parent_upos

        def filter_by_upos(dict_item: typing.Tuple[tuple, int], upos: str):
            (t, u), ct = dict_item
            return u == upos 

        child_dicts = [dict(token=t, upos=u, freq=ct) for (t,u), ct in filter(partial(filter_by_upos, upos=child_upos), corpus.token_stats.items())]
        parent_dicts = [dict(token=t, upos=u, freq=ct) for (t,u), ct in filter(partial(filter_by_upos, upos=parent_upos), corpus.token_stats.items())]
        self.child_token_stats = pd.DataFrame(child_dicts).sort_values('freq', ascending=False, ignore_index=True)
        self.parent_token_stats = pd.DataFrame(parent_dicts).sort_values('freq', ascending=False, ignore_index=True)

        self.pair_stats: Counter = Counter({(c,p): ct for ((c,cu),(p,pu)), ct in corpus.pair_stats.items() 
                                           if cu == child_upos and pu == parent_upos})

        if run_analyses:
            self.compute_combinations()
            self.compute_entropy()
    

    def generate_adjacency_matrix(self):
        """Generates an adjacency matrix with shape (n_child_tokens, n_parent_tokens),
            with the i,j element representing the count of child[i], parent[j] occurring
            together

        Returns:
            np.ndarray: adjacency matrix
        """
        child_tokens = self.child_token_stats['token']
        parent_tokens = self.parent_token_stats['token'] 

        child_to_ix = {token: ix for ix, token in enumerate(child_tokens)}
        parent_to_ix = {token: ix for ix, token in enumerate(parent_tokens)}

        # TODO: explore the possibility of using scipy.sparse
        # matrix[i, :] -> distribution over all parent tokens for ith child token 
        # matrix[:, i] -> distribution over all child tokens for ith parent token 
        matrix = np.zeros((len(child_tokens), len(parent_tokens)))

        for (c, p), ct in tqdm(self.pair_stats.items(), desc='constructing adjacency matrix'):
            # if cu == self.child_upos and pu == self.parent_upos:
            try:
                cix = child_to_ix[c]
                pix = parent_to_ix[p]
            except KeyError:
                print(c,p,ct)
            matrix[cix, pix] = ct

        return matrix


    def compute_combinations(self): 
        """counts the number of combinations possible for each child and parent UPOS according
            to self.child_upos and self.parent_upos at initialization. adds it to the two pandas
            DataFrame objects of this instance under column names `combinations[_collapsed]`
            where `_collapsed` is the version of the count that considers the counts of all observed
            occurrences as 1
        """
        if 'combinations' in self.child_token_stats and 'combinations' in self.parent_token_stats:
            return 

        child_upos, parent_upos = self.child_upos, self.parent_upos

        child_to_parent = defaultdict(int)
        collapsed_child_to_parent = defaultdict(int)
        parent_to_child = defaultdict(int)
        collapsed_parent_to_child = defaultdict(int)

        for (w, p), ct in tqdm(self.pair_stats.items(), desc='counting combinations'):
            # if (child_upos == '*' or wupos == child_upos) and (parent_upos == '*' or pupos == parent_upos):
            # if wupos == child_upos and pupos == parent_upos:
            child_to_parent[w] += ct
            collapsed_child_to_parent[w] += 1
            parent_to_child[p] += ct
            collapsed_parent_to_child[p] += 1

        self.child_token_stats['combinations'] = [child_to_parent[k] for k in self.child_token_stats['token']]
        self.parent_token_stats['combinations'] = [parent_to_child[k] for k in self.parent_token_stats['token']]

        self.child_token_stats['combinations_collapsed'] = [collapsed_child_to_parent[k] for k in self.child_token_stats['token']]
        self.parent_token_stats['combinations_collapsed'] = [collapsed_parent_to_child[k] for k in self.parent_token_stats['token']]

    
    def compute_entropy(self):
        """Computes entropy of each child_upos and parent_upos, and adds them to the stats DataFrame members
            of this instance
        """
        if 'entropy' in self.child_token_stats and 'entropy' in self.parent_token_stats:
            return 

        from scipy.stats import entropy

        matrix = self.generate_adjacency_matrix()
        log(f'computing entropy for {self.child_upos}')
        child_ent = entropy(matrix, axis=1)
        log(f'computing entropy for {self.parent_upos}')
        parent_ent = entropy(matrix, axis=0)

        self.child_token_stats['entropy'] = child_ent
        self.parent_token_stats['entropy'] = parent_ent

        for df in (self.child_token_stats, self.parent_token_stats):
            for key in ('entropy', ):
                df[key].fillna(0, inplace=True)

        # parent_numocc = matrix.sum(axis=0)
        # child_numocc = matrix.sum(axis=1)
        
        # parent_norm_mask = (parent_numocc < matrix.shape[0]) #.astype('int64')
        # child_norm_mask = (child_numocc < matrix.shape[1]) #.astype('int64')

        # parent_norm_denom = (parent_norm_mask * matrix.shape[0]) | 1
        # child_norm_denom = (child_norm_mask * matrix.shape[1]) | 1
        
        parent_norm_mask = (self.parent_token_stats.freq < matrix.shape[0])
        child_norm_mask = (self.child_token_stats.freq < matrix.shape[1])

        parent_norm_denom = (parent_norm_mask * self.parent_token_stats.freq) + (~parent_norm_mask * matrix.shape[0])
        child_norm_denom = (child_norm_mask * self.child_token_stats.freq) + (~child_norm_mask * matrix.shape[1])

        self.parent_token_stats['entropy_ceilinged'] = self.parent_token_stats['entropy'] / np.log2(parent_norm_denom)
        self.child_token_stats['entropy_ceilinged'] = self.child_token_stats['entropy'] / np.log2(child_norm_denom)


    def bipartite_layout(self):
        ...


    def inspect_neighborhood(self, token):
        return Counter({(c, p): ct for (c, p), ct in self.pair_stats.items() if token in (c, p)})


    def sample_combinations(self, n=1_000):
        import random
        items = list((T, ct) for T, ct in self.pair_stats.items())
        return random.choices(items, k=n)
        

    def generate_combinations(self, min_freq=2, n=100):
        
        C = self.child_token_stats[self.child_token_stats.freq >= min_freq].token.sample(n)
        P = self.parent_token_stats[self.parent_token_stats.freq >= min_freq].token.sample(n)

        samples = []
        for c, p in zip(C, P):
            pair = (c, self.child_upos), (p, self.parent_upos)
            samples.append(((c, p), self.pair_stats[pair]))
        return samples