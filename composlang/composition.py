import typing
from composlang.corpus import Corpus
import pandas as pd
from functools import partial
from collections import defaultdict, Counter  # , OrderedDict
from tqdm.auto import tqdm
from composlang.utils import log, pathify
import numpy as np
from scipy.sparse import dok_array

# def entropy_np(pk: np.typing.ArrayLike,
#                base: typing.Optional[float] = 2,
#                axis: int = 0):
#     pk_norm = pk / pk.sum(axis=axis)
#     base = np.e if base is None else base
#     return -(pk_norm * np.log(pk_norm) / np.log(base)).sum(axis=axis)


class CompositionAnalysis:

    matrix = None

    def __init__(
        self,
        corpus: Corpus = None,
        child_upos: str = "ADJ",
        parent_upos: str = "NOUN",
        run_analyses=False,
    ):

        self.child_upos: str = child_upos
        self.parent_upos: str = parent_upos

        if corpus is not None:
            self._crunch_corpus_stats(corpus=corpus)
            if run_analyses:
                self.run_analyses()

    def _crunch_corpus_stats(self, corpus: Corpus):

        child_upos = self.child_upos
        parent_upos = self.parent_upos

        def filter_by_upos(dict_item: typing.Tuple[tuple, int], upos: str):
            (t, u), ct = dict_item
            return u == upos

        child_dicts = [
            dict(token=t, upos=u, freq=ct)
            for (t, u), ct in filter(
                partial(filter_by_upos, upos=child_upos), corpus.token_stats.items()
            )
        ]
        parent_dicts = [
            dict(token=t, upos=u, freq=ct)
            for (t, u), ct in filter(
                partial(filter_by_upos, upos=parent_upos), corpus.token_stats.items()
            )
        ]

        self.n_tokens = sum(corpus.token_stats.values())  # stat needed for PMI
        self.token_stats = {
            k: v
            for (k, kupos), v in corpus.token_stats.items()
            if kupos in (self.child_upos, self.parent_upos)
        }

        # make DataFrames to track various stats about each lexical item, separated by UPOS
        self.child_df = pd.DataFrame(child_dicts).sort_values(
            "freq", ascending=False, ignore_index=True
        )
        self.parent_df = pd.DataFrame(parent_dicts).sort_values(
            "freq", ascending=False, ignore_index=True
        )

        # pair-related stuff
        self.pair_stats = Counter(
            {
                (c, p, deprel): ct
                for ((c, cu), (p, pu), deprel), ct in corpus.pair_stats.items()
                if cu == child_upos and pu == parent_upos
            }
        )
        self.pair_df = pd.DataFrame(
            [
                dict(child=c, parent=p, pair=(c, p), freq=ct, deprel=deprel)
                for (c, p, deprel), ct in self.pair_stats.items()
            ]
        ).sort_values("freq", ascending=False, ignore_index=True)

        # self.skip_pair_stats = Counter(
        #     {
        #         (c, p): ct
        #         for ((c, cu), (p, pu)), ct in corpus.skip_pair_stats.items()
        #         if cu == child_upos and pu == parent_upos
        #     }
        # )
        # self.skip_pair_df = pd.DataFrame(
        #     [
        #         dict(child=c, parent=p, pair=(c, p), freq=ct)
        #         for (c, p), ct in self.skip_pair_stats.items()
        #     ]
        # ).sort_values("freq", ascending=False, ignore_index=True)

    def run_analyses(self):
        self.compute_combinations()
        self.compute_entropy()
        self.compute_pmi()

    def to_csv(self, directory):
        directory = pathify(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for attr in ["child_df", "parent_df", "pair_df", "skip_pair_df"]:
            getattr(self, attr).to_csv(directory / f"{attr}.csv")

    def read_csv(self, directory):
        directory = pathify(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for attr in ["child_df", "parent_df", "pair_df", "skip_pair_df"]:
            obj = pd.read_csv(
                directory / f"{attr}.csv", index_col=0, converters={"token": str}
            )
            setattr(self, attr, obj)

    def generate_adjacency_matrix(self, skip=False, stat="freq", top=None, threshold=0):
        """Generates an adjacency matrix with shape (n_child_tokens, n_parent_tokens),
            with the i,j element representing the count of child[i], parent[j] occurring
            together

        Returns:
            np.ndarray: adjacency matrix
        """
        if self.matrix is not None:
            return self.matrix

        child_tokens = self.child_df["token"].iloc[:top]
        parent_tokens = self.parent_df["token"].iloc[:top]

        child_to_ix = {token: ix for ix, token in enumerate(child_tokens)}
        parent_to_ix = {token: ix for ix, token in enumerate(parent_tokens)}

        # TODO: explore the possibility of using scipy.sparse
        # matrix[i, :] -> distribution over all parent tokens for ith child token (e.g., parent=NOUN, child=ADJ)
        # matrix[:, i] -> distribution over all child tokens for ith parent token (e.g., parent=NOUN, child=ADJ)

        # matrix = np.zeros((len(child_tokens), len(parent_tokens)))
        matrix = dok_array((len(child_tokens), len(parent_tokens)), dtype=np.int64)

        view = (
            self.skip_pair_df if skip else self.pair_df
        )  # .sort_values(stat).iloc[:top]
        view = view[view[stat] > threshold]
        it = view.iterrows()
        total = len(view)
        for _, row in tqdm(it, total=total, desc="constructing adjacency matrix"):
            # if cu == self.child_upos and pu == self.parent_upos:
            c, p, value = row[["child", "parent", stat]]
            try:
                cix = child_to_ix[c]
                pix = parent_to_ix[p]
            except KeyError:
                # print(c,p,value)
                continue
            matrix[cix, pix] = value

        self.matrix = matrix
        return self.matrix

    def compute_combinations(self) -> None:
        """counts the number of combinations possible for each child and parent UPOS according
        to self.child_upos and self.parent_upos at initialization. adds it to the two pandas
        DataFrame objects of this instance under column names `combinations[_collapsed]`
        where `_collapsed` is the version of the count that considers the counts of all observed
        occurrences as 1
        """

        for key, stat_dict in (
            ("combinations", self.pair_stats),
            # ("skip_combinations", self.skip_pair_stats),
        ):
            if key in self.child_df and key in self.parent_df:
                continue

            child_to_parent = defaultdict(int)
            collapsed_child_to_parent = defaultdict(int)
            parent_to_child = defaultdict(int)
            collapsed_parent_to_child = defaultdict(int)

            for (w, p), ct in tqdm(stat_dict.items(), desc=f"counting {key}"):
                # if (child_upos == '*' or wupos == child_upos) and (parent_upos == '*' or pupos == parent_upos):
                # if wupos == child_upos and pupos == parent_upos:
                child_to_parent[w] += ct
                collapsed_child_to_parent[w] += 1
                parent_to_child[p] += ct
                collapsed_parent_to_child[p] += 1

            self.child_df[key] = [child_to_parent[k] for k in self.child_df["token"]]
            self.parent_df[key] = [parent_to_child[k] for k in self.parent_df["token"]]

            self.child_df[f"{key}_collapsed"] = [
                collapsed_child_to_parent[k] for k in self.child_df["token"]
            ]
            self.parent_df[f"{key}_collapsed"] = [
                collapsed_parent_to_child[k] for k in self.parent_df["token"]
            ]

    def compute_entropy(self):
        """Computes entropy of each child_upos and parent_upos, and adds them to the stats DataFrame members
        of this instance
        """

        for column_name, skip in [
            ("entropy", False),
            # ("entropy_skip", True),
        ]:

            if column_name in self.child_df and column_name in self.parent_df:
                return

            from scipy.stats import entropy

            matrix = self.generate_adjacency_matrix(skip=skip)
            log(f"computing {column_name} for {self.child_upos}")
            child_ent = entropy(matrix, axis=1)
            log(f"computing {column_name} for {self.parent_upos}")
            parent_ent = entropy(matrix, axis=0)

            self.child_df[column_name] = child_ent
            self.parent_df[column_name] = parent_ent

            for df in (self.child_df, self.parent_df):
                for key in (column_name,):
                    df[key].fillna(0, inplace=True)

            parent_norm_mask = self.parent_df.freq < matrix.shape[0]
            child_norm_mask = self.child_df.freq < matrix.shape[1]

            parent_norm_denom = (parent_norm_mask * self.parent_df.freq) + (
                ~parent_norm_mask * matrix.shape[0]
            )
            child_norm_denom = (child_norm_mask * self.child_df.freq) + (
                ~child_norm_mask * matrix.shape[1]
            )

            self.parent_df[f"{column_name}_ceilinged"] = self.parent_df[
                f"{column_name}"
            ] / np.log2(parent_norm_denom)
            self.child_df[f"{column_name}_ceilinged"] = self.child_df[
                f"{column_name}"
            ] / np.log2(child_norm_denom)

    def compute_pmi(self):
        """ """
        # we will have a PMI value per pair
        for key, stat_df in (
            ("pmi", self.pair_df),
            # ("skip_pmi", self.skip_pair_df)
        ):
            if key in self.child_df and key in self.parent_df:
                continue

            pmi = []
            for i, (w, p, (w, p), ct) in tqdm(
                stat_df.iterrows(), total=len(stat_df), desc=f"processing {key}"
            ):
                p_given_w_ct = np.log2(ct)
                p_ct = np.log2(self.token_stats[p])
                w_ct = np.log2(self.token_stats[w])
                pmi_w_p = (p_given_w_ct - w_ct) - (p_ct - np.log2(self.n_tokens))
                pmi += [pmi_w_p]

            stat_df[key] = pmi

    def bipartite_layout(self):
        raise NotImplementedError

    def inspect_neighborhood(self, token):
        return Counter(
            {(c, p): ct for (c, p), ct in self.pair_stats.items() if token in (c, p)}
        )

    def sample_combinations(self, n=1_000):
        import random

        items = list((T, ct) for T, ct in self.pair_stats.items())
        return random.choices(items, k=n)

    def generate_combinations(
        self, min_freq=1, n=100, use_freq_as_weights=True, seed=42
    ):
        """Generates `n` Adj x Noun combinations by independently sampling Adj and Noun `n` times
            with replacement.  By default, sampling is true to the observed Adj and Noun
            distributions, so will be approximately Zipfian, but turning this off using
            `use_freq_as_weights` results in sampling Adjs and Nouns uniformly, leading to many more
            unattested and low-frequency pairings.

        Args:
            min_freq (int, optional): Whether to only sample above a certain frequency for
                individual Adjs and Nouns. Defaults to 1.
            n (int, optional): Number of combinations to produce. Defaults to 100.
            use_freq_as_weights (bool, optional): Whether sampling should be faithful to observed distribution
                over individual lexical items.  If False, a uniform prior over lexical items is
                assumed. Defaults to True.

        Returns:
            list: A list of generated combinations
        """
        if use_freq_as_weights:
            C = self.child_df[self.child_df.freq >= min_freq].sample(
                n, replace=True, weights="freq", random_state=seed
            )
            P = self.parent_df[self.parent_df.freq >= min_freq].sample(
                n, replace=True, weights="freq", random_state=seed
            )
        else:
            C = self.child_df[self.child_df.freq >= min_freq].sample(
                n, replace=True, random_state=seed
            )
            P = self.parent_df[self.parent_df.freq >= min_freq].sample(
                n, replace=True, random_state=seed
            )

        samples = []
        for rowC, rowP in tqdm(
            zip([row for i, row in C.iterrows()], [row for i, row in P.iterrows()]),
            total=len(C),
        ):
            # print(rowC, rowP)
            colnames = ["token", "freq", "combinations"]
            c, cfreq, ccomb = rowC[colnames]
            p, pfreq, pcomb = rowP[colnames]

            freq = self.pair_stats[c, p]
            # try:
            #     f = self.pair_df[self.pair_df["pair"] == str((c, p))]["freq"].item()
            # except ValueError:
            #     f = 0
            # pair = (c, self.child_upos), (p, self.parent_upos)
            # samples.append(((c, p), self.pair_df[self.pair_df['pair'] == (c,p)]['freq']))
            samples.append(
                dict(
                    child=c,
                    childfreq=cfreq,
                    childcombs=ccomb,
                    parent=p,
                    parentfreq=pfreq,
                    parentcombs=pcomb,
                    pairfreq=freq,
                )
            )
        return samples
