import typing
from sys import stderr
from pathlib import Path


def log(*args, **kwargs):
    """
    placeholder logging method to be changed later
    """
    print("** info:", *args, **kwargs, file=stderr)


def pathify(fpth: typing.Union[Path, str, typing.Any]) -> Path:
    """
    returns a resolved `Path` object after expanding user and shorthands/symlinks
    """
    return Path(fpth).expanduser().resolve()


def iterable_from_directory_or_filelist(directory_or_filelist) -> typing.Iterable[Path]:
    """
    constructs an iterable over Path objects using a given directory, file, or
    list of files as either str or Path objects
    """
    # either it must be a list of files, or a directory containing files, or a file
    # first, assume it is a directory or individual file
    try:
        directory_or_filelist = pathify(directory_or_filelist)
        if directory_or_filelist.is_dir():
            files = [*directory_or_filelist.iterdir()]
        else:
            files = [directory_or_filelist]
    # next, assume we are given a list of filepaths as strs or or Path-like objects
    except TypeError as e:
        files = list(map(pathify, directory_or_filelist))

    return files


def minmax(a):
    return (a - a.min()) / (a.max() - a.min())


def ECDF_transform(a):
    from statsmodels.distributions.empirical_distribution import ECDF

    return ECDF(a)(a)


class JointDist:
    def __init__(
        self,
        adj_weights: typing.Collection[typing.Tuple[str, float]],
        noun_weights: typing.Collection[typing.Tuple[str, float]],
        pair_weights: typing.Collection[
            typing.Tuple[
                typing.Tuple[str, str],
                float,
            ]
        ],
        m=1_000,
    ):
        import numpy as np
        from tqdm.auto import tqdm
        from scipy.special import logsumexp

        self.m = m
        self.adj_index: typing.Dict[str, int] = {}
        self.noun_index: typing.Dict[str, int] = {}
        self.adj_weights = []
        self.noun_weights = []

        for i, a in enumerate(adj_weights):
            if i >= m:
                break
            if a not in self.adj_index:
                self.adj_index[a] = len(self.adj_index)
                self.adj_weights.append(adj_weights[a])
        for i, n in enumerate(noun_weights):
            if i >= m:
                break
            if n not in self.noun_index:
                self.noun_index[n] = len(self.noun_index)
                self.noun_weights.append(noun_weights[n])

        self.joint = np.zeros((m, m)) - np.inf
        for an, p in tqdm(pair_weights.items(), total=len(pair_weights)):
            try:
                self[an.split()] = p
            except KeyError:
                pass
            except TypeError:
                print(an)

        # normalize to get a proper joint distribution
        self.joint -= logsumexp(self.joint)

    def get_index(self, adj, noun) -> typing.Tuple[int, int]:
        aix = self.adj_index[adj] if isinstance(adj, str) else adj or ...
        nix = self.noun_index[noun] if isinstance(noun, str) else noun or ...
        return aix, nix

    def __getitem__(self, key) -> float:
        if isinstance(key, str):
            key = key.split()
        ix = self.get_index(*key)
        return self.joint[ix]

    def __setitem__(self, key, value):
        ix = self.get_index(*key)
        self.joint[ix] = value

    def conditionalize(self, axis: int = 0) -> "JointDist":
        """
        axis 0 corresponds to marginalizing over adjectives to get p(n|a).
            we sum over axis 0 (sum the distribution corresponding to each adjective)
            and divide by it, so that row is left with p(n|a)
        axis 1 corresponds to marginalizing over nouns to get p(a|n) (by summing over axis 1)
            we sum over axis 1 (sum the distribution corresponding to each noun)
            and divide by it, so that column is left with p(a|n)
        """
        import copy
        from scipy.special import logsumexp

        if axis == 0:
            new_joint = self.joint - logsumexp(self.joint, axis=1 - axis)[:, None]
        elif axis == 1:
            new_joint = self.joint - logsumexp(self.joint, axis=1 - axis)
        else:
            raise ValueError("axis must be 0 or 1 for bivariate joint distribution")
        self_copy = copy.deepcopy(self)
        self_copy.joint = new_joint
        return self_copy

    def get_marginal_of_axis(self, axis: int) -> "np.ndarray":
        """
        returns the marginal distribution (one-dimensional) along the specified axis
        axis 0 corresponds to marginalizing  to get p(a)
        """
        return self.joint.sum(axis=1 - axis)


def get_llm_results(
    model: str,
    study: str,
    basedir: Path = Path("./llm-results/"),
    adj_p: typing.Dict[str, float] = None,
    noun_p: typing.Dict[str, float] = None,
    pair_p: typing.Dict[typing.Tuple[str, str], float] = None,
    joint_p: JointDist = None,
):
    """ """
    import pandas as pd
    import numpy as np
    from scipy.special import logsumexp
    from functools import reduce

    # log(model, study, paradigm)
    assert all(x is not None for x in (adj_p, noun_p, pair_p)), "missing frequency data"

    [resultsdir] = (basedir / study).glob("benchmark-cfg=*")

    results = []
    for paradigm in ["logprobs", "likert"]:
        results_csvs = list(resultsdir.glob(f"eval={paradigm}*/model={model}/*.csv"))
        if paradigm == "logprobs":
            results_paradigm = [
                pd.read_csv(r_path, index_col=0) for r_path in results_csvs
            ]
        elif paradigm == "likert":
            results_paradigm = []
            for r_path in results_csvs:
                r = pd.read_csv(r_path, index_col=0)
                likert_kind = r_path.parts[3].split("=")[1]
                r.rename(
                    columns={"text_likert_noun_adjective": likert_kind}, inplace=True
                )
                results_paradigm.append(r)
        else:
            raise ValueError(f"Unrecognized paradigm: {paradigm}")

        results += results_paradigm

    if len(results) > 1:
        results = reduce(pd.merge, results)
    elif len(results) == 1:
        results = results[0]
    else:
        raise ValueError(f"Found {len(results)} results")

    # exclude human study results from corpus study results
    if study == "composlang":
        results = results[(results["arank"] >= 0) & (results["nrank"] >= 0)]
    elif study == "composlang-beh":
        results["rating"] += 4

    # any column header that is the result of model outputs
    model_columns = [
        "likert_constrained_original",
        "likert_constrained_optimized",
        "logp_A",
        "logp_N",
        "logp_N_A",
        "logp_AN",
    ]

    # make a note of what model this is
    results[("model")] = model

    # these are frequencies provided independently, calculated using some corpus (e.g. COCA)
    results[("clogp_A")] = results[("adjective")].apply(str.lower).map(adj_p)
    # results[("clogp_A")] = np.log((1 + results[("adj_freq")]) / sum(adj_freq.values()))

    results[("clogp_N")] = results[("noun")].apply(str.lower).map(noun_p)
    # results[("clogp_N")] = np.log( (1 + results[("noun_freq")]) / sum(noun_freq.values()))

    # results[("clogp_N_A")] = results[("clogp_N")] + results[("clogp_A")]

    results[("clogp_AN")] = (
        (results[("adjective")] + " " + results[("noun")]).apply(str.lower).map(pair_p)
    )
    # normalize the joint distribution so this subset sums to 1
    results["clogp_AN"] -= logsumexp(results["clogp_AN"])
    # results[("clogp_AN")] = np.log(
    #     (1 + results[("pair_freq")]) / sum(pair_freq.values())
    # )

    if study == "composlang":
        assert joint_p is not None, "missing joint distribution `JointDist` object"
        cond = joint_p.conditionalize(axis=0)
        results["clogp_N_A"] = (
            (results[("adjective")] + " " + results[("noun")])
            .apply(str.lower)
            .map(cond)
        )

    # in the realm of logprobs, we want -inf to fill in for missing values
    results.fillna(float("-inf"), inplace=True)

    # LOGPROBS directly from LLM
    for col in ["logp_N_A", "logp_AN", "logp_A", "logp_N"]:
        try:
            results[col]
            # ecdf = ECDF_transform(results[(col)])
            # results[(f"ecdf_{col}")] = ecdf

            if col == "logp_N_A":
                conditionals = results[(col)]
                # conditionals are in log space, and adj_freq is absolute counts.
                # rescale to sum of counts of all adjs (not just those in the dataset)
                results[(f"hybrid_{col}")] = conditionals + results["clogp_A"]

                marginals = results[(f"logp_A")]
                results[(f"bayes_{col}")] = conditionals + marginals

        except KeyError:
            if col == "logp_N_A":
                log(f"could not find logp_A in {model} x {study} results; skipping")
            else:
                log(f"could not find {col} in {model} x {study} results; skipping")

    # LIKERT
    for col in ["likert_constrained_original", "likert_constrained_optimized"]:
        try:
            results[(f"{col}")]  # -= 4
        except KeyError:
            log(f"could not find {col} in {model} x {study} results; skipping")

    return results.sort_index(axis=1)


def compute_fit(
    df,
    metric="corpus_logp_N_A",
    target="pair_freq",
    outlier_pct=0,
    flip=False,
    exp=False,
    surp=False,
    normx=False,
    normy=False,
    logy=True,
):
    import statsmodels.api as sm
    import numpy as np

    if outlier_pct > 0:
        lower_bound = df[target].quantile(outlier_pct)
        upper_bound = df[target].quantile(1 - outlier_pct)
        df = df[(df[target] < upper_bound) & (df[target] > lower_bound)]

    # transform target to log space
    if logy:
        y = np.log(df[target] + int(min(df[target]) == 0))
    else:
        y = df[target]
    x = df[metric]

    if surp:
        x = -x
    if exp:
        x = np.exp(x)
    if normx:
        x = (x - x.min()) / (x.max() - x.min())
        # x /= abs(x.sum())
    if normy:
        y = (y - y.min()) / (y.max() - y.min())
        # y /= abs(y.sum())
    if flip:
        x, y = y, x
    model = sm.OLS(y, sm.add_constant(x))
    fit = model.fit()
    return x, y, fit
