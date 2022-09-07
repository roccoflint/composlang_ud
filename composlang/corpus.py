# stdlib
import fileinput
import time
import typing
from collections import Counter, defaultdict
from functools import cached_property, reduce
from pathlib import Path

# installed packages
import numpy as np
from joblib import Parallel, delayed
from more_itertools import peekable
from sqlitedict import SqliteDict
from tqdm.auto import tqdm

# local module
from composlang.graph import WordGraph
from composlang.utils import iterable_from_directory_or_filelist, log, pathify
from composlang.word import Word
from composlang.cache import PickleCache, SQLiteCache


class Corpus:
    """
    A class to read and process data from a corpus in one place;
    accumulate aggregate statistics, etc.
    Expects a dependency-parsed corpus with one token per line.
    Flexible formats are supported as long as they are properly specified.
    The only restriction is that a `sentence_id` column is required in order
    to detect sentence boundaries.
    """

    # determining where to obtain input and how to process it
    _fmt, _sep, _lower = None, None, None
    _files = None
    _n_sentences = None

    # variables related to tracking the current state of reading the corpus
    # for pausing and resuming
    _current_file = None
    _lines_read = 0
    _sentences_seen = 0
    _total = None

    # tracking the linguistic features of interest within the corpus
    _token_stats = None  # (token:str, upos:str) -> num_occ:int
    _pair_stats = None  # ((token1:str, upos1:str), (token2, upos2)) -> num_occ:int
    _triplet_stats = None  # -> num_occ:int
    # in a child-parent relation in a dependency parse
    cache = None

    def __init__(
        self,
        directory_or_filelist: typing.Union[
            Path, str, typing.Iterable[typing.Union[Path, str]]
        ],
        cache_dir: typing.Union[str, Path] = "./cache/composlang",
        cache_tag: str = None,
        cache_cls: type = PickleCache,
        n_sentences: int = None,
        fmt: typing.List[str] = (
            "sentence_id:int",
            "text:str",
            "lemma:str",
            "id:int",
            "head:int",
            "upos:str",
            "deprel:str",
        ),
        sep: str = "\t",
        lowercase: bool = True,
    ):
        """ """
        self._fmt = fmt
        self._sep = sep
        self._lower = lowercase
        self._files = sorted(iterable_from_directory_or_filelist(directory_or_filelist))
        if self._files is None:
            log(f"could not find files at {directory_or_filelist}")
            # raise ValueError(f'could not find files at {directory_or_filelist}')

        self._token_stats = Counter()  # token -> num_occurrences
        self._pair_stats = Counter()  # (token, token) -> num_occurrences
        self._skip_pair_stats = Counter()  # (token, token) -> num_occurrences
        self._triplet_stats = {
            "obj": Counter(),
            "nsubj": Counter(),
        }  # -> num_occurrences

        # loads the pre-existing _pair_stats and _token_stats objects, or creates empty ones
        self._cache_dir = cache_dir
        self._cache_tag = cache_tag
        self._cache_cls = cache_cls
        self.cache = self._get_cache()
        self.load_cache()
        self._n_sentences = n_sentences or float(
            "inf"
        )  # upper limit for no. of sentences to process

    @classmethod
    def from_cache(
        cls,
        cache_file: typing.Union[str, Path],
        n_sentences=None,
        fmt=None,
        sep=None,
        lowercase=None,
    ):
        path = pathify(cache_file)
        cache_dir = path.parent
        cache_tag = path.parts[-1]
        return cls("/dev/null", cache_dir, cache_tag, n_sentences, fmt, sep, lowercase)

    def __len__(self) -> int:
        return self._sentences_seen

    # use context manager to handle closing of cache
    def __enter__(self):
        return self  # nothing to do here

    def __exit__(self, *args, **kws):
        self._close_cache()

    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__}; sentences_seen={self._sentences_seen:,}; tokens={len(self.token_stats):,}>"
        return s

    ################################################################
    #### accessing data of the class
    ################################################################

    @property
    def token_stats(self):
        return self._token_stats

    @property
    def pair_stats(self):
        return self._pair_stats

    @property
    def skip_pair_stats(self):
        return self._skip_pair_stats

    @property
    def triplet_stats(self):
        return self._triplet_stats

    @cached_property
    def upos_counts(self, group_by_token: bool = False):
        """get counts of the number of occurrences of each upos.

        Args:
            group_by_token (bool, optional): if True, consider each occurrence
                of a upos for the same token as one occurrence of the upos.
                if False, count each occurrence of a upos as distinct. i.e., the sum
                of all upos occurrences should be ~ the same as the number of tokens.
                Defaults to False.

        Returns:
            typing.Mapping: a (UPOS -> count) object
        """
        if group_by_token:
            return Counter(upos for token, upos in self._token_stats)

        # not grouping, so we want to consider each occurrence of each token
        d = defaultdict(int)
        for (token, upos), ct in self._token_stats.items():
            token
            d[upos] += ct
        return d

    ################################################################
    #### caching behavior
    ################################################################

    def _close_cache(self):
        try:
            self.cache.commit()
            self.cache.close()
        except Exception as e:
            log(f"encountered error in closing cache: {e}")
            pass

    def _get_cache(self, prefix=None, tag=None):
        """
        creates and returns an SqliteDict() object
        """
        tag = tag or self._cache_tag
        if tag is None:
            from hashlib import shake_128

            s = shake_128()
            s.update(bytes(" ".join(map(str, self._files)), "utf-8"))
            self._cache_tag = tag = s.hexdigest(5)

        root = Path(prefix or self._cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        root /= tag
        return self._cache_cls(str(root), flag="c")

    _attrs_to_cache = (
        ("_sentences_seen", int),
        ("_lines_read", int),
        ("_current_file", lambda: None),
        ("_token_stats", Counter),
        ("_pair_stats", Counter),
        ("_skip_pair_stats", Counter),
        ("_triplet_stats", dict),
        ("_files", lambda: None),
        ("_total", lambda: None),
        ("_n_sentences", lambda: None),
    )

    def load_cache(self, allow_empty=True):
        """
        recover core data of this instance from cache
        """
        log(f"attempt loading from cache at {self._cache_dir}/{self._cache_tag}")
        start = time.process_time()
        for attr, default in tqdm(
            self._attrs_to_cache, desc="loading key-value pairs from cache"
        ):
            print(f"{attr}", end=" ")
            try:
                obj = self.cache[attr]
                setattr(self, attr, obj)
            except KeyError as e:
                log(f"could not find attribute {attr} in cache")
                if not allow_empty:
                    raise e
        end = time.process_time()
        log(
            f"successfully loaded cached data from {self._cache_dir}/{self._cache_tag} in {end-start:.3f} seconds"
        )

    def to_cache(self):
        """
        dump critical state data of this instance to cache
        """
        log(f"caching to {self._cache_dir}/{self._cache_tag}")
        start = time.process_time()
        for attr, type in self._attrs_to_cache:
            obj = getattr(self, attr)
            self.cache[attr] = obj

        self.cache.commit()
        end = time.process_time()
        log(
            f"successfully cached to {self._cache_dir}/{self._cache_tag} in {end-start:.3f} seconds"
        )

    ################################################################
    #### read corpus
    ################################################################

    def read(self, batch_size: int = 5_000, parallel: bool = True):
        """Reads a parsed corpus file.
            the corpus file is formatted similar to the example in the `sample_input`
            directory of this project, or according to a custom format which specified at
            this object's instantiation (using `fmt`, `sep` arguments). See `Corpus.__init__`

        Args:
            batch_size (int, optional): Size of sentence batches to accummulate before
                processing. Writing to cache is also done in `batch_size` increments.
                Defaults to 5_000.
        """
        from stanza.models.common.doc import Document

        dummy_line = lambda: f'{"__EOF__"}{self._sep}' + self._sep.join(
            map(str, range(6 + 10))
        )
        # if we are resuming from previous state, we want to skip lines that are already processed.
        lines_to_skip = self._lines_read
        n_sentences = self._n_sentences  # upper limit

        # sum the total lines in a file for tqdm without loading them all in memory
        if self._total is None:
            if n_sentences >= float("inf"):
                with fileinput.input(files=self._files) as f:
                    self._total = sum(
                        1 for _ in tqdm(f, desc=f"counting # of lines in corpus")
                    )
                log(f"Preparing to read {self._total} lines")
            else:
                self._total = n_sentences
                log(f"Preparing to read {self._total} sentences")

        anchor_time = time.process_time()
        with fileinput.input(files=self._files) as f, tqdm(
            total=self._total, leave=False
        ) as T:
            # more_itertools.peekable allows seeing future context without using up item from iterator
            f = peekable(f)

            if n_sentences >= float(
                "inf"
            ):  # process all sentences; progressbar tracks # of lines
                T.update(self._lines_read)
            else:  # process a predetermiend # of sentences; also reflected in progressbar
                T.update(self._sentences_seen)

            sentence_batch = (
                []
            )  # accumulate parsed sentences to process concurrently, saving time
            this_sentence = []
            self._current_file = fileinput.filename()

            for line in f:
                # skip lines to catch up to the previously stored state
                if lines_to_skip > 0:
                    lines_to_skip -= 1
                    continue

                # if line.strip() == '': continue
                if self._current_file != fileinput.filename():
                    self._current_file = fileinput.filename()
                    log(f"processing {self._current_file}")

                parse = self.segment_line(line, sep=self._sep, fmt=self._fmt)
                if self._lower:
                    parse["text"] = parse["text"].lower()
                # sentence_id is only unique within a filename, so two files containing sequential
                # sentence IDs (e.g., [1,], [1,2,3,]) will cause result in the concatenation of two
                # distinct sentences (which would be an issue since the token_ids are valid within a sentence)
                parse["sentence_id"] = f"{fileinput.filename()}_{parse['sentence_id']}"
                this_sentence += [parse]

                # have we crossed a sentence boundary? alternatively, are we out of lines to process?
                next_parse = self.segment_line(
                    f.peek(dummy_line()), sep=self._sep, fmt=self._fmt
                )  # uses dummy line if no more lines to process
                next_parse[
                    "sentence_id"
                ] = f"{fileinput.filename()}_{next_parse['sentence_id']}"
                if next_parse["sentence_id"] != parse["sentence_id"]:

                    # process current sentence and reset the "current sentence"
                    [sent] = Document([this_sentence]).sentences
                    sentence_batch += [sent]

                    if (
                        (sents_read := len(sentence_batch)) >= batch_size
                        or self._sentences_seen + sents_read >= n_sentences
                    ):

                        ################################################################
                        #### this is where the sentencebatch is processed ##############
                        ################################################################
                        self.digest_sentencebatch(sentence_batch, parallel=parallel)
                        ################################################################

                        lines_read = sum(len(s.words) for s in sentence_batch)
                        sentence_batch = []

                        self._lines_read += lines_read
                        self._sentences_seen += sents_read
                        if self._total < float("inf"):
                            T.update(sents_read)
                        else:
                            T.update(lines_read)
                        self.to_cache()

                        this_time = time.process_time()
                        log(
                            f"processed sentence_batch of size {sents_read} in {this_time-anchor_time:.3f} sec "
                            f"({sents_read/(this_time-anchor_time):.3f} sents/sec)"
                        )
                        log(
                            f"accumulated unique tokens: {len(self._token_stats):,}; "
                            f"accumulated unique pairs: {len(self._pair_stats):,}; "
                            f"sentences seen: {self._sentences_seen:,}"
                        )
                        anchor_time = this_time

                    this_sentence = []

                if self._sentences_seen >= n_sentences:
                    self.to_cache()
                    break

            log(f"finished processing after seeing {self._sentences_seen} sentences.")

    def digest_sentencebatch(
        self,
        sb: typing.List["stanza.models.common.doc.Sentence"],
        parallel: bool = True,
    ):
        """Digests a sentencebatch containing sentences by computing its token
            and pair occurrence stats and updating the instance's counter objects
            tracking the global stats for this corpus

        Args:
            sb (list): sentencebatch containing stanza Sentences
        """
        # accumulate statistics about words and word pairs in the sentence
        stats = Parallel(n_jobs=(-1 if parallel else 1))(
            delayed(self._digest_sentence)(sent) for sent in sb
        )
        for (
            token_stat,
            pair_stat,
            skip_pair_stat,
            triplet_stat_obj,
            triplet_stat_nsubj,
        ) in stats:
            self._token_stats.update(token_stat)
            self._pair_stats.update(pair_stat)
            self._skip_pair_stats.update(skip_pair_stat)
            self._triplet_stats["obj"].update(triplet_stat_obj)
            self._triplet_stats["nsubj"].update(triplet_stat_nsubj)

    @classmethod
    def _digest_sentence(
        cls, sent: "stanza.models.common.doc.Sentence", return_context=True
    ) -> typing.Tuple[Counter, Counter]:
        """'digests' a sentence into counts of tokens and token pairs in it

        Args:
            sent (stanza.models.common.doc.Sentence): input Sentence

        Returns:
            typing.Tuple[Counter, Counter]: token_stats, pair_stats for this sentence
        """
        token_stat = Counter([Word(w.text, w.upos) for w in sent.words])
        pair_stat, skip_pair_stat, triplet_stat_obj, triplet_stat_nsubj = [
            *map(Counter, cls._extract_edges_from_sentence(sent))
        ]
        # if return_context:
        #     return token_stat, pair_stat, triplet_stat, ' '.join(w.text for w in sent.words)
        return (
            token_stat,
            pair_stat,
            skip_pair_stat,
            triplet_stat_obj,
            triplet_stat_nsubj,
        )

    @classmethod
    def segment_line(cls, line: str, sep: str, fmt: typing.Iterable[str]) -> dict:
        """Reads the columns from a line corresponding to a single token in a parse.  Returns them
            as a labeled dictionary, with labels corresponding to the `fmt` list in the order of
            appearance.

        Args:
            line (str): _description_
            sep (str): _description_
            fmt (typing.Iterable[str]): _description_

        Returns:
            dict: _description_
        """
        import pydoc

        doc = {}
        row = line.strip().split(sep)
        # if fmt is specified, override self._fmt; else fall back on self._fmt
        for i, item in enumerate(fmt):
            label, typ = item.split(":")
            # if an explicit Python type is provided, cast the label to it
            typecast = pydoc.locate(typ or "str")
            try:
                value = typecast(row[i])
            except IndexError:
                log("ERR:", line, line.strip().split(sep))
                raise
            doc[label] = value
        return doc

    @classmethod
    def _extract_edges_from_sentence(
        cls,
        sentence: "stanza.models.common.doc.Sentence",
        triple=("ADJ", "NOUN", "VERB"),  # we're looking for VERB_ADJ_NOUN
    ) -> typing.Iterable[typing.Tuple[Word, Word]]:
        """Extracts all the edges in the dependecy tree of the sentence, returns them
            as tuples of `Word` (with text and upos information preserved)

        Args:
            sentence: a stanza Sentence containing a dependency parse

        Returns:
            typing.List[typing.Tuple[Word]]: Edges in the sentence
        """
        edges = []
        skip_edges = []
        triplets_obj = []
        triplets_nsubj = []
        for w in sentence.words:
            # if w is root, it has no head, so skip
            if w.head == 0:
                continue
            p = sentence.words[w.head - 1]
            edges += [(Word(w.text, w.upos), Word(p.text, p.upos))]

            # this is to additionally extract the VERB
            if p.head == 0:
                continue
            q = sentence.words[p.head - 1]
            skip_edges += [(Word(w.text, w.upos), Word(q.text, q.upos))]

            if (w.upos, p.upos, q.upos) == triple:
                if p.deprel in ("obj",):
                    triplets_obj += [
                        (
                            Word(q.text, q.upos),
                            Word(w.text, w.upos),
                            Word(p.text, p.upos),
                        )
                    ]
                if p.deprel in ("nsubj", "nsubj:pass"):
                    triplets_nsubj += [
                        (
                            Word(q.text, q.upos),
                            Word(w.text, w.upos),
                            Word(p.text, p.upos),
                        )
                    ]

        return edges, skip_edges, triplets_obj, triplets_nsubj

    ################################################################
    #### miscellaneous analysis-related stuff
    ################################################################

    def generate_graph(
        self, child_upos: str = None, parent_upos: str = None
    ) -> "networkx.Graph":
        """ """
        raise NotImplementedError
        wg = WordGraph(
            (
                (w, p)
                for w, p in self.pair_stats
                if (child_upos is None or w.upos == child_upos)
                and (parent_upos is None or p.upos == parent_upos)
            )
        )

        return wg

        # def extract_combinations(self,
        #                          child_upos, parent_upos,
        #                          collapse_by_token: bool = True,
        #                          ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        #     child_to_parent = defaultdict(int)
        #     parent_to_child = defaultdict(int)

        #     for (w,p), ct in self._pair_stats.items():
        #         if (child_upos == '*' or w.upos == child_upos) and (parent_upos == '*' or p.upos == parent_upos):
        #             child_to_parent[w] += 1 if collapse_by_token else ct
        #             parent_to_child[p] += 1 if collapse_by_token else ct

        #     return child_to_parent, parent_to_child

        child_to_parent_arr = np.array(
            sorted(child_to_parent.values(), key=lambda c: -c)
        )
        parent_to_child_arr = np.array(
            sorted(parent_to_child.values(), key=lambda c: -c)
        )

        child_to_parent_labels = np.array(
            sorted(child_to_parent.keys(), key=lambda k: -child_to_parent[k])
        )
        parent_to_child_labels = np.array(
            sorted(parent_to_child.keys(), key=lambda k: -parent_to_child[k])
        )

        return (
            child_to_parent_arr,
            child_to_parent_labels,
            parent_to_child_arr,
            parent_to_child_labels,
        )


class ChainedCorpus(Corpus):
    def __init__(
        self,
        directory_or_filelist: typing.Union[
            Path, str, typing.Iterable[typing.Union[Path, str]]
        ],
        cache_dir=None,
        cache_tag=None,
        fmt: typing.List[str] = (
            "sentence_id:int",
            "text:str",
            "lemma:str",
            "id:int",
            "head:int",
            "upos:str",
            "deprel:str",
        ),
        sep: str = "\t",
        lowercase: bool = True,
        load=True,
    ):

        self._cache_dir = cache_dir
        self._cache_tag = cache_tag
        self._directory_or_filelist = directory_or_filelist
        if load:
            self.load_cache()

    def load_cache(self, directory_or_filelist=None):
        # store Corpus objects initialized from cache files in a list
        # corpus_objs = []
        directory_or_filelist = directory_or_filelist or self._directory_or_filelist
        for path in iterable_from_directory_or_filelist(directory_or_filelist):
            c = Corpus.from_cache(path)
            # corpus_objs += [c]
            # we want to aggregate statistics of subparts using their cached objects
            for attr, typ in self._attrs_to_cache:
                if typ in (int, Counter):
                    # choose the appropriate add function according to class
                    redfn = getattr(typ, "__add__")
                    # construct the reduced (aggregated) object, an int or a Counter, to assign to self
                    reduce_operand = [getattr(c, attr)]
                    if hasattr(self, attr) and type(getattr(self, attr)) == typ:
                        reduce_operand += [getattr(self, attr)]
                    ob = reduce(redfn, reduce_operand)
                    setattr(self, attr, ob)

            # close connection to SQLite after done loading
            c._close_cache()

            if self._cache_dir is not None and self._cache_tag is not None:
                log(f"caching {self}")
                self.to_cache(cache_dir=self._cache_dir, cache_tag=self._cache_tag)

    def _get_cache(self, cache_dir, cache_tag):
        return super()._get_cache(cache_dir, cache_tag)

    # def load_cache(self, allow_empty=True):
    #     raise NotImplementedError
    def to_cache(self, cache_dir=None, cache_tag=None):

        if cache_dir is not None:
            self._cache_dir = cache_dir
        if cache_tag is not None:
            self._cache_tag = cache_tag

        if self.cache is None:
            self.cache = self._get_cache(self._cache_dir, self._cache_tag)
        super().to_cache()
