import fileinput
import pydoc
import time
import typing
from collections import Counter, defaultdict
from functools import reduce, cached_property
from pathlib import Path

import numpy as np
import stanza
# from diskcache import Cache
from joblib import Parallel, delayed
from more_itertools import peekable
from sqlitedict import SqliteDict
from tqdm import tqdm

from composlang.graph import WordGraph
from composlang.utils import iterable_from_directory_or_filelist, log, pathify
from composlang.word import Word


class Corpus:
    '''
    A class to read and process data from a corpus in one place;
    accumulate aggregate statistics, etc.
    Expects a dependency-parsed corpus with one token per line.
    Flexible formats are supported as long as they are properly specified.
    The only restriction is that a `sentence_id` column is required in order
    to detect sentence boundaries.
    '''

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
    _token_stats = None # (token:str, upos:str) -> num_occ:int
    _pair_stats = None # ((token1:str, upos1:str), (token2, upos2)) -> num_occ:int
    # in a child-parent relation in a dependency parse
    cache = None

    def __init__(self,
                 directory_or_filelist: typing.Union[Path, str,
                                                     typing.Iterable[typing.Union[Path, str]]],

                 cache_dir: typing.Union[str, Path] = './cache/composlang',
                 cache_tag: str = None,

                 n_sentences: int = None, 
                 fmt: typing.List[str] = ('sentence_id:int', 'text:str', 'lemma:str',
                                          'id:int', 'head:int', 'upos:str', 'deprel:str'),
                 sep: str = '\t', lowercase: bool = True
                 ):
        ''' '''
        self._fmt = fmt
        self._sep = sep
        self._lower = lowercase
        self._files = sorted(iterable_from_directory_or_filelist(directory_or_filelist))
        if self._files is None:
            log(f'could not find files at {directory_or_filelist}')
            # raise ValueError(f'could not find files at {directory_or_filelist}')

        self._pair_stats = Counter() # (token, token) -> num_occurrences
        self._token_stats = Counter() # token -> num_occurrences

        # loads the pre-existing _pair_stats and _token_stats objects, or creates empty ones
        self._cache_dir = cache_dir
        self._cache_tag = cache_tag
        self.cache = self._get_cache()
        self.load_cache()
        self._n_sentences = n_sentences or float('inf') # upper limit for no. of sentences to process


    @classmethod
    def from_cache(cls, cache_file: typing.Union[str, Path], 
                   n_sentences=None, fmt=None, sep=None, lowercase=None):
        path = pathify(cache_file)
        cache_dir = path.parent
        cache_tag = path.parts[-1]
        return cls('/dev/null', cache_dir, cache_tag, n_sentences, fmt, sep, lowercase)


    def __len__(self) -> int:
        return self._sentences_seen

    # use context manager to handle closing of cache
    def __enter__(self):
        return self # nothing to do here

    def __exit__(self, *args, **kws):
        self._close_cache()


    ################################################################ 
    #### accessing data of the class
    ################################################################

    @cached_property
    def token_stats(self):
        return Counter(self._token_stats)

    @cached_property
    def pair_stats(self):
        return Counter(self._pair_stats)

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
            d[upos] += ct
        return d

    def get_upos_pairs(self, child_upos, parent_upos) -> typing.Tuple[Word, Word, int]:
        '''
        Constructs a generator over all pairs of tokens that match the given child_upos and
        parent_upos values

        Args:
            child_upos (str): one of the several UPOSes in the corpus (e.g., NOUN, ADJ).
                the "child" token (the token appearing as a child of the parent in the
                dependency parse) will have to match this UPOS
            parent_upos (str): see `child_upos`
        '''
        for w, p in self.pair_stats:
            if w.upos == child_upos and p.upos == parent_upos:
                yield w, p


    ################################################################ 
    #### caching behavior
    ################################################################ 

    def _close_cache(self):
        try:
            self.cache.commit()
            self.cache.close()
        except Exception as e:
            log(f'encountered error in closing cache: {e}')
            pass

    def _get_cache(self, prefix=None, tag=None):
        '''
        creates and returns an SqliteDict() object
        '''
        tag = tag or self._cache_tag
        if tag is None:
            from hashlib import shake_128
            s = shake_128()
            s.update(bytes(' '.join(map(str, self._files)), 'utf-8'))
            self._cache_tag = tag = s.hexdigest(5)

        root = Path(prefix or self._cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        root /= tag
        return SqliteDict(str(root), flag='c')

    _attrs_to_cache = (('_sentences_seen', int), ('_lines_read', int),
                      ('_current_file', lambda: None), ('_pair_stats', Counter),
                      ('_token_stats', Counter), ('_files', lambda: None),
                      ('_total', lambda: None), ('_n_sentences', lambda: None))

    def load_cache(self, allow_empty=True):
        '''
        recover core data of this instance from cache 
        '''
        log(f'attempt loading from cache at {self._cache_dir}/{self._cache_tag}')
        start = time.process_time()
        for attr, default in self._attrs_to_cache:
            try:
                obj = self.cache[attr]
                setattr(self, attr, obj)
            except KeyError as e:
                log(f'could not find attribute {attr} in cache')
                if not allow_empty:
                    raise e
        end = time.process_time()
        log(f'successfully loaded cached data from {self._cache_dir}/{self._cache_tag} in {end-start:.3f} seconds')

    def to_cache(self):
        '''
        dump critical state data of this instance to cache
        '''
        log(f'caching to {self._cache_dir}/{self._cache_tag}')
        start = time.process_time()
        for attr, _ in self._attrs_to_cache:
            obj = getattr(self, attr)
            self.cache[attr] = obj

        self.cache.commit()
        end = time.process_time()
        log(f'successfully cached to {self._cache_dir}/{self._cache_tag} in {end-start:.3f} seconds')


    ################################################################ 
    #### read corpus
    ################################################################ 

    def read(self, batch_size: int = 5_000):
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
        
        dummy_line = lambda: f'{"__EOF__"}{self._sep}' + self._sep.join(map(str, range(6 + 10)))
        # if we are resuming from previous state, we want to skip lines that are already processed.
        lines_to_skip = self._lines_read
        n_sentences = self._n_sentences # upper limit

        # sum the total lines in a file for tqdm without loading them all in memory
        if self._total is None:
            if n_sentences >= float('inf'):
                with fileinput.input(files=self._files) as f:
                    self._total = sum(1 for _ in tqdm(f, desc=f'counting # of lines in corpus'))
                log(f'Preparing to read {self._total} lines')
            else:
                self._total = n_sentences
                log(f'Preparing to read {self._total} sentences')

        anchor_time = time.process_time()
        with fileinput.input(files=self._files) as f, tqdm(total=self._total, leave=False) as T:
            # more_itertools.peekable allows seeing future context without using up item from iterator
            f = peekable(f)

            if n_sentences >= float('inf'): # process all sentences; progressbar tracks # of lines
                T.update(self._lines_read)
            else: # process a predetermiend # of sentences; also reflected in progressbar
                T.update(self._sentences_seen)

            sentence_batch = [] # accumulate parsed sentences to process concurrently, saving time
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
                    log(f'processing {self._current_file}')

                parse = self.segment_line(line, sep=self._sep, fmt=self._fmt)
                # sentence_id is only unique within a filename, so two files containing sequential
                # sentence IDs (e.g., [1,], [1,2,3,]) will cause result in the concatenation of two
                # distinct sentences (which would be an issue since the token_ids are valid within a sentence)
                parse['sentence_id'] = f"{fileinput.filename()}_{parse['sentence_id']}"
                this_sentence += [parse]

                # have we crossed a sentence boundary? alternatively, are we out of lines to process?
                next_parse = self.segment_line(f.peek(dummy_line()), sep=self._sep, fmt=self._fmt) # uses dummy line if no more lines to process
                next_parse['sentence_id'] = f"{fileinput.filename()}_{next_parse['sentence_id']}"
                if next_parse['sentence_id'] != parse['sentence_id']:

                    # process current sentence and reset the "current sentence"
                    [sent] = Document([this_sentence]).sentences
                    sentence_batch += [sent]

                    if (sents_read := len(sentence_batch)) >= batch_size or self._sentences_seen+sents_read >= n_sentences:

                        self.digest_sentencebatch(sentence_batch)
                        lines_read = sum(len(s.words) for s in sentence_batch)
                        sentence_batch = []

                        self._lines_read += lines_read
                        self._sentences_seen += sents_read
                        if self._total < float('inf'):
                            T.update(sents_read)
                        else:
                            T.update(lines_read)
                        self.to_cache()

                        this_time = time.process_time()
                        log(f'processed sentence_batch of size {sents_read} in {this_time-anchor_time:.3f} sec '
                            f'({sents_read/(this_time-anchor_time):.3f} sents/sec)')
                        log(f'accumulated unique tokens: {len(self._token_stats):,}; '
                            f'accumulated unique pairs: {len(self._pair_stats):,}; '
                            f'sentences seen: {self._sentences_seen:,}')
                        anchor_time = this_time

                    this_sentence = []


                if self._sentences_seen >= n_sentences:
                    self.to_cache()
                    break

            log(f'finished processing after seeing {self._sentences_seen} sentences.')


    def digest_sentencebatch(self, sb: typing.List['stanza.models.common.doc.Sentence']):
        """Digests a sentencebatch containing sentences by computing its token
            and pair occurrence stats and updating the instance's counter objects
            tracking the global stats for this corpus

        Args:
            sb (list): sentencebatch containing stanza Sentences
        """        
        # accumulate statistics about words and word pairs in the sentence
        stats = Parallel(n_jobs=-2)(delayed(self._digest_sentence)(sent) for sent in sb)
        for token_stat, pair_stat in stats:
            self._token_stats.update(token_stat)
            self._pair_stats.update(pair_stat)

    @classmethod
    def _digest_sentence(cls, sent: 'stanza.models.common.doc.Sentence') -> typing.Tuple[Counter, Counter]:
        """'digests' a sentence into counts of tokens and token pairs in it

        Args:
            sent (stanza.models.common.doc.Sentence): input Sentence

        Returns:
            typing.Tuple[Counter, Counter]: token_stats, pair_stats for this sentence
        """        
        ts = Counter([Word(w.text, w.upos) for w in sent.words])
        ps = Counter(cls._extract_edges_from_sentence(sent))
        return ts, ps

    @classmethod
    def segment_line(cls, line: str, sep:str, fmt:typing.Iterable[str]) -> dict:
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
        doc = {}
        row = line.strip().split(sep)
        # if fmt is specified, override self._fmt; else fall back on self._fmt
        for i, item in enumerate(fmt):
            label, typ = item.split(':')
            # if an explicit Python type is provided, cast the label to it
            typecast = pydoc.locate(typ or 'str')
            try:
                value = typecast(row[i])
            except IndexError as e:
                log('ERR:', line, line.strip().split(sep))
                raise
            doc[label] = value
        return doc


    @classmethod
    def _extract_edges_from_sentence(cls,
                                     sentence: 'stanza.models.common.doc.Sentence') -> \
                                     typing.Iterable[typing.Tuple[Word, Word]]:
        """Extracts all the edges in the dependecy tree of the sentence, returns them
            as tuples of `Word` (with text and upos information preserved)

        Args:
            sentence: a stanza Sentence containing a dependency parse

        Returns:
            typing.List[typing.Tuple[Word]]: Edges in the sentence
        """                                     
        edges = []
        for w in sentence.words:
            # if w is root, it has no head, so skip
            if w.head == 0:
                continue
            p = sentence.words[w.head-1]
            edges += [(Word(w.text, w.upos), Word(p.text, p.upos))]

        return edges

 
    ################################################################ 
    #### miscellaneous analysis-related stuff
    ################################################################ 

    def generate_graph(self, child_upos: str = None, 
                       parent_upos: str = None) -> "networkx.Graph":
        '''
        '''
        wg = WordGraph(((w,p) for w,p in self.extract_edges()
                        if (child_upos is None or w.upos == child_upos) and \
                           (parent_upos is None or p.upos == parent_upos)))

        return wg


    def extract_combinations(self,
                             child_upos, parent_upos,
                             unique: bool = True,
                             ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        child_to_parent = defaultdict(int)
        parent_to_child = defaultdict(int)

        for (w,p), ct in self._pair_stats.items():
            if (child_upos == '*' or w.upos == child_upos) and (parent_upos == '*' or p.upos == parent_upos):
                child_to_parent[w] += 1 if unique else ct
                parent_to_child[p] += 1 if unique else ct

        child_to_parent_arr = np.array(sorted(child_to_parent.values(), key=lambda c:-c))
        parent_to_child_arr = np.array(sorted(parent_to_child.values(), key=lambda c:-c))

        child_to_parent_labels = np.array(sorted(child_to_parent.keys(), key=lambda k:-child_to_parent[k]))
        parent_to_child_labels = np.array(sorted(parent_to_child.keys(), key=lambda k:-parent_to_child[k]))

        return child_to_parent_arr, child_to_parent_labels, parent_to_child_arr, parent_to_child_labels



class ChainedCorpus(Corpus):

    def __init__(self,
                 directory_or_filelist: typing.Union[
                     Path, str, typing.Iterable[typing.Union[Path, str]]],
                 cache_dir=None, cache_tag=None,
                 fmt: typing.List[str] = ('sentence_id:int', 'text:str',
                                          'lemma:str', 'id:int', 'head:int',
                                          'upos:str', 'deprel:str'),
                 sep: str = '\t', lowercase: bool = True):

        # store Corpus objects initialized from cache files in a list
        corpus_objs = []
        for path in iterable_from_directory_or_filelist(directory_or_filelist):
            c = Corpus.from_cache(path)
            corpus_objs += [c]

        # we want to aggregate statistics of subparts using their cached objects
        for attr, typ in self._attrs_to_cache:
            if typ in (int, Counter):
                # choose the appropriate add function according to class
                redfn = getattr(typ, '__add__')
                # construct the reduced (aggregated) object, an int or a Counter, to assign to self
                ob = reduce(redfn, [getattr(c, attr) for c in corpus_objs])
                setattr(self, attr, ob)

        # close connection to SQLite after done loading
        for c in corpus_objs:
            c._close_cache()


    def _get_cache(self, cache_dir, cache_tag):
        return super()._get_cache(cache_dir, cache_tag)
    def load_cache(self, allow_empty=True):
        raise NotImplementedError
    def to_cache(self, cache_dir, cache_tag):
        if self.cache is None:
            self.cache = self._get_cache(cache_dir, cache_tag)
        super().to_cache()