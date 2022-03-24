
from corpus import Corpus

c = Corpus('COCA', cache_dir='./cache', tag='coca_inf', n_sentences=float('inf'))
# _ = list(c.read()) # initial run-through for constructing a cache
_ = list(c.extract_edges()) # for computing pairwise occurrence stats
