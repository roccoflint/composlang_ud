
import IPython
from corpus import Corpus

c = Corpus('COCA', n_sentences=1e5)
# _ = list(c.read()) # initial run-through for pruning behavior
_ = list(c.extract_edges()) # for computing pairwise occurrence stats

c.to_cache(prefix='./cache', tag='coca_1e5')
