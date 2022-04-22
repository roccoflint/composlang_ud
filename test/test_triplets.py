
from composlang.corpus import Corpus

with Corpus('COCA', cache_tag='test.db', n_sentences=1e3) as c:
    c.read(batch_size=100, parallel=False)

print(c.triplet_stats)
