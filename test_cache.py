
from src.corpus import Corpus

with Corpus('./COCA', 
            cache_dir='./cache', #cache_tag='coca_inf', 
            n_sentences=float('inf')) as coca:
    coca.read(batch_size=5_000)

