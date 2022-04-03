
from composlang.corpus import Corpus
import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser('composlang')
    p.add_argument('--path', help='Path to directory storing corpus files, or path to a text file',
                   required=True)
    p.add_argument('--tag', help='tag to use for caching. by default, a hash of the input filename is used. '
                                 'if the input is a directory containing multiple files, then the hash of '
                                 'the concatenation of these filenames is used', 
                   default=None, type=str, required=False)
    p.add_argument('--cache_dir', help='directory to use for caching. by default, $PWD/cache is used.', 
                   default='./cache', type=str, required=False)
    p.add_argument('--batch_size', help='batch process sentences with parallelized stats computation. '
                                        'default: 5000',
                   type=int, default=5_000, required=False)

    args = p.parse_args()
    print(args)

    with Corpus(args.path, cache_dir=args.cache_dir, cache_tag=args.tag, 
                n_sentences=float('inf')) as coca:
        coca.read(batch_size=args.batch_size)
