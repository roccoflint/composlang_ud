    
import numpy as np
from matplotlib import pyplot as plt

def fit_zipf(freq):
    
    from scipy.stats import linregress
    
    logfreq = np.log(np.sort(freq)[::-1])
    logrank = np.log(np.arange(1, len(freq)+1, 1))
    
    res = linregress(logrank, logfreq)
    return np.exp(res.intercept + res.slope*logrank), res
    
    
    
def plot_rank_freq_distribution(ranked_freq, labels, upos):

    fig, (left, right) = plt.subplots(1,2, figsize=(20,5))

    def plot_comparisons(candidate, axes):
        
        fitted, res = fit_zipf(candidate)
        axes.plot(fitted, '-', label=f'fitted using linregress\n$r={res.rvalue:.2f},p={res.pvalue:.2f}$')
        # axes.fill_between(np.arange(1, len(fitted)+1),
        #                   np.exp(np.log(fitted)-res.stderr), np.exp(np.log(fitted)+res.stderr),
        #                   color='gray', alpha=0.2)
        np.random.seed(1)
        #### ZIPFs for comparison
        for a in np.arange(1.5, 3, .3):
            z = np.sort(np.random.zipf(a, (len(candidate,))))[::-1]
            # z = z[z <= max(candidate)+10]
            axes.plot(z, '--', label=f'draw from Zipf (a={a:.2f})')


    ####################
    #### LEFT
    ####################
    left.plot(ranked_freq, 'b.' if upos in ('NOUN', 'VERB') else 'r.', label=f'{upos} compos.', 
                 linewidth=5, alpha=.7, )

    plot_comparisons(ranked_freq, left)

    left.set(xlabel=f'{upos} compositionality rank',
                ylabel=f'# lexical that combine with {upos}',
                yscale='log')
    left.set_ylim([.9, max(ranked_freq)+10])
    
    ulim = np.ceil(np.log(ranked_freq[0])/np.log(2))+1
    ulim = int(ulim)
    left.set_yticks(2**np.arange(ulim), 2**np.arange(ulim))

    xticks = [*np.arange(1, len(ranked_freq), len(ranked_freq)//10)]
    # log(xticks)
    left.set_xticks(xticks,
                    labels=[f'{a}\n{b:.0e}' for a,b in zip(labels[xticks], xticks)],
                    rotation=60)
    left.legend()


    ####################
    #### RIGHT
    ####################
    right.plot(ranked_freq, 'b.' if upos in ('NOUN', 'VERB') else 'r.',  label=f'{upos} compos.', 
                 linewidth=5, alpha=.7,  )

    plot_comparisons(ranked_freq, right)

    right.set(xlabel=f'{upos} compositionality rank\n(log scale)',
                ylabel=f'# lexical items that combine with {upos}\n(log scale)',
                xscale='log', yscale='log')
    right.set_ylim([.9, max(ranked_freq)+10])

    ulim = np.ceil(np.log(ranked_freq[0])/np.log(2))+1
    ulim = int(ulim)
    right.set_yticks(2**np.arange(ulim), 2**np.arange(ulim))
    
    xticks = [*2**np.arange(ulim)] + [*np.arange(2**ulim, len(ranked_freq), (len(ranked_freq)-2**ulim)//3)]
    # log(xticks)
    right.set_xticks(xticks,
                     labels=[f'{a}\n{b:.0e}' for a,b in zip(labels[xticks], xticks)],
                     rotation=60)
    right.legend()

    return fig, (left, right)