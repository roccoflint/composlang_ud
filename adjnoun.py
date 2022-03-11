# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.9.5 ('composlang-iD_d0IlX-py3.9')
#     language: python
#     name: python395jvsc74a57bd04712f51a2b6565feb4b6c838cc733f8f7925e6eb21082244c1212b918115d3b9
# ---

# %% [markdown]
# # Compos-Lang
# ## How compositional is language?

# %%
from pathlib import Path
from collections import defaultdict, Counter

from matplotlib import pyplot as plt
import numpy as np

from more_itertools import peekable
from tqdm.notebook import tqdm

# %%
# %load_ext autoreload
# %autoreload 2
from corpus import Corpus

# %% [markdown]
# ## Load an example COCA file 

# %%
coca = Corpus('COCA')

# %%
sents = [*coca.read(10_000)]

# %%
len(sents)

# %% [markdown]
# ## Look at some token pairs of interest (ADJ, NOUN)

# %%
for s in tqdm(sents[:10]):
    for a, n, s.text in get_pairs(s, 'ADJ', 'NOUN'):
        print(f'{a.text:<16} {n.text:<16}')
    if s.text:
        print(f'\t{s.text}')
        print('-'*79)

# %% [markdown]
# ## Now, let's collect all pairs in a mapping:
#
# - noun -> set of adjectives that modify it
# - adj -> set of nouns that modify it

# %%
noun_to_adj = defaultdict(set)
adj_to_noun = defaultdict(set)

for s in tqdm(sents):
    for a, n, s.text in get_pairs(s, 'ADJ', 'NOUN'):
        noun_to_adj[n.text].add((a.text, s.text))
        adj_to_noun[a.text].add((n.text, s.text))


# %% [markdown]
# ## The structure above stores the unique contexts of each as well, but we want to condense this information into counts

# %%
def count_mappings(d):
    '''
    returns the total mappings 
    (including re-occurrence in multiple contexts)
    as well as the total unique mappings 
    '''
    num_mappings = defaultdict(int)
    num_unique_mappings = defaultdict(int)
    
    for key in d:
        unique_mappings = {val for val, context in d[key]}
        num_unique_mappings[key] = len(unique_mappings)
        num_mappings[key] = len(d[key])
    
    return dict(total=num_mappings, unique=num_unique_mappings)


# %%
unique_noun_to_adj = count_mappings(noun_to_adj)['unique']
total_noun_to_adj = count_mappings(noun_to_adj)['total']

unique_adj_to_noun = count_mappings(adj_to_noun)['unique']
total_adj_to_noun = count_mappings(adj_to_noun)['total']

# %% tags=[]
noun_combos = np.array(sorted(unique_noun_to_adj.values(), key=lambda t:-t))
adj_combos = np.array(sorted(unique_adj_to_noun.values(), key=lambda t:-t))

noun_combos_labels = np.array(sorted(unique_noun_to_adj.keys(), key=lambda t:-unique_noun_to_adj[t]))
adj_combos_labels = np.array(sorted(unique_adj_to_noun.keys(), key=lambda t:-unique_adj_to_noun[t]))


# %%
def plot_from_combo_map(noun_combos, adj_combos, 
                        noun_combos_labels, adj_combos_labels):

    fig, ax = plt.subplots(2,2, figsize=(18,10))

    def plot_comparisons(candidate, axes):
        np.random.seed(1)
        #### ZIPFs for comparison
        for a in np.arange(1.5, 2.5, .1):
            z = np.sort(np.random.zipf(a, (len(candidate,))))[::-1]
            # z = z[z <= max(candidate)+10]
            axes.plot(z, '--', label=f'Zipf (a={a:.2f})')
        # #### EXP for comparison
        # alpha = 10+np.median(candidate)/np.log(2)
        # exp = np.sort(1+np.random.exponential(alpha, (len(candidate),)))[::-1]
        # axes.plot(exp, '--', label=f'Exp({alpha:.2f})')



    ####################
    #### TOP LEFT
    ####################
    ax[0,0].plot(noun_combos, 'b.', label='NOUN compos.', 
                 linewidth=5, alpha=.7, )

    plot_comparisons(noun_combos, ax[0,0])

    ax[0,0].set(xlabel='NOUN compositionality rank',
                ylabel='#ADJs that combine with NOUN',
                yscale='log')
    ax[0,0].set_ylim([.9, max(noun_combos)+10])
    
    ulim = np.ceil(np.log(noun_combos[0])/np.log(2))+1
    ax[0,0].set_yticks(2**np.arange(ulim), 2**np.arange(ulim))

    noun_xticks = [*np.arange(1, len(noun_combos), 1_000)]
    ax[0,0].set_xticks(noun_xticks,
                       labels=[f'{a}\n{b:.0e}' for a,b in zip(noun_combos_labels[noun_xticks], noun_xticks)],
                       rotation=60)
    ax[0,0].legend()


    ####################
    #### TOP RIGHT
    ####################
    ax[0,1].plot(noun_combos, 'b.',  label='NOUN compos.', 
                 linewidth=5, alpha=.7,  )

    plot_comparisons(noun_combos, ax[0,1])

    ax[0,1].set(xlabel='NOUN compositionality rank\n(log scale)',
                ylabel='#ADJs that combine with NOUN\n(log scale)',
                xscale='log', yscale='log')
    ax[0,1].set_ylim([.9, max(noun_combos)+10])

    ulim = np.ceil(np.log(noun_combos[0])/np.log(2))+1
    ax[0,1].set_yticks(2**np.arange(ulim), 2**np.arange(ulim))
    log_noun_xticks = [*2**np.arange(8)] + [*np.arange(2**8, len(noun_combos), 2_300)]
    ax[0,1].set_xticks(log_noun_xticks,
                       labels=[f'{a}\n{b:.0e}' for a,b in zip(noun_combos_labels[log_noun_xticks], log_noun_xticks)],
                       rotation=60)


    ax[0,1].legend()



    ####################
    #### BOTTOM LEFT
    ####################
    ax[1,0].plot(adj_combos, 'r-', label='ADJ compos.',
                 linewidth=5, alpha=.7,)
    ax[1,0].set_yticks(np.arange(0, max(adj_combos)+10, max(adj_combos)//10))
    ax[1,0].set(xlabel='ADJ compositionality rank',
                ylabel='#NOUNs that combine with ADJ')

    plot_comparisons(adj_combos, ax[1,0])
    ax[1,0].legend()



    ####################
    #### BOTTOM RIGHT
    ####################
    ax[1,1].plot(adj_combos, 'r-', label='ADJ compos.',
                 linewidth=5, alpha=.7,)
    ax[1,1].set(xlabel='ADJ compositionality rank\n(log scale)',
                ylabel='#NOUNs that combine with ADJ\n(log scale)',
                xscale='log', yscale='log')

    plot_comparisons(adj_combos, ax[1,1])
    ax[1,1].legend()

    return fig, ax

# %%
f,a = plot_from_combo_map(noun_combos, 
                          adj_combos, 
                          noun_combos_labels, 
                          adj_combos_labels)

plt.tight_layout()
plt.show()
