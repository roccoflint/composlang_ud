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
#     display_name: composlang
#     language: python
#     name: composlang
# ---

# %% [markdown]
# # Compos-Lang
# ## How compositional is language?

# %% tags=[]
from pathlib import Path
from collections import defaultdict, Counter

from matplotlib import pyplot as plt
import numpy as np

from more_itertools import peekable
from tqdm.notebook import tqdm

# %%
# %load_ext autoreload
# %autoreload 2
from corpus import Corpus, plot_rank_freq_distribution

# %% [markdown]
# ## Initialize a Corpus object to read COCA 

# %%
coca = Corpus('COCA_explore', n_sentences=500, store=False)
child = 'ADJ'
parent = 'NOUN'
len(coca)

# %% [markdown]
# ## Look at some summary stats

# %%
combos_obs = len(list(coca.extract_upos_pairs(child, parent)))

upos_counts = coca.upos_counts(unique=False)
print(f'Total {child} occurrences observed: {upos_counts[child]}\n'
      f'Total {parent} occurrences observed: {upos_counts[parent]}')

upos_counts = coca.upos_counts(unique=True)
print(f'Unique {child} instances observed: {upos_counts[child]}\n'
      f'Unique {parent} instances observed: {upos_counts[parent]}')
combos_possible = (upos_counts[child] * upos_counts[parent])

print(f'Total {child}-{parent} possible: {combos_possible:.1e}. Total {child}-{parent} combos observed: {combos_obs:.1e}'
      f'\nRealized fraction: {combos_obs/combos_possible:.2e}')

# %% tags=[]
from pyvis.network import Network
# # import networkx as nx

wg = coca.generate_graph([child, parent])
nt = Network(notebook=True)
# # # populates the nodes and edges data structures
nt.from_nx(wg.g)
# # pos = nx.circular_layout(wg.g)

nt.show('nx.html')
# # nx.draw_circular(wg.g)

# %%
i = 0
for a, n, *s in coca.extract_upos_pairs(child, parent, include_context=True):
    i += 1
    print(f'{a.text:<16} {n.text:<16}')
    if s: print(f'\t{" ".join(w.text for w in s[0])}')
    print('-'*79)
    if i >= 7: break

# %% tags=[]
child_combos, child_labels, parent_combos, parent_labels = coca.extract_combinations(child, parent, threshold=0)
child_combos_any, child_labels_any, _, _ = coca.extract_combinations(child, '*', threshold=0)
_, _, parent_combos_any, parent_labels_any = coca.extract_combinations('*', parent, threshold=0)

print(f"# {parent} appearing in combination as {child}-{parent}: {len(parent_combos)}, "
      f"# {parent} appearing in total: {coca.upos_counts(unique=True)[parent]}, ratio: "
      f"{len(parent_combos)/coca.upos_counts(unique=True)[parent]:.2f}")
print(f"# {parent} appearing in combination as {'*'}-{parent}: {len(parent_combos_any)}, "
      f"# {parent} appearing in total: {coca.upos_counts(unique=True)[parent]}, ratio: "
      f"{len(parent_combos_any)/coca.upos_counts(unique=True)[parent]:.2f}")

print()

print(f"# {child} appearing in combination as {child}-{parent}: {len(child_combos)}, "
      f"# {child} appearing in total: {coca.upos_counts(unique=True)[child]}, ratio: "
      f"{len(child_combos)/coca.upos_counts(unique=True)[child]:.2f}")
print(f"# {child} appearing in combination as {child}-{'*'}: {len(child_combos_any)}, "
      f"# {child} appearing in total: {coca.upos_counts(unique=True)[child]}, ratio: "
      f"{len(child_combos_any)/coca.upos_counts(unique=True)[child]:.2f}")

# %%
f_parent, (left_parent, right_parent) = plot_rank_freq_distribution(parent_combos, parent_labels, parent)
f_child, (left_child, right_child) = plot_rank_freq_distribution(child_combos, child_labels, child)

# plt.tight_layout()
plt.show()
