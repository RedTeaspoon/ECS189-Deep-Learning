import sys, os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from local_code.stage_4_code.Dataset_JokeLoader import Dataset_JokeLoader
from local_code.stage_4_code.Method_RNNGeneration import Method_RNNGeneration
from local_code.stage_4_code.Result_Saver_Generation import Result_Saver_Generation
from local_code.stage_4_code.Setting_Run_RNNGeneration  import Setting_Run_RNNGeneration
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
#import pandas as pd
#-----------Choosing Device---------------
if torch.cuda.is_available():
    device = torch.device('cuda')
#elif torch.backends.mps.is_available():
    #device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device:", device)
#-----------------------------------------
#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- Dataset loader ----
    loader = Dataset_JokeLoader(
        dName='JokeLoader',
        dDescription='Load jokes for generation',
        dataset_root=os.path.join('text_generation','data')
    )
    loader.dataset_source_folder_path = os.path.join(
        grandparent_dir, 'data', 'stage_4_data'
    ) + os.sep

    # ---- Build vocabulary inline ----
    data_all = loader.load()['all']
    #-------------------------------------
    # raw_jokes: List[str], token_jokes: List[List[str]]
    raw_jokes, token_jokes = data_all['raw'], data_all['tokens']

    # Build a mapping from 3-word prefixes to the original joke
    prefix3_to_joke = {}
    for raw, tokens in zip(raw_jokes, token_jokes):
        if len(tokens) >= 3:
            key = tuple(tokens[:3])
            # If multiple jokes share the same prefix, you could store a list; here we keep the first
            prefix3_to_joke.setdefault(key, raw)
    # -------------------------------------
    tokens_list = data_all['tokens']
    counter = {}
    for tokens in tokens_list:
        for w in tokens:
            counter[w] = counter.get(w, 0) + 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for w in counter:
        vocab[w] = len(vocab)

    # ---- Model initialization ----
    model = Method_RNNGeneration(
        mName='RNNGeneration',
        mDescription='RNN text generator for jokes',
        vocab=vocab,
    )
    model.to(device)

    # ---- Pipeline components ----
    setting = Setting_Run_RNNGeneration('GenSetting', 'Train & sample generator')
    evaluator = Evaluate_Accuracy('Eval', 'Dummy eval')
    saver = Result_Saver_Generation('Saver', 'Save generation results')
    saver.result_destination_folder_path = os.path.join(
        grandparent_dir, 'result', 'stage_4_result'
    ) + os.sep
    saver.result_destination_file_name = 'joke_generation'

    # wire modules & run
    setting.prepare(loader, model, saver, evaluator)
    setting.print_setup_summary()
    setting.load_run_save_evaluate()

    import pickle

    # Path to the saved results
    result_file = os.path.join(
        grandparent_dir, 'result', 'stage_4_result', 'joke_generation.pkl'
    )

    # Load and display
    with open(result_file, 'rb') as f:
        res = pickle.load(f)

    print('\n=== Generated Samples ===')
    for seed, gen in res.get('samples', {}).items():
        print(f"Seed [{seed}] → {gen}")

    print('Generation finished.')
    print('************ Finish ************')
    # ------------------------------------------------------
    import pickle

    # load results
    res_file = os.path.join(grandparent_dir, 'result', 'stage_4_result', 'joke_generation.pkl')
    with open(res_file, 'rb') as f:
        res = pickle.load(f)

    # build prefix lookup
    data_all = loader.load()['all']
    raw_jokes, token_jokes = data_all['raw'], data_all['tokens']
    prefix3 = {}
    for raw, toks in zip(raw_jokes, token_jokes):
        if len(toks) >= 3:
            prefix3.setdefault(tuple(toks[:3]), raw)

    # seeds to compare
    data_all = loader.load()['all']
    raw_jokes = data_all['raw']
    token_jokes = data_all['tokens']

    prefix3_to_joke = {}
    prefix5_to_joke = {}
    for raw, toks in zip(raw_jokes, token_jokes):
        if len(toks) >= 3:
            key3 = tuple(toks[:3])
            prefix3_to_joke.setdefault(key3, raw)
        if len(toks) >= 5:
            key5 = tuple(toks[:5])
            prefix5_to_joke.setdefault(key5, raw)

    # --- Define your seeds ---
    three_word_seeds = [
        ('what', 'did', 'the'),
        ('why', 'did', 'the'),
        ('once', 'upon', 'a'),
        ('i', 'asked', 'the')
    ]
    five_word_seeds = [
        ('what', 'did', 'the', 'bartender', 'say'),
        ('why', 'did', 'the', 'chicken', 'cross'),
        ('once', 'upon', 'a', 'time', 'there'),
        ('i', 'asked', 'the', 'librarian', 'if')
    ]

    # --- 3-Word Seed Comparison ---
    print("\n=== 3-Word Seed Comparison ===")
    for seed in three_word_seeds:
        seed_str = ' '.join(seed)
        generated = model.generate(list(seed), max_gen_len=50)
        truth = prefix3_to_joke.get(seed, "(no exact match)")
        print(f"\nSeed (3-word): \"{seed_str}\"")
        print(f"  → Generated   : {generated}")
        print(f"  → Ground Truth: {truth}")

    # --- 5-Word Seed Comparison ---
    print("\n=== 5-Word Seed Comparison ===")
    for seed in five_word_seeds:
        seed_str = ' '.join(seed)
        generated = model.generate(list(seed), max_gen_len=50)
        truth = prefix5_to_joke.get(seed, "(no exact match)")
        print(f"\nSeed (5-word): \"{seed_str}\"")
        print(f"  → Generated   : {generated}")
        print(f"  → Ground Truth: {truth}")

# ────────────────────────────────────────────────────────────────────────────────


# 2) Trigram overlap (3-word)
def ngram_set(tokens, n):
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

print("\n3-Word Trigram Overlap:")
for seed in three_word_seeds:
    gen_tokens = model.generate(list(seed), max_gen_len=50).split()
    truth      = prefix3_to_joke.get(tuple(seed), "").lower().split()
    if len(truth) < 3:
        print(f"  Seed {' '.join(seed)} → (no ground-truth)")
        continue
    truth_tris = ngram_set(truth, 3)
    gen_tris   = ngram_set(gen_tokens, 3)
    overlap    = len(truth_tris & gen_tris) / len(truth_tris)
    print(f"  Seed {' '.join(seed)} → {overlap:.0%} overlap")

# 3) Teacher-forced next-token accuracy (3-word)
correct, total = 0, 0
for seed in three_word_seeds:
    truth = prefix3_to_joke.get(tuple(seed), "")
    tokens = truth.lower().split()
    hidden = None
    for t in range(len(tokens)-1):
        inp = torch.tensor([[vocab.get(tokens[t],1)]], device=device)
        logits, hidden = model(inp, hidden)
        pred = logits.argmax(-1).item()
        if pred == vocab.get(tokens[t+1],1):
            correct += 1
        total += 1
if total > 0:
    print(f"\nNext-token accuracy: {correct/total:.1%}")

    