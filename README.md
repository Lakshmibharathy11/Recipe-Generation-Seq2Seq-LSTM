# 🍳 Recipe Generation with a Seq2Seq LSTM

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%203-orange) ![Task](https://img.shields.io/badge/NLP-seq2seq%20text%20generation-green)

Generate step-by-step cooking instructions from a list of ingredients and recipe tags, using an **encoder–decoder LSTM with BPE subword tokenization**, trained on **~230K recipes** from Food.com and benchmarked against a retrieval baseline.

> Part of the **Plate Planner** project for DATA 245 (Machine Learning), San José State University.
> This repo is the generative component. Plate Planner's ingredient-substitution and recipe-recommendation service (FastAPI + Neo4j + FAISS) is a separate codebase.

**Highlights**
- Built in three iterations: a word-level baseline, then BPE subwords, then a corrected pipeline with an audit. Each version was driven by a measured failure of the previous one.
- Found and fixed **train/test leakage**, **token truncation that dropped recipe openings**, and **padding-inflated accuracy** in the earlier pipeline.
- Evaluation goes beyond one BLEU number: 5 metrics on the full held-out test set, a **non-neural retrieval baseline**, and a **paired-bootstrap significance test**.
- Includes a Streamlit demo app.

---

## 📌 Problem

Given what a user has (ingredients) and what they want (tags such as `30 minutes or less`, `vegetarian`, `desserts`), generate plausible cooking steps.

```
Input  : winter squash  mexican seasoning  honey  butter  olive oil  salt | 60 minutes or less  vegetables ...
Output : preheat oven to 400 degrees f . cut squash in half and remove seeds . brush with olive oil ...
```

This is a **sequence-to-sequence** task. An encoder reads the ingredient and tag sequence, and a decoder generates the instructions one subword at a time.

---

## 📊 Data

| | |
|---|---|
| **Source** | [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) (`RAW_recipes.csv`) |
| **Size** | 231,637 recipes, ~230K after cleaning |
| **Input** | ingredients + tags (lowercased, hyphens in tags → spaces) |
| **Target** | steps joined with ` . `, so the output can be split back into numbered steps |
| **Split** | 81% train / 9% val / 10% test (`seed=42`), made **before** tokenization; recipe ids asserted disjoint |

Length profile (BPE tokens): median target ≈ 97 tokens. 3.6% of recipes exceed the 300-token cap and are truncated **at the end**, always keeping the BOS and EOS tokens. EDA (ingredient frequencies, cooking-time buckets, word clouds) is in [`01_eda.ipynb`](notebooks/01_eda.ipynb).

---

## 🧠 Model

```
 ingredients + tags                                   cooking steps
        │                                                   ▲
 ┌──────▼──────┐                                   ┌────────┴────────┐
 │ Embedding   │ 256-d, mask_zero                  │ Dense + softmax │ 8,000
 └──────┬──────┘                                   └────────▲────────┘
 ┌──────▼──────┐    final (h, c) states       ┌─────────────┴──────┐
 │ LSTM (512)  │ ───────────────────────────► │    LSTM (512)      │
 └─────────────┘    initialize decoder        └─────────────▲──────┘
     ENCODER                                    ┌───────────┴─────┐
                                                │ Embedding 256-d │ ◄── previous token (BOS first)
                                                └─────────────────┘
                                                     DECODER
```

| | |
|---|---|
| Tokenizer | SentencePiece BPE, 8,000 pieces, trained on **train split only**; PAD/BOS/EOS/UNK = 0/1/2/3 |
| Max lengths | 132 input / 300 target tokens |
| Parameters | 11.35M |
| Training | Teacher forcing, Adam (lr 1e-3, clipnorm 1.0), batch 64, ≤10 epochs, early stopping on val loss (patience 2) |
| Loss / accuracy | Sparse categorical cross-entropy over **non-padding tokens only** |
| Decoding | Batched greedy decoding until EOS (max 299 tokens) |

---

## 🔁 How the model evolved

| Version | Notebook | What changed | What we learned |
|---|---|---|---|
| **v1** word-level | [`02_v1_word_level_seq2seq.ipynb`](notebooks/02_v1_word_level_seq2seq.ipynb) | Keras `Tokenizer`, 5K input / 10K target vocab | Val accuracy 0.818, but outputs often collapsed into `<OOV> <OOV> <OOV> …`. Token accuracy was rewarding frequent OOV and padding tokens, not good recipes. |
| **v2** BPE | [`03_v2_bpe_seq2seq.ipynb`](notebooks/03_v2_bpe_seq2seq.ipynb) | 8K SentencePiece subwords | OOV problem gone, fluent instructions. Reported BLEU 0.040 on 10 samples, but an audit (below) showed that number wasn't trustworthy. |
| **v3** corrected | [`04_v3_fixed_pipeline.ipynb`](notebooks/04_v3_fixed_pipeline.ipynb) | Fixed pipeline + full evaluation | Final numbers below. |

### Audit of v2 and the fixes in v3

| # | Problem found in v2 | Impact | Fix in v3 |
|---|---|---|---|
| 1 | Model trained on the full dataset, *then* tested on a 10% split of the same data | **Leakage**: test BLEU was measured on recipes the model had already seen | Split first; train on train, tune on val, report on test; disjointness asserted in code |
| 2 | `<start>` / `<end>` written as plain text | BPE split them into fragments (`start ⁇` appeared in outputs) | Use SentencePiece's native BOS/EOS ids |
| 3 | `pad_sequences` default `truncating="pre"` | Long recipes **lost their start token and opening steps** | Post-truncation that always keeps BOS…EOS |
| 4 | Padding included in loss/accuracy | Accuracy inflated by easy-to-predict pads | `mask_zero=True` + per-token sample weights |
| 5 | BLEU on 10 samples, no baseline | No way to tell whether the model adds value | 5 metrics on the full test set, retrieval baseline, bootstrap CIs |
| 6 | Steps joined with spaces | Couldn't split output back into steps | Steps joined with ` . ` |

---

## 🧪 Evaluation

**Systems compared** on the held-out test set (~23K recipes):
- **Seq2seq LSTM + BPE (v3)**: this model.
- **Retrieval baseline**: finds the most similar *training* recipe by TF-IDF cosine similarity over ingredients + tags and returns its steps. A strong, cheap baseline: the model has to beat "copy the closest existing recipe" to justify itself.
- **Human reference**: the real recipe, which sets a ceiling for the reference-free metrics.

**Metrics**
| Metric | What it measures |
|---|---|
| Corpus BLEU-4 | n-gram overlap with the reference, pooled over the test set |
| Sentence BLEU | per-recipe BLEU, averaged (smoothing method 4, comparable to v2's 0.040) |
| ROUGE-L F1 | longest-common-subsequence overlap; rewards correct step order (2K-recipe subset) |
| Ingredient coverage | share of the input ingredients actually mentioned in the generated steps |
| Distinct-2 | unique bigrams / total bigrams; low values flag repetitive, generic text |

### Results

> ⏳ **Pending the full v3 training run.** Run [`04_v3_fixed_pipeline.ipynb`](notebooks/04_v3_fixed_pipeline.ipynb) on a Colab GPU. Its last cell prints this table ready to paste, and writes `models/eval_metrics_v3.json`.

| System | Corpus BLEU-4 | Sentence BLEU | ROUGE-L | Ingredient coverage | Distinct-2 |
|---|---|---|---|---|---|
| Retrieval baseline (TF-IDF 1-NN) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **Seq2seq LSTM + BPE (v3)** | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Human reference (ceiling) | – | – | – | _tbd_ | _tbd_ |

**Significance:** paired bootstrap (2,000 resamples) of the per-recipe difference between the LSTM and retrieval, with a 95% CI and p-value: _tbd_.

<details>
<summary>Pipeline smoke test (20K-recipe sample, baseline only; not final results)</summary>

The non-neural parts of the v3 notebook were validated on a 20K-recipe sample (16.2K train / 2K test):

| System | Corpus BLEU-4 | Sentence BLEU | ROUGE-L | Ingredient coverage | Distinct-2 |
|---|---|---|---|---|---|
| Retrieval baseline | 0.052 | 0.039 | 0.214 | 0.373 | 0.196 |
| Human reference | – | – | – | 0.703 | 0.205 |

Even a simple nearest-recipe lookup on a small sample reaches about the same sentence BLEU as v2's reported 0.040. This shows why v2's single number couldn't support any claim on its own, and why v3 adds a baseline.
</details>

### Why BLEU is low for every system
A recipe can be written correctly in many ways, and BLEU only rewards word overlap with *one* reference. Low absolute BLEU is normal for open-ended generation, so **the comparison against the baseline matters more than the absolute value**. Ingredient coverage and Distinct-2 catch the two typical seq2seq failures that BLEU misses: ignoring the given ingredients, and producing generic, repetitive text.

---

## 🖥️ Demo app

[`app/`](app) is a Streamlit UI. You enter ingredients, a time budget and a diet; [`your_model.py`](app/your_model.py) maps them to Food.com-style tags (for example 30 min → `30 minutes or less`, Vegan → `vegan vegetarian`), runs greedy decoding and splits the output into numbered steps.

```bash
# after training: put the notebook's models/ folder in the repo root (or notebooks/models/)
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 🚀 Reproduce

0. **Setup:** `git clone https://github.com/Lakshmibharathy11/Recipe-Generation-Seq2Seq-LSTM.git && pip install -r requirements.txt`
1. **Data:** download `RAW_recipes.csv` from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) into `data/`. In Colab, upload it when the notebook prompts you.
2. **Train and evaluate:** open [`04_v3_fixed_pipeline.ipynb`](notebooks/04_v3_fixed_pipeline.ipynb) in Google Colab (GPU runtime) and **Run all**. This takes about 30 to 40 minutes on a T4. Set `SAMPLE_N = 20000` for a ~5-minute smoke run.
3. **Outputs** (written to `models/`):

| File | Contents |
|---|---|
| `recipe_seq2seq_bpe_v3.keras` | best checkpoint by val loss |
| `bpe_v3.model` / `.vocab` | SentencePiece tokenizer |
| `history_v3.json`, `training_curves_v3.png` | training curves |
| `eval_metrics_v3.json` | all metrics + bootstrap results |
| `sample_predictions_v3.csv` | 200 test inputs with reference, retrieval and LSTM outputs |

4. **Generate** in the notebook:
```python
generate("chicken breast garlic lemon olive oil salt pepper 30 minutes or less main dish")
```

---

## 📁 Structure

```
Recipe-Generation-Seq2Seq-LSTM/
├── notebooks/
│   ├── 01_eda.ipynb                     # exploratory data analysis
│   ├── 02_v1_word_level_seq2seq.ipynb   # v1: word-level baseline
│   ├── 03_v2_bpe_seq2seq.ipynb          # v2: BPE (kept for reference; see audit)
│   └── 04_v3_fixed_pipeline.ipynb       # ✅ final: corrected pipeline + evaluation
├── app/
│   ├── app.py                           # Streamlit UI
│   └── your_model.py                    # v3 inference wrapper
├── data/                                # RAW_recipes.csv goes here (git-ignored)
├── models/                              # trained artifacts (git-ignored, too large for GitHub)
├── requirements.txt
└── README.md
```

> Trained models (~140 MB each) and the dataset aren't committed. Re-create them with the v3 notebook.

---

## 🔭 Next steps

- **Attention** (Bahdanau/Luong), so the decoder can look back at specific ingredients. This should raise ingredient coverage.
- **Beam search** or nucleus sampling instead of greedy decoding, to reduce generic outputs.
- A **fine-tuned small Transformer** (for example T5-small) as a stronger neural baseline.
- **Human evaluation** of a sample for coherence and food safety (timings and temperatures).

---

## 🛠️ Tech stack

`Python` · `TensorFlow / Keras 3` · `SentencePiece` · `scikit-learn` (TF-IDF retrieval, splits) · `NLTK` (BLEU) · `pandas` · `NumPy` · `Matplotlib` · `Streamlit` · `Google Colab (GPU)`

## 📚 References

- Sutskever et al., *Sequence to Sequence Learning with Neural Networks*, NeurIPS 2014.
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016.
- Kudo & Richardson, *SentencePiece*, EMNLP 2018.
- Majumder et al., *Generating Personalized Recipes from Historical User Preferences*, EMNLP 2019 (Food.com dataset).
