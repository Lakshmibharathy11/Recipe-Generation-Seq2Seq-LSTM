"""Inference wrapper for the v3 seq2seq recipe model (see notebooks/04_v3_fixed_pipeline.ipynb).

Expects the artifacts the notebook writes, in <repo>/models/ (or notebooks/models/):
    recipe_seq2seq_bpe_v3.keras   trained model
    bpe_v3.model                  SentencePiece tokenizer
Set RECIPE_MODEL_DIR to load them from somewhere else.
"""
import os
from functools import lru_cache

import numpy as np

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_DIR = os.environ.get("RECIPE_MODEL_DIR") or next(
    (d for d in (os.path.join(_ROOT, "models"), os.path.join(_ROOT, "notebooks", "models"))
     if os.path.isdir(d)),
    os.path.join(_ROOT, "models"),
)
MODEL_PATH = os.path.join(MODEL_DIR, "recipe_seq2seq_bpe_v3.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "bpe_v3.model")

MAX_LEN_INPUT = 132    # must match the notebook
MAX_LEN_TARGET = 300

# The model was trained on Food.com tags, so map the UI choices onto real tag text.
DIET_TAGS = {"Any": "", "Vegan": "vegan vegetarian", "Vegetarian": "vegetarian", "Non-Vegetarian": "meat"}
BASE_TAGS = "time to make course main ingredient preparation main dish"


def _time_tag(minutes):
    for limit, tag in [(15, "15 minutes or less"), (30, "30 minutes or less"),
                       (60, "60 minutes or less"), (240, "4 hours or less")]:
        if minutes <= limit:
            return tag
    return ""


@lru_cache(maxsize=1)
def _load():
    """Load the model and tokenizer once, on first use."""
    import keras
    import sentencepiece as spm
    import tensorflow as tf

    for p in (MODEL_PATH, TOKENIZER_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run notebooks/04_v3_fixed_pipeline.ipynb and copy its models/ folder "
                f"to the repo root (or set RECIPE_MODEL_DIR)."
            )
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model = keras.models.load_model(MODEL_PATH)
    enc_emb, enc_lstm = model.get_layer("enc_emb"), model.get_layer("enc_lstm")
    dec_emb, dec_lstm, dec_dense = (model.get_layer(n) for n in ("dec_emb", "dec_lstm", "dec_dense"))

    @tf.function(reduce_retracing=True)
    def encode(x):
        _, h, c = enc_lstm(enc_emb(x), mask=tf.not_equal(x, 0))
        return h, c

    @tf.function(reduce_retracing=True)
    def step(tok, h, c):
        out, h, c = dec_lstm(dec_emb(tok[:, None]), initial_state=[h, c])
        return tf.argmax(dec_dense(out[:, -1, :]), axis=-1, output_type=tf.int32), h, c

    return sp, encode, step, tf


def _generate_steps(input_text):
    sp, encode, step, tf = _load()
    ids = sp.encode(input_text.lower())[:MAX_LEN_INPUT]
    x = np.zeros((1, MAX_LEN_INPUT), dtype=np.int32)
    x[0, :len(ids)] = ids

    h, c = encode(tf.constant(x))
    tok, out = tf.constant([sp.bos_id()], dtype=tf.int32), []
    for _ in range(MAX_LEN_TARGET - 1):
        tok, h, c = step(tok, h, c)
        t = int(tok.numpy()[0])
        if t == sp.eos_id():
            break
        out.append(t)
    return sp.decode(out)


def generate_recipe(ingredients, time, diet):
    items = [i.strip().lower() for i in ingredients.split(",") if i.strip()]
    input_text = " ".join(items + [_time_tag(time), BASE_TAGS, DIET_TAGS.get(diet, "")])
    text = _generate_steps(" ".join(input_text.split()))

    steps = [s.strip().capitalize() for s in text.split(" . ") if s.strip()]
    title = f"{diet + ' ' if diet != 'Any' else ''}{' & '.join(items[:2]).title()} recipe".strip()
    return {"title": title, "ingredients": items, "instructions": steps}
