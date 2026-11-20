"""Llama rewriter (Person B, Madina).

Interface (do not change without telling the team):
    rewrite(text, spans, rules, feedback=None)
        -> {"revised": str, "changes": [{"original", "revised", "reason"}]}

How it works:
- Model: config.REWRITER_MODEL (Llama 3.1 8B Instruct) in 4-bit NF4, loaded lazily on
  the first real call by get_model(). No fallback model: if Llama cannot be loaded,
  get_model() raises with the reason (no token, no license access, out of memory).
  check_model_access() tests the token and licence in seconds, before the download.
  get_model() / generate() / generate_batch() are reusable by reviewers.py.
- Prompts: prompts/rewriter_system.txt, prompts/rewriter_user.txt,
  prompts/rewriter_retry.txt.
- Output is parsed and validated (schema, spans really in the text, no new numbers
  or capitalised names). On failure the model gets ONE retry with the errors; if
  that fails too, the input is returned unchanged and the failure goes to the
  audit log. A changed text is never returned without a matching change list.
- DEBIAS_FAKE=1 (config.USE_FAKES) keeps the fake stand-in; tests and CI use it.
  torch / transformers are only imported inside get_model() / generate_detailed().
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from string import Template
from typing import Callable

from . import config, contract, utils

PROMPT_FILES = ("rewriter_system.txt", "rewriter_user.txt", "rewriter_retry.txt")


# ---------------------------------------------------------------------------
# Fake stand-in (DEBIAS_FAKE=1)
# ---------------------------------------------------------------------------
def _fake_rewrite(text: str, spans: list[str], rules: list[str], feedback: str | None) -> dict:
    return {"revised": text, "changes": []}


# ---------------------------------------------------------------------------
# Model loading and generation (reusable by reviewers.py)
# ---------------------------------------------------------------------------
_TOKEN_HELP = (
    "Colab: key icon on the left > add a secret named HF_TOKEN (a Read token from "
    "https://huggingface.co/settings/tokens) and switch ON 'Notebook access' for this notebook. "
    "Kaggle: Add-ons > Secrets > HF_TOKEN. Then run the setup cell again."
)


def access_error_message(model_id: str, exc: Exception | None) -> str:
    """Plain-language reason why `model_id` cannot be downloaded, with the fix.

    Matches exceptions by class name so it works across huggingface_hub versions.
    """
    page = f"https://huggingface.co/{model_id}"
    if exc is None:
        return f"No Hugging Face token found, and {model_id} is a gated model. {_TOKEN_HELP}"
    names = {cls.__name__ for cls in type(exc).__mro__}
    text = str(exc)
    if "GatedRepoError" in names or "403" in text:
        return (
            f"Your Hugging Face account has no access to {model_id} yet. Open {page} while logged in "
            f"with the SAME account the token belongs to and accept the Llama license form. Approval "
            f"usually takes minutes to a few hours; the status is at https://huggingface.co/settings/gated-repos "
            f"(it must say 'accepted'). Then restart the runtime and run the setup cell again."
        )
    if "401" in text or "Invalid user token" in text or "RepositoryNotFoundError" in names:
        return f"The HF_TOKEN was rejected (expired, mistyped or revoked). Make a new Read token. {_TOKEN_HELP}"
    return f"Could not reach {model_id}: {type(exc).__name__}: {exc}"


def check_model_access(model_id: str = config.REWRITER_MODEL) -> str:
    """Fast check before the big download: token present, valid, and allowed into the gated repo.

    Downloads only the model's small config.json. Returns the Hugging Face username.
    Raises RuntimeError with the fix if anything is missing.
    """
    from huggingface_hub import HfApi, hf_hub_download

    token = utils.get_hf_token()  # never printed or logged
    if not token:
        raise RuntimeError(access_error_message(model_id, None))
    try:
        user = HfApi().whoami(token=token)["name"]
        hf_hub_download(model_id, "config.json", token=token)
    except Exception as e:
        raise RuntimeError(access_error_message(model_id, e)) from None
    return user


@lru_cache(maxsize=1)
def get_model():
    """Return (model, tokenizer, model_id). Loads config.REWRITER_MODEL once, on the first call.

    4-bit NF4 with double quantisation, float16 compute (the T4 has no fast bf16).
    There is no fallback model: if Llama cannot be loaded this raises with the reason,
    so every result is guaranteed to come from config.REWRITER_MODEL.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = config.REWRITER_MODEL
    utils.set_seed(config.SEED)
    check_model_access(model_id)
    token = utils.get_hf_token()  # never printed or logged
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, quantization_config=quant,
            device_map="auto", dtype=torch.float16,
        )
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            f"Out of GPU memory loading {model_id}. Use a T4 (or bigger) GPU runtime, and if another "
            f"model is already loaded, Runtime > Restart session, then run the setup cell again."
        ) from None
    except Exception as e:
        raise RuntimeError(f"Could not load {model_id}: {type(e).__name__}: {e}") from e
    if not torch.cuda.is_available():
        print(f"[rewriter] WARNING: no GPU, {model_id} will be extremely slow")
    tokenizer.padding_side = "left"  # decoder-only models need left padding in batches
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, model_id


def loaded_model_id() -> str | None:
    """Id of the model get_model() loaded, or None if nothing is loaded yet."""
    return get_model()[2] if get_model.cache_info().currsize else None


def generate_detailed(batch: list[list[dict]]) -> list[dict]:
    """Greedy generation for a batch of chat message lists.

    Returns one dict per input: {"text": new text only, "new_tokens": int,
    "hit_limit": bool (stopped at config.MAX_NEW_TOKENS, output may be cut off)}.
    """
    import torch

    model, tokenizer, _ = get_model()
    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
    # The chat template already contains the BOS token, so do not add it twice.
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False, temperature=None, top_p=None, top_k=None,  # greedy, no sampling warnings
            max_new_tokens=config.MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = out[:, enc["input_ids"].shape[1]:].tolist()

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tokenizer.eos_token_id}
    results = []
    for row in new_ids:
        stop = next((i for i, t in enumerate(row) if t in eos_ids), None)
        n = len(row) if stop is None else stop + 1
        text = tokenizer.decode(row[:n], skip_special_tokens=True).strip()
        results.append({"text": text, "new_tokens": n, "hit_limit": stop is None})
    return results


def generate(messages: list[dict]) -> str:
    """Greedy generation for one chat; returns only the new text."""
    return generate_detailed([messages])[0]["text"]


def generate_batch(batch: list[list[dict]]) -> list[str]:
    """Greedy generation for several chats at once (left padding)."""
    return [r["text"] for r in generate_detailed(batch)]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_FEEDBACK_BLOCK = re.compile(r"<<IF_FEEDBACK>>\n(.*?)<<END_IF_FEEDBACK>>\n", re.S)


def _prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


def prompt_versions() -> dict:
    """Short hash of each prompt file, to save next to results."""
    return {n: hashlib.sha256(_prompt(n).encode()).hexdigest()[:10] for n in PROMPT_FILES}


def build_messages(text: str, spans: list[str], rules: list[str], feedback: str | None = None) -> list[dict]:
    template = _prompt("rewriter_user.txt")
    # Keep the feedback section only when there is feedback.
    template = _FEEDBACK_BLOCK.sub(lambda m: m.group(1) if feedback else "", template)
    user = Template(template).substitute(
        text=text,
        spans="\n".join(f'- "{s}"' for s in spans),
        rules="\n\n".join(r.strip() for r in rules),
        feedback=feedback or "",
    )
    return [
        {"role": "system", "content": _prompt("rewriter_system.txt").strip()},
        {"role": "user", "content": user.strip()},
    ]


def _retry_message(errors: list[str]) -> dict:
    content = Template(_prompt("rewriter_retry.txt")).substitute(errors="\n".join(f"- {e}" for e in errors))
    return {"role": "user", "content": content.strip()}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def strip_fences(raw: str) -> str:
    """Remove markdown code fences such as ```json ... ```."""
    return re.sub(r"```(?:json|JSON)?", "", raw)


def first_json_object(s: str) -> str | None:
    """Return the first balanced {...} block (braces inside strings are ignored)."""
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, escaped = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_output(raw: str) -> dict:
    """Model text -> dict. Raises ValueError with a short reason."""
    block = first_json_object(strip_fences(raw))
    if block is None:
        raise ValueError("no complete JSON object found (the output may be cut off)")
    try:
        obj = json.loads(block, strict=False)  # strict=False tolerates raw newlines in strings
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e.msg} at character {e.pos}") from None
    if not isinstance(obj, dict):
        raise ValueError("the JSON is not an object")
    return obj


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
# Capitalised words that may start a rewritten sentence without being a new name.
_SENTENCE_STARTERS = {
    "a", "an", "the", "according", "it", "this", "that", "these", "those", "there",
    "some", "both", "in", "on", "at", "as", "after", "before", "during", "while",
    "however", "reports", "officials", "sources", "witnesses", "no", "many", "several",
}


def _strip_possessive(word: str) -> str:
    return re.sub(r"['’]s$", "", word)


def find_new_facts(original: str, revised: str) -> dict:
    """Numbers and capitalised words in `revised` that are not in `original`.

    Simple regex check, not a full fact check. Case-insensitive against the
    original's words, so a word that is only capitalised because it now starts a
    sentence is not counted; a few common sentence starters ("According", "The")
    are ignored at the start of a sentence.
    """
    new_numbers = sorted(set(_NUMBER.findall(revised)) - set(_NUMBER.findall(original)))
    orig_words = {_strip_possessive(w).lower() for w in _WORD.findall(original)}
    new_names = set()
    for m in _WORD.finditer(revised):
        word = _strip_possessive(m.group())
        if not word[0].isupper() or word.lower() in orig_words:
            continue
        before = revised[: m.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:;\"'(“”\n"
        if sentence_start and word.lower() in _SENTENCE_STARTERS:
            continue
        new_names.add(word)
    return {"new_numbers": new_numbers, "new_names": sorted(new_names)}


def validate_output(text: str, out: dict) -> list[str]:
    """Return a list of problems with a parsed model output (empty list = valid)."""
    try:
        contract.validate_rewrite(out)
    except contract.ContractError as e:
        return [f"wrong schema: {e}"]
    errors = []
    revised, changes = out["revised"], out["changes"]
    if not revised.strip():
        errors.append('"revised" is empty')
    for i, c in enumerate(changes, 1):
        if not c["original"].strip():
            errors.append(f'change {i}: "original" is empty')
        elif c["original"] not in text:
            errors.append(f'change {i}: original "{c["original"]}" is not copied exactly from TEXT')
        if c["revised"] not in revised:
            errors.append(f'change {i}: revised "{c["revised"]}" does not appear in your revised text')
        if c["original"] == c["revised"]:
            errors.append(f"change {i}: original and revised are the same; do not list it")
        if not c["reason"].strip():
            errors.append(f"change {i}: reason is empty")
    unchanged = revised.strip() == text.strip()
    if not unchanged and not changes:
        errors.append('the text was changed but "changes" is empty; list every change')
    if unchanged and changes:
        errors.append('"changes" lists edits but the revised text is identical to TEXT')
    facts = find_new_facts(text, revised)
    if facts["new_numbers"]:
        errors.append(f"revised text adds numbers that are not in TEXT: {facts['new_numbers']}")
    if facts["new_names"]:
        errors.append(f"revised text adds names that are not in TEXT: {facts['new_names']}")
    return errors


def _clean(out: dict, text: str) -> dict:
    """Keep only the contract keys; an unchanged text returns exactly the input."""
    changes = [{k: c[k] for k in ("original", "revised", "reason")} for c in out["changes"]]
    if out["revised"].strip() == text.strip():
        return {"revised": text, "changes": []}
    return {"revised": out["revised"], "changes": changes}


# ---------------------------------------------------------------------------
# rewrite()
# ---------------------------------------------------------------------------
def rewrite_detailed(
    text: str,
    spans: list[str],
    rules: list[str],
    feedback: str | None = None,
    generate_fn: Callable[[list[dict]], str | dict] | None = None,
) -> tuple[dict, dict]:
    """Like rewrite(), but also returns run info for notebooks and tests.

    info = {"skipped", "fake", "model_id", "retries", "fell_back",
            "attempts": [{"attempt", "raw", "parsed", "errors", "new_tokens",
                          "hit_limit", "new_numbers", "new_names"}]}
    `generate_fn` replaces the model (tests pass hand-written outputs).
    """
    info = {"skipped": False, "fake": False, "model_id": None, "retries": 0, "fell_back": False, "attempts": []}
    if not spans:
        info["skipped"] = True
        return {"revised": text, "changes": []}, info
    if config.USE_FAKES and generate_fn is None:
        info["fake"] = True
        return _fake_rewrite(text, spans, rules, feedback), info

    gen = generate_fn or (lambda msgs: generate_detailed([msgs])[0])
    messages = build_messages(text, spans, rules, feedback)
    for attempt in (1, 2):
        g = gen(messages)
        if isinstance(g, str):
            g = {"text": g, "new_tokens": None, "hit_limit": None}
        raw = g["text"]
        out, facts = None, {"new_numbers": [], "new_names": []}
        try:
            out = parse_output(raw)
            errors = validate_output(text, out)
            if isinstance(out.get("revised"), str):
                facts = find_new_facts(text, out["revised"])
        except ValueError as e:
            errors = [str(e)]
            if g["hit_limit"]:
                errors.append(f"the answer hit the {config.MAX_NEW_TOKENS}-token limit; keep reasons very short")
        info["attempts"].append({"attempt": attempt, "raw": raw, "parsed": out is not None, "errors": errors,
                                 "new_tokens": g["new_tokens"], "hit_limit": g["hit_limit"], **facts})
        if generate_fn is None:
            info["model_id"] = loaded_model_id()
        if not errors:
            return _clean(out, text), info
        if attempt == 1:
            info["retries"] = 1
            messages = messages + [{"role": "assistant", "content": raw}, _retry_message(errors)]

    info["fell_back"] = True
    utils.audit_log({
        "event": "rewrite_failed",
        "model_id": info["model_id"],
        "prompts": prompt_versions(),
        "text": text,
        "spans": spans,
        "feedback": feedback,
        "attempts": info["attempts"],
    })
    return {"revised": text, "changes": []}, info


def rewrite(text: str, spans: list[str], rules: list[str], feedback: str | None = None) -> dict:
    result, _ = rewrite_detailed(text, spans, rules, feedback)
    return result
