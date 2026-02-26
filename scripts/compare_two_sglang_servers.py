#!/usr/bin/env python3
"""
Compare two SGLang servers (e.g. no-spec vs EAGLE) on:
1) greedy output token ids
2) teacher-forced logprobs / top-k logprobs for a reference completion

Typical usage:
  # start two servers first:
  #   30000: no speculative
  #   30001: --speculative-algorithm EAGLE ...
  python scripts/compare_two_sglang_servers.py \
    --url0 http://127.0.0.1:30000 \
    --url1 http://127.0.0.1:30001 \
    --prompt-file prompt_1.txt \
    --max-new-tokens 64 \
    --top-logprobs 20
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


def _session() -> requests.Session:
    # Avoid picking up HTTP(S)_PROXY from the environment for local calls.
    s = requests.Session()
    s.trust_env = False
    return s


def _post_json(
    sess: requests.Session,
    base_url: str,
    path: str,
    payload: Dict[str, Any],
    *,
    timeout_s: float,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + path
    resp = sess.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _get_json(
    sess: requests.Session,
    base_url: str,
    path: str,
    *,
    timeout_s: float,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + path
    resp = sess.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _summarize_value(v: Any, *, max_len: int = 160) -> str:
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float)):
        return str(v)
    if isinstance(v, str):
        s = v.replace("\n", "\\n")
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return json.dumps(s, ensure_ascii=False)
    if isinstance(v, list):
        if len(v) <= 8 and all(isinstance(x, (bool, int, float, str, type(None))) for x in v):
            return json.dumps(v, ensure_ascii=False)
        return f"list(len={len(v)})"
    if isinstance(v, dict):
        return f"dict(keys={len(v)})"
    return f"{type(v).__name__}"


def _print_server_info_diff(
    sess: requests.Session,
    *,
    url0: str,
    url1: str,
    timeout_s: float,
) -> None:
    """
    Print a compact diff of /get_server_info to catch config mismatches that can
    cause deterministic-but-different outputs (chat template, stop strings, etc.).
    """
    info0 = _get_json(sess, url0, "/get_server_info", timeout_s=timeout_s)
    info1 = _get_json(sess, url1, "/get_server_info", timeout_s=timeout_s)

    # Drop volatile keys that change over time or are huge.
    ignore = {
        "internal_states",
        "version",
        "scheduler",
        "scheduler_info",
        "active_requests",
        "uptime_seconds",
        "last_receive_timestamp",
        "start_time",
    }

    keys = sorted(set(info0.keys()) | set(info1.keys()))
    diffs = []
    for k in keys:
        if k in ignore:
            continue
        v0 = info0.get(k, None)
        v1 = info1.get(k, None)
        if v0 == v1:
            continue
        diffs.append((k, v0, v1))

    if not diffs:
        print("[ServerInfo] no diff (excluding volatile keys)")
        return

    print(f"[ServerInfo] diffs={len(diffs)} (excluding volatile keys)")
    # Heuristic: show the most likely generation-affecting keys first.
    prefer = [
        "model_path",
        "tokenizer_path",
        "served_model_name",
        "weight_version",
        "chat_template",
        "completion_template",
        "reasoning_parser",
        "tool_call_parser",
        "sampling_defaults",
        "preferred_sampling_params",
        "random_seed",
        "enable_deterministic_inference",
        "dtype",
        "kv_cache_dtype",
        "disable_cuda_graph",
        "disable_radix_cache",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "context_length",
        "tp_size",
        "pp_size",
        "dp_size",
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "speculative_eagle_topk",
    ]
    prefer_set = set(prefer)
    diffs.sort(key=lambda x: (0 if x[0] in prefer_set else 1, prefer.index(x[0]) if x[0] in prefer_set else x[0]))

    for k, v0, v1 in diffs:
        print(f"  - {k}: { _summarize_value(v0) }  !=  { _summarize_value(v1) }")


def _load_hf_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for --use-chat-template. "
            "Install it in your env (pip/conda) and retry."
        ) from e

    # Qwen-style tokenizers are usually safe; keep trust_remote_code=True for
    # custom tokenizers used in internal checkpoints.
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _chat_prompt_ids(
    tokenizer,
    *,
    system: Optional[str],
    user: str,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> List[int]:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs = dict(chat_template_kwargs or {})
    # Mirror SGLang's serving_chat.py: always pass reasoning_effort.
    kwargs.setdefault("reasoning_effort", "medium")
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
        raise TypeError(f"Unexpected apply_chat_template output: {type(ids)=}")
    return [int(x) for x in ids]


def tokenize(
    sess: requests.Session,
    base_url: str,
    prompt: str,
    *,
    add_special_tokens: bool = True,
    timeout_s: float,
) -> List[int]:
    data = _post_json(
        sess,
        base_url,
        "/v1/tokenize",
        {"prompt": prompt, "add_special_tokens": add_special_tokens},
        timeout_s=timeout_s,
    )
    tokens = data["tokens"]
    if not isinstance(tokens, list) or (tokens and not isinstance(tokens[0], int)):
        raise TypeError(f"Unexpected tokenize response: {type(tokens)=}, {tokens[:3]=}")
    return [int(x) for x in tokens]


def detokenize(
    sess: requests.Session,
    base_url: str,
    token_ids: Sequence[int],
    *,
    skip_special_tokens: bool = True,
    timeout_s: float,
) -> str:
    data = _post_json(
        sess,
        base_url,
        "/v1/detokenize",
        {"tokens": list(map(int, token_ids)), "skip_special_tokens": skip_special_tokens},
        timeout_s=timeout_s,
    )
    text = data.get("text", "")
    if not isinstance(text, str):
        raise TypeError(f"Unexpected detokenize response: {type(text)=}")
    return text


def generate(
    sess: requests.Session,
    base_url: str,
    *,
    input_ids: Sequence[int],
    sampling_params: Dict[str, Any],
    return_logprob: bool = False,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: bool = False,
    timeout_s: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "input_ids": list(map(int, input_ids)),
        "sampling_params": sampling_params,
        "stream": False,
    }
    if return_logprob:
        payload["return_logprob"] = True
    if logprob_start_len is not None:
        payload["logprob_start_len"] = int(logprob_start_len)
    if top_logprobs_num is not None:
        payload["top_logprobs_num"] = int(top_logprobs_num)
    if return_text_in_logprobs:
        payload["return_text_in_logprobs"] = True

    return _post_json(sess, base_url, "/generate", payload, timeout_s=timeout_s)


def _first_mismatch(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


@dataclass(frozen=True)
class ForcedTokenStat:
    idx: int
    token_id: int
    token_text: Optional[str]
    logprob0: Optional[float]
    logprob1: Optional[float]
    abs_diff: Optional[float]
    rank0: Optional[int]
    rank1: Optional[int]
    top1_id0: Optional[int]
    top1_id1: Optional[int]


def _parse_topk_entry(
    entry: Any,
) -> Tuple[Optional[int], Dict[int, float]]:
    """Return (top1_token_id, {token_id: logprob})."""
    if entry is None:
        return None, {}
    if not isinstance(entry, list):
        raise TypeError(f"Unexpected top_logprobs entry type: {type(entry)=}")
    token_to_logprob: Dict[int, float] = {}
    top1_id: Optional[int] = None
    top1_lp: Optional[float] = None
    for item in entry:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        lp = item[0]
        tid = item[1]
        if lp is None:
            continue
        tid_i = int(tid)
        lp_f = float(lp)
        token_to_logprob[tid_i] = lp_f
        if top1_lp is None or lp_f > top1_lp:
            top1_lp = lp_f
            top1_id = tid_i
    return top1_id, token_to_logprob


def _forced_stats_for_ref_completion(
    *,
    prompt_len: int,
    ref_output_ids: Sequence[int],
    eval_meta0: Dict[str, Any],
    eval_meta1: Dict[str, Any],
) -> List[ForcedTokenStat]:
    """
    We evaluate token logprobs using a teacher-forcing request where:
      origin_input_ids = prompt_ids + ref_output_ids
      logprob_start_len = prompt_len - 1

    The returned meta_info contains:
      input_token_logprobs: List[(logprob, token_id, token_text?)] aligned to
        origin_input_ids[logprob_start_len:].
      input_top_logprobs: List[Optional[List[(logprob, token_id, token_text?)]]] aligned similarly.
    """
    start = prompt_len - 1
    if start < 0:
        raise ValueError(f"{prompt_len=} is too small")

    token_logprobs0 = eval_meta0.get("input_token_logprobs", [])
    token_logprobs1 = eval_meta1.get("input_token_logprobs", [])
    top_logprobs0 = eval_meta0.get("input_top_logprobs", [])
    top_logprobs1 = eval_meta1.get("input_top_logprobs", [])

    expected_len = 1 + len(ref_output_ids)
    if len(token_logprobs0) < expected_len or len(token_logprobs1) < expected_len:
        raise ValueError(
            "Not enough input_token_logprobs returned. "
            f"{len(token_logprobs0)=}, {len(token_logprobs1)=}, {expected_len=}. "
            "Tip: ensure return_logprob=True and logprob_start_len is set."
        )

    stats: List[ForcedTokenStat] = []
    for i, token_id in enumerate(ref_output_ids):
        # +1 because the first entry corresponds to origin_input_ids[start] (last prompt token),
        # whose token logprob is None by design.
        entry0 = token_logprobs0[i + 1]
        entry1 = token_logprobs1[i + 1]

        lp0 = entry0[0] if isinstance(entry0, (list, tuple)) and entry0 else None
        lp1 = entry1[0] if isinstance(entry1, (list, tuple)) and entry1 else None
        txt0 = entry0[2] if isinstance(entry0, (list, tuple)) and len(entry0) >= 3 else None
        txt1 = entry1[2] if isinstance(entry1, (list, tuple)) and len(entry1) >= 3 else None
        token_text = txt0 if txt0 is not None else txt1

        abs_diff: Optional[float] = None
        if lp0 is not None and lp1 is not None:
            abs_diff = abs(float(lp0) - float(lp1))

        entry_top0 = top_logprobs0[i + 1] if i + 1 < len(top_logprobs0) else None
        entry_top1 = top_logprobs1[i + 1] if i + 1 < len(top_logprobs1) else None
        top1_id0, token_to_lp0 = _parse_topk_entry(entry_top0)
        top1_id1, token_to_lp1 = _parse_topk_entry(entry_top1)

        # rank (1-based) among returned top-k list (if present)
        def _rank(entry_top: Any) -> Optional[int]:
            if entry_top is None or not isinstance(entry_top, list):
                return None
            for r, item in enumerate(entry_top, start=1):
                if isinstance(item, (list, tuple)) and len(item) >= 2 and int(item[1]) == int(token_id):
                    return r
            return None

        rank0 = _rank(entry_top0)
        rank1 = _rank(entry_top1)

        stats.append(
            ForcedTokenStat(
                idx=i,
                token_id=int(token_id),
                token_text=token_text,
                logprob0=None if lp0 is None else float(lp0),
                logprob1=None if lp1 is None else float(lp1),
                abs_diff=abs_diff,
                rank0=rank0,
                rank1=rank1,
                top1_id0=top1_id0,
                top1_id1=top1_id1,
            )
        )

    return stats


def _format_token(token_text: Optional[str], token_id: int) -> str:
    if token_text is None:
        return f"{token_id}"
    # escape newlines for readability
    safe = token_text.replace("\n", "\\n").replace("\r", "\\r")
    return f"{token_id}({safe})"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url0", default="http://127.0.0.1:30000")
    p.add_argument("--url1", default="http://127.0.0.1:30001")
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--prompt-file", action="append", default=[])
    p.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Tokenize each prompt as a single-turn chat via HF tokenizer.apply_chat_template, "
        "then call /generate with input_ids (matches /v1/chat/completions prompt formatting).",
    )
    p.add_argument(
        "--chat-system",
        default=None,
        help="System prompt for --use-chat-template (omit to send user-only).",
    )
    p.add_argument(
        "--chat-template-kwargs-json",
        default=None,
        help="JSON dict passed to tokenizer.apply_chat_template (e.g. '{\"thinking\": true}').",
    )
    p.add_argument(
        "--tokenizer-path",
        default=None,
        help="HF tokenizer path for --use-chat-template (defaults to /get_model_info.tokenizer_path from url0).",
    )
    p.add_argument(
        "--print-server-info-diff",
        action="store_true",
        help="Fetch /get_server_info from both servers and print a compact diff.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed (maps to sampling_params.sampling_seed for /generate)",
    )
    p.add_argument("--top-logprobs", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-3, help="logprob abs-diff tolerance")
    p.add_argument("--runs", type=int, default=1, help="repeat generation runs to probe nondeterminism")
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument(
        "--reference",
        choices=("url0", "url1"),
        default="url0",
        help="which server's greedy completion to teacher-force",
    )
    args = p.parse_args()

    prompts: List[str] = []
    prompts.extend(args.prompt)
    for fp in args.prompt_file:
        with open(fp, "r", encoding="utf-8") as f:
            prompts.append(f.read())
    prompts = [x for x in (s.strip("\n") for s in prompts) if x.strip()]
    if not prompts:
        raise SystemExit("No prompt provided. Use --prompt or --prompt-file.")

    sess = _session()

    # Basic connectivity check
    model_info0 = None
    for url in (args.url0, args.url1):
        try:
            info = _get_json(sess, url, "/get_model_info", timeout_s=args.timeout_s)
            served = info.get("model_path", "<unknown>")
            print(f"[OK] {url} model_path={served}")
            if url == args.url0:
                model_info0 = info
        except Exception as e:
            print(f"[ERR] cannot reach {url}: {e}")
            return 2

    if args.print_server_info_diff:
        try:
            _print_server_info_diff(sess, url0=args.url0, url1=args.url1, timeout_s=args.timeout_s)
        except Exception as e:
            print(f"[WARN] failed to diff /get_server_info: {e}")

    tokenizer = None
    chat_template_kwargs = None
    if args.use_chat_template:
        tok_path = args.tokenizer_path
        if tok_path is None:
            if not model_info0:
                raise RuntimeError("Missing model_info from url0; cannot infer tokenizer_path")
            tok_path = model_info0.get("tokenizer_path")
        if not tok_path:
            raise RuntimeError("--use-chat-template requires --tokenizer-path or server /get_model_info.tokenizer_path")
        if args.chat_template_kwargs_json:
            chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
            if not isinstance(chat_template_kwargs, dict):
                raise TypeError("--chat-template-kwargs-json must be a JSON object/dict")
        tokenizer = _load_hf_tokenizer(str(tok_path))

    gen_sampling = {
        "temperature": float(args.temperature),
        "max_new_tokens": int(args.max_new_tokens),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "min_p": float(args.min_p),
        "repetition_penalty": float(args.repetition_penalty),
        "presence_penalty": float(args.presence_penalty),
        "frequency_penalty": float(args.frequency_penalty),
    }
    if args.seed is not None:
        gen_sampling["sampling_seed"] = int(args.seed)

    # Teacher-forcing request: we still set max_new_tokens=1 (not 0) because
    # when speculative decoding is enabled on a server, prefill-only optimizations
    # are disabled. max_new_tokens=0 would immediately hit FINISH_LENGTH(0) and
    # can lead to confusing outputs. We ignore the extra generated token.
    eval_sampling = {
        **gen_sampling,
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
    }

    for pi, prompt in enumerate(prompts):
        print("\n" + "=" * 80)
        print(f"[Prompt {pi}] {prompt[:1200]}")

        if args.use_chat_template:
            assert tokenizer is not None
            prompt_ids = _chat_prompt_ids(
                tokenizer,
                system=args.chat_system,
                user=prompt,
                chat_template_kwargs=chat_template_kwargs,
            )
        else:
            prompt_ids0 = tokenize(sess, args.url0, prompt, timeout_s=args.timeout_s)
            prompt_ids1 = tokenize(sess, args.url1, prompt, timeout_s=args.timeout_s)
            if prompt_ids0 != prompt_ids1:
                print(
                    f"[WARN] tokenize mismatch between servers: len0={len(prompt_ids0)} len1={len(prompt_ids1)}"
                )
            prompt_ids = prompt_ids0
        print(f"[Info] prompt_tokens={len(prompt_ids)}")

        # Run generation a few times to probe nondeterminism.
        outputs0: List[List[int]] = []
        outputs1: List[List[int]] = []
        for r in range(args.runs):
            out0 = generate(
                sess,
                args.url0,
                input_ids=prompt_ids,
                sampling_params=gen_sampling,
                timeout_s=args.timeout_s,
            )
            out1 = generate(
                sess,
                args.url1,
                input_ids=prompt_ids,
                sampling_params=gen_sampling,
                timeout_s=args.timeout_s,
            )
            o0 = out0.get("output_ids") or []
            o1 = out1.get("output_ids") or []
            outputs0.append([int(x) for x in o0])
            outputs1.append([int(x) for x in o1])

            mm = _first_mismatch(outputs0[-1], outputs1[-1])
            status = "MATCH" if mm is None else f"DIFF@{mm}"
            print(
                f"[Gen run {r}] {status} len0={len(outputs0[-1])} len1={len(outputs1[-1])}"
            )

        uniq0 = {tuple(x) for x in outputs0}
        uniq1 = {tuple(x) for x in outputs1}
        if len(uniq0) > 1 or len(uniq1) > 1:
            print(
                f"[Nondet] url0_unique={len(uniq0)} url1_unique={len(uniq1)} "
                "(same prompt, same params, multiple runs)"
            )

        ref_output_ids = outputs0[-1] if args.reference == "url0" else outputs1[-1]
        ref_from = args.url0 if args.reference == "url0" else args.url1
        print(f"[Ref] teacher-force completion from {ref_from} ({len(ref_output_ids)} tokens)")

        # Teacher-force evaluation: compute logprobs/top-k for the SAME token sequence.
        eval_ids = list(prompt_ids) + list(ref_output_ids)
        logprob_start_len = len(prompt_ids) - 1

        eval0 = generate(
            sess,
            args.url0,
            input_ids=eval_ids,
            sampling_params=eval_sampling,
            return_logprob=True,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=int(args.top_logprobs) if args.top_logprobs is not None else None,
            return_text_in_logprobs=True,
            timeout_s=args.timeout_s,
        )
        eval1 = generate(
            sess,
            args.url1,
            input_ids=eval_ids,
            sampling_params=eval_sampling,
            return_logprob=True,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=int(args.top_logprobs) if args.top_logprobs is not None else None,
            return_text_in_logprobs=True,
            timeout_s=args.timeout_s,
        )

        meta0 = eval0.get("meta_info", {})
        meta1 = eval1.get("meta_info", {})

        forced = _forced_stats_for_ref_completion(
            prompt_len=len(prompt_ids),
            ref_output_ids=ref_output_ids,
            eval_meta0=meta0,
            eval_meta1=meta1,
        )

        diffs = [x.abs_diff for x in forced if x.abs_diff is not None and math.isfinite(x.abs_diff)]
        if diffs:
            max_abs = max(diffs)
            mean_abs = statistics.mean(diffs)
            p99 = statistics.quantiles(diffs, n=100)[98] if len(diffs) >= 100 else None
            bad = sum(1 for d in diffs if d > args.tol)
            print(
                f"[Logprob] mean_abs_diff={mean_abs:.3e} max_abs_diff={max_abs:.3e}"
                + (f" p99={p99:.3e}" if p99 is not None else "")
                + f"  >tol({args.tol:g})={bad}/{len(diffs)}"
            )
        else:
            print("[Logprob] No comparable logprobs returned (all None?)")

        # If outputs differed, print the first mismatch token and its local stats.
        mm = _first_mismatch(outputs0[-1], outputs1[-1])
        if mm is not None:
            print(f"[Mismatch] greedy outputs differ at token index {mm}")
            # Show a small window around the mismatch in teacher-forced stats.
            w0 = max(0, mm - 2)
            w1 = min(len(forced), mm + 5)
            for s in forced[w0:w1]:
                mark = ">>" if s.idx == mm else "  "
                tok = _format_token(s.token_text, s.token_id)
                print(
                    f"{mark} step={s.idx:4d} tok={tok:>16} "
                    f"lp0={s.logprob0!s:>10} lp1={s.logprob1!s:>10} "
                    f"abs={s.abs_diff!s:>10} "
                    f"rank0={s.rank0!s:>4} rank1={s.rank1!s:>4} "
                    f"top1_0={s.top1_id0!s:>8} top1_1={s.top1_id1!s:>8}"
                )
        else:
            # Still print a quick sanity on top-1 agreement for the reference tokens.
            top1_disagree = sum(
                1
                for s in forced
                if (s.top1_id0 is not None and s.top1_id1 is not None and s.top1_id0 != s.top1_id1)
            )
            print(f"[Top1] top1 disagreement positions={top1_disagree}/{len(forced)} (among returned top-k)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
