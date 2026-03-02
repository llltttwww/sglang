import requests
import json
import os

SERVER_URL = "http://127.0.0.1:30001"

# Avoid picking up HTTP(S)_PROXY from the environment for local calls.
_SESSION = requests.Session()
_SESSION.trust_env = False

def chat_with_sglang(
    messages,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.0,
    use_chat_template: bool = True,  # 新增开关参数
):
    headers = {
        "Content-Type": "application/json",
    }

    # 如果 use_chat_template 为 True，使用 chat 模式的模板
    if use_chat_template:
        url = f"{SERVER_URL}/v1/chat/completions"
        payload = {
            "model": "qwen3-next-kimi",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            # SGLang defaults separate_reasoning=True; disable it so content stays in one field
            # even if a reasoning parser is enabled on the server.
            "separate_reasoning": False,
        }
    else:
        # 如果 use_chat_template 为 False，走 completions 接口（chat/completions 不接受 prompt 字段）
        url = f"{SERVER_URL}/v1/completions"
        payload = {
            "model": "qwen3-next-kimi",
            "prompt": messages[0]['content'],  # 直接使用用户消息作为原始 prompt
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }

    resp = _SESSION.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    if use_chat_template:
        msg = data["choices"][0]["message"]
        # Some responses may include reasoning_content separately (depending on server config).
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        if reasoning and not content:
            content = reasoning
        elif reasoning:
            content = reasoning + "\n\n" + content
    else:
        content = data["choices"][0]["text"]
    return content, data


def lm_completion(
    prompt: str,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.0,
    use_chat_template: bool = True,  # 新增开关参数
):
    url = f"{SERVER_URL}/v1/completions"
    headers = {"Content-Type": "application/json"}

    # 如果 use_chat_template 为 True，构建 chat 模式的消息
    if use_chat_template:
        messages = [
            {"role": "system", "content": "You are an AI assistant which respond by thinking step by step."},
            {"role": "user", "content": prompt},
        ]
        return chat_with_sglang(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            use_chat_template=use_chat_template,
        )
    else:
        # 如果 use_chat_template 为 False，直接传递原始 prompt
        payload = {
            "model": "qwen3-next-kimi",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }

        resp = _SESSION.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["text"]
    return text, data


def debug_generate_logprobs(
    prompt: str,
    temperature: float = 0.0,
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.0,
    top_logprobs_num: int = 5,
    max_print_tokens: int = 16,
):
    """Call SGLang /generate with logprobs enabled to help debug degenerate outputs."""
    url = f"{SERVER_URL}/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        },
        "return_logprob": True,
        "top_logprobs_num": top_logprobs_num,
        "return_text_in_logprobs": True,
        "stream": False,
    }

    resp = _SESSION.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    print("=== /generate (debug logprobs) ===")
    print(data.get("text", ""))

    meta = data.get("meta_info", {}) or {}
    out_lps = meta.get("output_token_logprobs") or []
    out_top = meta.get("output_top_logprobs") or []

    print("--- output_token_logprobs (first tokens) ---")
    for i, item in enumerate(out_lps[:max_print_tokens]):
        try:
            logprob, token_id, token_text = item
        except Exception:
            print(i, item)
            continue
        if logprob is None:
            lp_str = "null"
        elif isinstance(logprob, (int, float)):
            lp_str = f"{logprob:.6f}"
        else:
            lp_str = str(logprob)
        print(f"{i:02d} id={token_id} logprob={lp_str} token={token_text!r}")

        if i < len(out_top) and out_top[i]:
            top_items = out_top[i]
            def _fmt_top(x):
                try:
                    lp, tid, tt = x
                except Exception:
                    return repr(x)
                if lp is None:
                    lp_s = "null"
                elif isinstance(lp, (int, float)):
                    lp_s = f"{lp:.3f}"
                else:
                    lp_s = str(lp)
                return f"{tid}:{lp_s}:{(tt or '')!r}"

            top_str = ", ".join(_fmt_top(x) for x in top_items[:top_logprobs_num])
            print(f"     top{top_logprobs_num}: {top_str}")

    return data


if __name__ == "__main__":
    # 例子，演示使用开关
    prompt="""
    You are a careful assistant. Follow the format shown in the examples:
    - Write a short step-by-step reasoning.
    - End with a single line that starts with "Final:" and contains only the final answer.

    Example 1
    Question: A box has 3 red balls and 2 blue balls. Two balls are drawn without replacement. What is the probability both are red?
    Reasoning: Total ways to choose 2 from 5 is C(5,2)=10. Favorable ways to choose 2 from 3 red is C(3,2)=3. Probability = 3/10.
    Final: 3/10

    Example 2
    Question: A jacket costs $200. It is discounted by 20% and then an extra $10 is taken off. What is the final price?
    Reasoning: After 20% off, price is 200 * 0.8 = 160. Then subtract 10 to get 150.
    Final: $150

    Now answer:
    Question: If a train travels 180 kilometers in 3 hours at a constant speed, what is its speed in kilometers per hour?
    """

    # # 设置 use_chat_template 为 True，使用带系统提示的 chat 模式
    # out, raw = lm_completion(prompt, use_chat_template=True)
    # print("=== Chat completion ===")
    # print(out)

    # 设置 use_chat_template 为 False，使用原始 prompt    
    # Match HF generate() example: greedy + mild repetition penalty.
    out, raw = lm_completion(prompt, max_new_tokens=512, repetition_penalty=1.00, use_chat_template=False)
    print("=== LM completion ===")
    print(out)

    if os.environ.get("DEBUG_RAW") == "1":
        print("=== Raw response ===")
        print(json.dumps(raw, ensure_ascii=False, indent=2))

    if os.environ.get("DEBUG_LOGPROBS") == "1":
        debug_generate_logprobs(
            prompt.strip(),
            temperature=0.0,
            max_new_tokens=64,
            repetition_penalty=1.0,
            top_logprobs_num=5,
        )
