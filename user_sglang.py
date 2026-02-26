import requests
import json

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


if __name__ == "__main__":
    # 例子，演示使用开关
    prompt="""
Who are you?
    """

    # # 设置 use_chat_template 为 True，使用带系统提示的 chat 模式
    # out, raw = lm_completion(prompt, use_chat_template=True)
    # print("=== Chat completion ===")
    # print(out)

    # 设置 use_chat_template 为 False，使用原始 prompt    
    # Match HF generate() example: greedy + mild repetition penalty.
    out, raw = lm_completion(prompt, max_new_tokens=128, repetition_penalty=1.00, use_chat_template=True)
    print("=== LM completion ===")
    print(out)
