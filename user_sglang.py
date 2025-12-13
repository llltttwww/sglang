import requests
import json

SERVER_URL = "http://127.0.0.1:30000"

def chat_with_sglang(
    messages,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
):
    url = f"{SERVER_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": "qwen3-next-kimi",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
    resp.raise_for_status()
    data = resp.json()
    # 返回格式类似：
    # {
    #   "choices": [
    #       {
    #           "message": {"role": "assistant", "content": "..."},
    #           ...
    #       }
    #   ]
    # }
    content = data["choices"][0]["message"]["content"]
    return content, data


# if __name__ == "__main__":
#     messages = [
#         {"role": "system", "content": "You're an AI assistant. "},
#         {"role": "user", "content": "Hello! Nice to meet you! My name is Jone Smith. What's your name?"},
#     ]
#     reply, raw = chat_with_sglang(messages)
#     print("=== Assistant ===")
#     print(reply)


import requests
import json

SERVER_URL = "http://127.0.0.1:30000"

def lm_completion(
    prompt: str,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
):
    url = f"{SERVER_URL}/v1/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "qwen3-next-kimi",
        "prompt": prompt,              # 这里是原始 prompt，完全不经过 chat_template
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    return text, data

if __name__ == "__main__":
    # 想要“空 messages”的效果，就等价于从新文档开头开始续写：
    prompt = "<|begin_text|> I'm Peter, a man. "   # 或者随便写点你自己的 prefix
    out, raw = lm_completion(prompt)
    print("=== LM completion ===")
    print(out)
