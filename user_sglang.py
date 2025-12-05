import requests
import json

SERVER_URL = "http://127.0.0.1:30000"

def chat_with_sglang(
    messages,
    temperature: float = 2.0,
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


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "你是一个AI。"},
        {"role": "user", "content": "Hello。"},
    ]
    reply, raw = chat_with_sglang(messages)
    print("=== Assistant ===")
    print(reply)
