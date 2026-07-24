"""
K3 — Ngày 1: Khám Phá LLM API (Phiên bản hỗ trợ Gemini Provider)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn chạy:
    1. Đảm bảo đã thiết lập GEMINI_API_KEY trong file .env
    2. Chạy thử demo: python template_gemini.py
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

# Nạp API key từ file .env
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính cho Gemini (USD / 1K token)
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gemini-3.6-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-3.5-flash-lite": {"input": 0.0000375, "output": 0.00015},
}

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_MINI_MODEL = os.getenv("GEMINI_MINI_MODEL", "gemini-3.5-flash-lite")


# ===========================================================================
# PART 1 — API CƠ BẢN
# ===========================================================================

def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi Gemini Chat Completions API qua OpenAI SDK compatibility, 
    trả về nội dung phản hồi + độ trễ.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    if latency <= 0:
        latency = 0.0001
        
    response_text = response.choices[0].message.content
    return response_text, latency


def call_gemini_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gemini-1.5-flash — nhanh hơn và rẻ hơn.
    """
    return call_gemini(
        prompt,
        model=GEMINI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )


def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.
    """
    pro_text, pro_latency = call_gemini(prompt, model=GEMINI_MODEL)
    flash_text, flash_latency = call_gemini_mini(prompt)
    
    # Ước lượng thô: 0.75 từ ≈ 1 token
    pro_tokens = len(pro_text.split()) / 0.75
    pro_cost = (pro_tokens / 1000) * PRICING_PER_1K_TOKENS[GEMINI_MODEL]["output"]
    
    return {
        "gemini_pro_response": pro_text,
        "gemini_flash_response": flash_text,
        "gemini_pro_latency": pro_latency,
        "gemini_flash_latency": flash_latency,
        "gemini_pro_cost_estimate": pro_cost,
    }


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN
# ===========================================================================

def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với system prompt và user prompt.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    if latency <= 0:
        latency = 0.0001
        
    response_text = response.choices[0].message.content
    return response_text, latency


def count_tokens(text: str, model: str = GEMINI_MODEL) -> int:
    """
    Đếm số token của một đoạn text. Vì tiktoken không hỗ trợ trực tiếp các model 
    của Google Gemini, chúng ta sẽ dùng ước lượng fallback: trung bình 1 token ≈ 4 ký tự.
    """
    return max(1, len(text) // 4)


def estimate_cost(prompt: str, response: str, model: str = GEMINI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token ước tính.
    """
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS[GEMINI_MINI_MODEL])
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN
# ===========================================================================

def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming với model gemini-1.5-flash.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    history = []
    print("=== Gemini Streaming Chatbot (Gõ 'quit' hoặc 'exit' để thoát) ===")
    while True:
        user_msg = input("\nBạn: ")
        if user_msg.strip().lower() in ("quit", "exit"):
            break
        
        messages = history + [{"role": "user", "content": user_msg}]
        stream = client.chat.completions.create(
            model=GEMINI_MINI_MODEL,
            messages=messages,
            stream=True,
        )
        
        print("Gemini: ", end="", flush=True)
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()
        
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]


def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Thực hiện gọi fn() với cơ chế retry exponential backoff.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise e
            time.sleep(base_delay * (2 ** attempt))


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH
# ===========================================================================

def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh sử dụng Gemini model.
    """
    if get_input is None:
        get_input = input
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0
    
    while True:
        if max_turns is not None and num_turns >= max_turns:
            break
        try:
            user_msg = get_input()
        except EOFError:
            break
        if user_msg.strip().lower() in ("quit", "exit"):
            break
            
        messages = [{"role": "system", "content": persona}] + history + [{"role": "user", "content": user_msg}]
        
        def call_api():
            return client.chat.completions.create(
                model=GEMINI_MINI_MODEL,
                messages=messages,
                stream=True,
            )
            
        stream = retry_with_backoff(call_api)
        
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()
        
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]
        
        num_turns += 1
        total_tokens += count_tokens(user_msg) + count_tokens(reply)
        total_cost += estimate_cost(user_msg, reply)["total_cost"]
        
    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history
    }


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần GEMINI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("LỖI: Chưa cấu hình GEMINI_API_KEY trong file .env")
        exit(1)
        
    print("=== So sánh model Gemini ===")
    try:
        result = compare_models(
            "Giải thích sự khác biệt giữa AI và Machine Learning trong một câu ngắn."
        )
        for key, value in result.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")
        print("Vui lòng kiểm tra lại GEMINI_API_KEY trong file .env hoặc kết nối mạng.")

    print("\n=== Trợ lý CLI Gemini (gõ 'quit' để thoát) ===")
    try:
        stats = run_assistant(
            persona="Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.",
            max_turns=2
        )
        print("\n--- Thống kê phiên chat ---")
        for key, value in stats.items():
            if key != "history":
                print(f"{key}: {value}")
    except Exception as e:
        print(f"Lỗi khi chạy Trợ lý CLI: {e}")
