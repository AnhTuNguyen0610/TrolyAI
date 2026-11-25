# test_llm_config.py — kiểm tra chọn provider qua biến môi trường (không gọi mạng)
import os

from agent_core import tao_llm


def dat_env(**kw):
    for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL",
              "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(k, None)
    os.environ.update(kw)


def base_url(llm):
    return str(llm.openai_api_base)


dat_env(LLM_PROVIDER="gemini", GOOGLE_API_KEY="g")
llm = tao_llm(0.3)
assert "generativelanguage.googleapis.com" in base_url(llm) and llm.model_name == "gemini-2.5-flash"
print("PASS - gemini")

dat_env(LLM_PROVIDER="groq", GROQ_API_KEY="g")
llm = tao_llm(0.3)
assert "api.groq.com" in base_url(llm) and llm.model_name == "llama-3.3-70b-versatile"
print("PASS - groq")

dat_env(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k")
assert "openrouter.ai" in base_url(tao_llm(0.3))
print("PASS - openrouter")

dat_env(LLM_PROVIDER="ollama")
llm = tao_llm(0.3)
assert "localhost:11434" in base_url(llm) and llm.model_name == "llama3.2"
print("PASS - ollama (không cần key)")

dat_env(LLM_PROVIDER="groq", GROQ_API_KEY="g", LLM_MODEL="model-khac")
assert tao_llm(0.3).model_name == "model-khac"
print("PASS - đổi model bằng LLM_MODEL")

dat_env(LLM_PROVIDER="gemini")
try:
    tao_llm(0.3)
    raise AssertionError("phải báo thiếu key")
except RuntimeError as e:
    assert "GOOGLE_API_KEY" in str(e)
print("PASS - báo rõ khi thiếu API key")

dat_env(LLM_PROVIDER="abc")
try:
    tao_llm(0.3)
    raise AssertionError("phải báo provider sai")
except ValueError:
    pass
print("PASS - báo lỗi provider không hợp lệ")

print("\nCấu hình LLM đúng!")
