"""
Lab #4: System Prompt Engineering & Tool Calling Engine

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + Tool Schemas.

Chế độ chọn bằng biến môi trường LLM_MODE:
  - mock   (mặc định): intent bằng từ khóa, không cần API key, kết quả ổn định.
  - gemini : Gemini làm bộ não của agent qua function calling.
             Cần GEMINI_API_KEY trong .env ở thư mục gốc lab.
             Model: GEMINI_MODEL (mặc định gemini-3.5-flash-lite).

Định nghĩa "iterations": số vòng gọi tool của agent (tối thiểu 1).
  Một vòng = một lượt thực thi tool(s). Trong một vòng có thể gọi nhiều tool song song.
"""

import json
import os
import re
import sys
import time
import unicodedata
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from google import genai
    from google.genai import types
except ImportError:  # SDK chưa cài: vẫn chạy được chế độ mock
    genai = None
    types = None

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

LLM_MODE = os.environ.get("LLM_MODE", "mock").strip().lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
## PERSONA
- Tên: VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.
- Vai trò: Tư vấn sản phẩm VinFast (xe điện), dịch vụ Vinpearl (nghỉ dưỡng), và ghi nhận yêu cầu hỗ trợ khách hàng.
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác. Luôn trả lời bằng tiếng Việt.

## AVAILABLE TOOLS
{tools}

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm, giá hay chính sách. Thông tin sản phẩm và giá PHẢI lấy từ search_product_catalog.
2. Khi khách báo lỗi, phản hồi hoặc cần hỗ trợ, PHẢI gọi submit_support_ticket. Không tự tạo mã ticket.
3. Chỉ gọi tool khi cần. Với câu hỏi chính sách, chỉ trả lời theo mục KNOWLEDGE. Nếu KNOWLEDGE không có thông tin, nói chưa có dữ liệu và hướng khách liên hệ tổng đài Vingroup.
4. Nếu tool trả về danh sách rỗng, bắt đầu bằng đúng câu "Rất tiếc, không tìm thấy sản phẩm phù hợp." Không đoán, không nêu giá hay sản phẩm không có trong kết quả tool.
5. Nếu một tin nhắn có nhiều yêu cầu, xử lý tất cả trong cùng một lượt.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về hệ sinh thái Vingroup: VinFast, Vinpearl, VinWonders và các dịch vụ liên quan.
- Từ chối lịch sự các yêu cầu ngoài phạm vi (thời tiết, lập trình, tin tức...) và gợi ý khách liên hệ kênh phù hợp.

## KNOWLEDGE
- Pin xe điện VinFast VF 5 Plus được bảo hành 10 năm (theo danh mục sản phẩm).

## OUTPUT CONTRACT
- Khi cần dữ liệu, gọi tool bằng function calling. KHÔNG viết tên tool, tham số hay Thought/Action ra văn bản.
- Câu trả lời cuối cùng: ngắn gọn, rõ ràng, tiếng Việt, không có nhãn Thought/Action/Observation/Final Answer.
"""


def build_system_prompt() -> str:
    """Điền danh sách tool vào placeholder {tools}. Dùng replace, không dùng .format()."""
    tools_text = "\n".join(f"- {t['name']}: {t['description']}" for t in TOOL_DEFINITIONS)
    return SYSTEM_PROMPT.replace("{tools}", tools_text)


# ═══════════════════════════════════════════════════════════════════════════
# Dữ liệu cho chế độ mock
# ═══════════════════════════════════════════════════════════════════════════

NOT_FOUND_MSG = "Rất tiếc, không tìm thấy sản phẩm phù hợp."
OUT_OF_SCOPE_MSG = (
    "Xin lỗi, tôi chỉ hỗ trợ thông tin về sản phẩm và dịch vụ của hệ sinh thái Vingroup "
    "(VinFast, Vinpearl, VinWonders)."
)
FAQ_ANSWERS = {
    "bảo hành pin": (
        "Theo thông tin sản phẩm hiện có, pin xe điện VinFast VF 5 Plus được bảo hành 10 năm. "
        "Chi tiết chính sách, vui lòng liên hệ đại lý VinFast gần nhất."
    ),
}
FAQ_DEFAULT_MSG = "Xin lỗi, tôi chưa có thông tin về câu hỏi này. Bạn vui lòng liên hệ tổng đài Vingroup để được hỗ trợ."

CATALOG_WORDS = ("giá", "dưới", "xem", "tìm", "mua", "có xe điện nào", "có resort")
CATALOG_DU_LICH_WORDS = ("resort", "nghỉ", "du lịch", "vinpearl", "vinwonders", "phòng")
CATALOG_XE_WORDS = ("xe", "vinfast", "vf ")
TICKET_WORDS = ("lỗi", "hỏng", "ẩm mốc", "phản hồi", "khiếu nại", "ghi nhận", "hỗ trợ", "xử lý", "tôi tên", "tên tôi")
FAQ_WORDS = ("chính sách", "bảo hành", "bao lâu", "như thế nào", "là gì")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _has_any(text_lower: str, words) -> bool:
    return any(w in text_lower for w in words)


def _extract_max_price(text_lower: str) -> Optional[int]:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(triệu|tỷ)", text_lower)
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    unit = 1_000_000 if m.group(2) == "triệu" else 1_000_000_000
    return int(value * unit)


def _extract_customer_name(text: str) -> Optional[str]:
    m = re.search(r"tên\s+(?:tôi\s+)?(?:là\s+)?([^,.;]+)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def _detect_priority(text_lower: str) -> str:
    # Thứ tự quan trọng: "không gấp" phải được xét trước "gấp".
    if "không gấp" in text_lower:
        return "low"
    if _has_any(text_lower, ("nghiêm trọng", "gấp", "khẩn")):
        return "high"
    if "trung bình" in text_lower:
        return "medium"
    if _has_any(text_lower, ("thấp", "nhẹ")):
        return "low"
    return "medium"


def detect_intent(user_input: str) -> Dict[str, Any]:
    """
    TODO 3: Xác định intent bằng từ khóa. Mỗi cờ được xét độc lập,
    nên một câu có thể kích hoạt cả catalog và ticket cùng lúc.
    """
    text = _normalize(user_input)
    lower = text.lower()

    category = None
    if _has_any(lower, CATALOG_DU_LICH_WORDS):
        category = "du_lich"
    elif _has_any(lower, CATALOG_XE_WORDS):
        category = "xe_dien"

    needs_catalog = category is not None and _has_any(lower, CATALOG_WORDS)
    needs_ticket = _has_any(lower, TICKET_WORDS)
    is_faq = (not needs_catalog) and (not needs_ticket) and _has_any(lower, FAQ_WORDS)
    out_of_scope = not (needs_catalog or needs_ticket or is_faq)

    intents: Dict[str, Any] = {
        "needs_catalog": needs_catalog,
        "needs_ticket": needs_ticket,
        "is_faq": is_faq,
        "out_of_scope": out_of_scope,
    }
    if needs_catalog:
        intents["category"] = category
        intents["max_price"] = _extract_max_price(lower)
    if needs_ticket:
        intents["customer_name"] = _extract_customer_name(text) or "Khách hàng"
        intents["priority"] = _detect_priority(lower)
        intents["issue_description"] = _extract_issue(text)
    return intents


def _extract_issue(text: str) -> str:
    """Lấy các câu có từ khóa ticket, bỏ phần tên khách."""
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    kept = [s for s in sentences if _has_any(s.lower(), TICKET_WORDS)] or sentences
    issue = ". ".join(kept)
    issue = re.sub(r"tên\s+(?:tôi\s+)?(?:là\s+)?[^,.;]+[,.;]\s*", "", issue, flags=re.IGNORECASE)
    return issue.strip() or text.strip()


def _format_price(vnd: int) -> str:
    return f"{vnd:,}".replace(",", ".") + " VNĐ"


def _format_catalog(results: List[Dict[str, Any]]) -> str:
    if not results or "error" in results[0]:
        return NOT_FOUND_MSG if not results else "Hệ thống tra cứu đang gặp sự cố, vui lòng thử lại sau."
    lines = [f"- {p['name']}: {_format_price(p['price_vnd'])} ({p['availability']})" for p in results]
    return "Tôi tìm được các sản phẩm phù hợp:\n" + "\n".join(lines)


def _format_ticket(ticket: Dict[str, Any]) -> str:
    return (
        f"Ticket {ticket['ticket_id']} đã được tạo cho {ticket['customer_name']} "
        f"(mức độ: {ticket['priority']}). Bộ phận hỗ trợ sẽ liên hệ với bạn sớm."
    )


def _faq_answer(user_input: str) -> str:
    lower = _normalize(user_input).lower()
    for keyword, answer in FAQ_ANSWERS.items():
        if keyword in lower:
            return answer
    return FAQ_DEFAULT_MSG


# ═══════════════════════════════════════════════════════════════════════════
# Gemini helpers
# ═══════════════════════════════════════════════════════════════════════════

_client = None


def _gemini_ready() -> bool:
    return genai is not None and bool(os.environ.get("GEMINI_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _generate(**kwargs):
    """Gọi Gemini, tự thử lại khi bị 429/503 (free tier giới hạn ~15 request/phút)."""
    from google.genai import errors

    client = _get_client()
    for attempt in range(1, 6):
        try:
            return client.models.generate_content(model=GEMINI_MODEL, **kwargs)
        except errors.APIError as e:
            if e.code not in (429, 503) or attempt == 5:
                raise
            time.sleep(min(15 * attempt, 45))


def _gemini_declarations():
    """Đổi TOOL_DEFINITIONS sang FunctionDeclaration của Gemini (JSON Schema chữ thường)."""
    return [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters_json_schema=t["parameters"],
        )
        for t in TOOL_DEFINITIONS
    ]


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Gọi Gemini 1 lượt, không tool, không system prompt → quan sát hallucination.
        if LLM_MODE == "gemini" and _gemini_ready():
            try:
                response = _generate(contents=user_input)
                return {
                    "answer": (response.text or "").strip(),
                    "tool_calls": [],
                    "status": "success",
                    "mode": f"gemini_baseline:{GEMINI_MODEL}",
                }
            except Exception as e:
                return {"answer": f"Lỗi gọi Gemini: {e}", "tool_calls": [], "status": "error", "mode": "gemini_baseline"}

        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline",
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        if LLM_MODE == "gemini":
            if _gemini_ready():
                return self._run_gemini(user_input)
            print("[WARN] LLM_MODE=gemini nhưng thiếu GEMINI_API_KEY hoặc SDK. Chuyển sang mock.")
        return self._run_mock(user_input)

    # ---- helpers -----------------------------------------------------------

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        fn = TOOL_MAP.get(name)
        if fn is None:
            result: Any = {"error": f"Tool không tồn tại: {name}"}
        else:
            try:
                result = fn(**args)
            except Exception as e:  # tool lỗi không được làm sập agent
                result = {"error": str(e)}
        self.trace.append({"step": "tool", "tool": name, "args": args, "result": result})
        return result

    # ---- mock mode ---------------------------------------------------------

    def _run_mock(self, user_input: str) -> Dict[str, Any]:
        # TODO 3: Phân tích intent
        intents = detect_intent(user_input)
        self.trace.append({"step": "intent", "intents": intents})

        # TODO 4: Mọi tool được chọn từ intent nên một vòng là đủ.
        parts: List[str] = []
        tool_rounds = 0

        if intents["needs_catalog"]:
            args: Dict[str, Any] = {"category": intents["category"]}
            if intents["max_price"] is not None:
                args["max_price"] = intents["max_price"]
            results = self._call_tool("search_product_catalog", args)
            parts.append(_format_catalog(results))
            tool_rounds += 1

        if intents["needs_ticket"]:
            ticket = self._call_tool("submit_support_ticket", {
                "customer_name": intents["customer_name"],
                "issue_description": intents["issue_description"],
                "priority": intents["priority"],
            })
            if "error" in ticket:
                parts.append(f"Không thể tạo ticket: {ticket['error']}")
            else:
                parts.append(_format_ticket(ticket))
            tool_rounds += 1

        if intents["is_faq"]:
            parts.append(_faq_answer(user_input))
        if intents["out_of_scope"]:
            parts.append(OUT_OF_SCOPE_MSG)

        answer = "\n\n".join(parts)
        self.trace.append({"step": "final", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": max(1, tool_rounds),
            "status": "completed",
            "mode": "mock",
        }

    # ---- gemini mode -------------------------------------------------------

    def _run_gemini(self, user_input: str) -> Dict[str, Any]:
        config = types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            tools=[types.Tool(function_declarations=_gemini_declarations())],
            # Tự xử lý function call để ghi đầy đủ trace.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents: List[Any] = [types.Content(role="user", parts=[types.Part(text=user_input)])]
        self.trace.append({"step": "user", "input": user_input})

        tool_rounds = 0
        while True:
            response = _generate(contents=contents, config=config)
            calls = response.function_calls or []

            if not calls:
                answer = (response.text or "").strip()
                self.trace.append({"step": "final", "answer": answer})
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": max(1, tool_rounds),
                    "status": "completed",
                    "mode": f"gemini:{GEMINI_MODEL}",
                }

            if tool_rounds >= self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": tool_rounds,
                    "status": "max_iterations_reached",
                    "mode": f"gemini:{GEMINI_MODEL}",
                }

            tool_rounds += 1
            # Giữ nguyên content của model (có thought signature) khi gửi lại.
            contents.append(response.candidates[0].content)
            response_parts = []
            for call in calls:
                result = self._call_tool(call.name, dict(call.args or {}))
                response_parts.append(
                    types.Part.from_function_response(name=call.name, response={"result": result})
                )
            contents.append(types.Content(role="user", parts=response_parts))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = " ".join(sys.argv[1:]) or "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."
    print(f"LLM_MODE = {LLM_MODE}")

    print("\n=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Status:", result["status"], "| Iterations:", result["iterations"], "| Mode:", result["mode"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
