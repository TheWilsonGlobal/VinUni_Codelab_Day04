# Starter Code — Lab #4

> Dành cho học viên VinUni AI Product Labs.

## Cách Sử Dụng

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Hoàn thiện bài lab:**
   - Mở `tools.py` → Hoàn thiện 2 hàm tool + JSON Schemas.
   - Mở `template.py` → Hoàn thiện 4 TODO:
     - TODO 1: System Prompt
     - TODO 2: ChatbotBaseline
     - TODO 3: Intent Detection
     - TODO 4: Agent Loop

3. **Chạy thử:**
   ```bash
   python3 template.py
   ```

4. **Kiểm thử autograder:**
   ```bash
   python3 -m pytest ../autograder/test_agent.py -v
   ```

## Lưu Ý
- Code chạy ở chế độ **Mock Simulator** mặc định (không cần API key). Autograder luôn chạy ở chế độ này để kết quả ổn định.
- Chạy với Gemini (function calling):
  1. Tạo file `.env` ở thư mục gốc lab (cạnh `student_guide.md`), **không commit** file này:
     ```
     GEMINI_API_KEY=your_key
     ```
  2. Chạy: `LLM_MODE=gemini python3 template.py` (PowerShell: `$env:LLM_MODE="gemini"; python template.py`)
  3. Đổi model (tùy chọn): biến `GEMINI_MODEL`, mặc định `gemini-3.5-flash-lite`.
- Free tier của Gemini giới hạn khoảng 15 request/phút. Code tự thử lại khi gặp lỗi 429/503.
- Kiểm thử với Gemini: `LLM_MODE=gemini python3 -m pytest ../autograder/test_agent.py -v`
