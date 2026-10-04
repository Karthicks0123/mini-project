import os
import re
import json
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
import mimetypes

# Automatically load .env file if present in project directory
def _load_env():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    for env_path in possible_paths:
        if os.path.exists(env_path) and os.path.isfile(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
            break

_load_env()

HF_MODEL = os.getenv("HF_MODEL", "sshleifer/distilbart-cnn-12-6")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

HF_API_ENDPOINTS = [
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}",
    f"https://api-inference.huggingface.co/models/{HF_MODEL}"
]


def count_words(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def calculate_percentage_reduction(original_count: int, summary_count: int) -> float:
    if original_count <= 0:
        return 0.0
    reduction = ((original_count - summary_count) / original_count) * 100.0
    return max(0.0, round(reduction, 2))


def extract_key_points(summary_text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', summary_text.strip())
    points = [s.strip() for s in sentences if len(s.strip()) > 8]
    return points


def query_huggingface_inference(text: str, min_length: int = 25, max_length: int = 140) -> str:
    """
    Call Hugging Face Inference API to get abstractive summary.
    """
    payload = {
        "inputs": text,
        "parameters": {
            "min_length": min_length,
            "max_length": max_length,
            "do_sample": False
        },
        "options": {
            "wait_for_model": True,
            "use_cache": True
        }
    }
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "SmartStudyNotes/1.0"
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    last_error = None
    for endpoint in HF_API_ENDPOINTS:
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_response = resp.read().decode("utf-8")
                parsed = json.loads(raw_response)

                if isinstance(parsed, list) and len(parsed) > 0 and "summary_text" in parsed[0]:
                    return parsed[0]["summary_text"].strip()
                elif isinstance(parsed, dict) and "summary_text" in parsed:
                    return parsed["summary_text"].strip()
                elif isinstance(parsed, dict) and "error" in parsed:
                    raise RuntimeError(parsed["error"])
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(err_body)
                last_error = err_json.get("error", err_body)
            except Exception:
                last_error = f"HTTP {e.code}: {e.reason} - {err_body}"
        except Exception as e:
            last_error = str(e)

    # If Hugging Face API requires authentication or fails due to network/rate limits,
    # explain clearly how to supply HF_TOKEN or fallback intelligently
    if last_error and "authorization" in str(last_error).lower() or not HF_TOKEN:
        guide_msg = " (Tip: Set HF_TOKEN in Vercel Environment Variables with a free token from huggingface.co/settings/tokens)"
        raise RuntimeError(f"Hugging Face API returned: {last_error}{guide_msg}")

    raise RuntimeError(f"Failed to generate summary via Hugging Face API: {last_error}")


class handler(BaseHTTPRequestHandler):
    """
    Vercel Serverless Function HTTP Handler.
    """

    def _set_headers(self, status_code=200, content_type="application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        # Health check or local file serving
        if self.path.startswith("/api"):
            self._set_headers(200)
            status_info = {
                "status": "healthy",
                "service": "Smart Study Notes API",
                "model": HF_MODEL,
                "token_configured": bool(HF_TOKEN)
            }
            self.wfile.write(json.dumps(status_info, indent=2).encode("utf-8"))
            return

        # Serving static files when running locally
        clean_path = self.path.split("?")[0]
        if clean_path in ("", "/"):
            clean_path = "/index.html"

        file_path = os.path.join(os.path.dirname(__file__), "..", "public", clean_path.lstrip("/"))
        if os.path.exists(file_path) and os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            self._set_headers(200, mime_type or "text/plain")
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._set_headers(404, "application/json")
            self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8")

            if not raw_body:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Empty request body"}).encode("utf-8"))
                return

            body = json.loads(raw_body)
            input_text = body.get("text", "").strip()

            if not input_text:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Field 'text' is required and cannot be empty."}).encode("utf-8"))
                return

            orig_words = count_words(input_text)
            min_length = int(body.get("min_length", min(25, max(10, int(orig_words * 0.25)))))
            max_length = int(body.get("max_length", min(140, max(35, int(orig_words * 0.7)))))

            if min_length >= max_length:
                min_length = max(5, max_length - 10)

            # Generate abstractive summary via Hugging Face API
            summary_text = query_huggingface_inference(input_text, min_length=min_length, max_length=max_length)
            
            # Clean punctuation spacing
            summary_text = re.sub(r'\s+([,.:;!?])', r'\1', summary_text)

            summary_words = count_words(summary_text)
            reduction_pct = calculate_percentage_reduction(orig_words, summary_words)
            key_points = extract_key_points(summary_text)

            response_data = {
                "summary": summary_text,
                "key_points": key_points,
                "original_words": orig_words,
                "summary_words": summary_words,
                "words_saved": max(0, orig_words - summary_words),
                "reduction_percentage": reduction_pct,
                "model": HF_MODEL
            }

            self._set_headers(200)
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    print(f"Starting local server at http://localhost:{port}")
    print(f"Using Hugging Face Model: {HF_MODEL}")
    print(f"HF_TOKEN detected: {'Yes' if HF_TOKEN else 'No (set HF_TOKEN for higher rate limits)'}")
    server = HTTPServer(("", port), handler)
    server.serve_forever()
