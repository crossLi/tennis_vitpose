import json
from http.server import BaseHTTPRequestHandler, HTTPServer

TOKEN = "dev-token-123"

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n=int(self.headers.get("Content-Length") or "0")
        raw=self.rfile.read(n).decode("utf-8","ignore")
        tk=self.headers.get("X-AI-Token") or ""
        try: obj=json.loads(raw)
        except Exception as e: obj={"raw":raw,"err":str(e)}
        print("\n=== RECEIVED ===")
        print("path=", self.path)
        print("X-AI-Token=", repr(tk), "✅" if tk==TOKEN else "⚠️ 期望 "+TOKEN)
        print("seq_no=", obj.get("seq_no"), "taskId=", obj.get("taskId"), "stage=", obj.get("stage"), "step=", obj.get("step"))
        print("timestamp=", obj.get("timestamp"), type(obj.get("timestamp")).__name__)
        rs=obj.get("result") or {}
        print("result.stage=", rs.get("stage"), "result.progress=", rs.get("progress"))
        print("—— body 前 220 字 ——"); print(raw[:220])
        self.send_response(200)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(b'{"code":0,"ok":true}')
    def log_message(self,*a,**k): pass

HTTPServer(("0.0.0.0",9999),H).serve_forever()