import http.server
import socketserver
import json
import os
import sys

# Path to the DLL metadata
METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "users", "user_abc123", "metadata_links.json")

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class VisualizerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == '/api/memory':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    data = f.read()
                self.wfile.write(data.encode())
            else:
                self.wfile.write(json.dumps({"nodes": {}, "head_id": None}).encode())
        else:
            return super().do_GET()

if __name__ == "__main__":
    print(f"Starting Visualizer at http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), VisualizerHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            sys.exit(0)
