from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import socket

# Change this to the folder where your questions.json actually lives
os.chdir("/Users/User/Downloads/Trial 999 copy 6/Combined_Working/Combine trial last 2/Combine  trial/Untitled copy/html/data/")

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

print("🚀 Quiz Server started on port 8000...")
ReusableHTTPServer(('0.0.0.0', 8000), CORSRequestHandler).serve_forever()