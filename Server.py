import gc
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header('Content-type', 'application/json')
        self.send_header("Connection", "close")
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0')
        self.send_header('Expires', '0')
        self.end_headers()
        return

    def do_GET(self):
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


def run(server_class=ThreadedHTTPServer, handler_class=S, port=8080):
    from sys import argv
    if len(argv) == 2:
        port = int(argv[1])

    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        gc.collect()
        print('traceback_details: ', traceback_details)
        print("interrupted")
        sys.exit(0)
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')
