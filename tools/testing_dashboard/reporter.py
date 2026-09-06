"""Best-effort metadata transport: no reporting exception may become a test failure."""
import json
import sys
import threading
import urllib.request

class Reporter:
    def __init__(self, address, run):
        self.address = address.rstrip("/")
        self.run = run
        self.failed = False
        self.error = None
        self.lock = threading.Lock()

    def send(self, event, **values):
        with self.lock:
            if self.failed: return False
            try:
                body = json.dumps({"event":event,"run":self.run,**values}, allow_nan=False).encode()
                request = urllib.request.Request(self.address + "/testing/api/events", data=body,
                                                 headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request, timeout=1) as response:
                    result = json.load(response)
                if not result.get("accepted"): raise ValueError("event not accepted")
                return True
            except Exception as error:
                self.failed = True; self.error = str(error)
                # Never warnings.warn(): users may run pytest with -W error.
                print("gffx testing reporting disabled: " + self.error, file=sys.stderr)
                return False
