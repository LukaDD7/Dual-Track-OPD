"""Dedicated HTTP service for FC-OPD student condition scoring."""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Iterator, Sequence

from .conditions import Condition
from .online_batch import OnlineFCOPDSample, OnlineStudentScores, StudentForcedScorer
from .student_scorer import StudentScorer
from .student_scorer_client import _sample_from_dict, _scores_to_dict


class StudentScorerHTTPServer(HTTPServer):
    scorer: StudentForcedScorer


def _handler_for(scorer: StudentForcedScorer) -> type[BaseHTTPRequestHandler]:
    class StudentScorerRequestHandler(BaseHTTPRequestHandler):
        server: StudentScorerHTTPServer

        def log_message(self, format: str, *args: object) -> None:
            import sys

            print(format % args, file=sys.stderr, flush=True)

        def _write_json(self, status: HTTPStatus, payload: object) -> None:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json(HTTPStatus.OK, {"status": "ok"})
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/score":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    raise ValueError("request body is empty")
                body = json.loads(self.rfile.read(content_length))
                raw_samples = body.get("samples")
                if not isinstance(raw_samples, list) or not raw_samples:
                    raise ValueError("samples must be a non-empty list")
                conditions = [Condition(condition) for condition in body.get("conditions", ())]
                if not conditions:
                    raise ValueError("conditions must be non-empty")
                samples = [_sample_from_dict(item) for item in raw_samples]
                result = scorer(samples, conditions)
                responses = result if isinstance(result, list) else [result]
                if len(responses) != len(samples):
                    raise RuntimeError("student scorer response count does not match sample count")
                self._write_json(
                    HTTPStatus.OK,
                    {"responses": [_scores_to_dict(response) for response in responses]},
                )
            except Exception as exc:
                import sys
                import traceback as _tb

                print(f"[student-scorer] ERROR: {exc}", file=sys.stderr, flush=True)
                _tb.print_exc(file=sys.stderr)
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": type(exc).__name__, "message": str(exc)},
                )

    return StudentScorerRequestHandler


def create_student_scorer_server(
    scorer: StudentForcedScorer,
    host: str = "127.0.0.1",
    port: int = 0,
) -> StudentScorerHTTPServer:
    server = StudentScorerHTTPServer((host, port), _handler_for(scorer))
    server.scorer = scorer
    return server


@contextmanager
def running_student_scorer_server(
    scorer: StudentForcedScorer,
    host: str = "127.0.0.1",
    port: int = 0,
) -> Iterator[StudentScorerHTTPServer]:
    server = create_student_scorer_server(scorer, host=host, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _build_scorer(args: argparse.Namespace) -> StudentScorer:
    kwargs: dict[str, Any] = {
        "model_path": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "top_k": args.top_k,
    }
    return StudentScorer(**kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--model", required=True, help="Local path to student model")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    scorer = _build_scorer(args)
    server = create_student_scorer_server(scorer, host=args.host, port=args.port)
    print(json.dumps({"address": server.server_address, "status": "ok"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
