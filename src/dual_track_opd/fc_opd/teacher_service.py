"""Standalone synchronous HTTP service for teacher top-k scoring."""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator, Sequence

from .teacher_protocol import TeacherScoreRequest
from .teacher_scorer import SyntheticTeacherScorer, TeacherScorer


class TeacherHTTPServer(HTTPServer):
    scorer: TeacherScorer


def _handler_for(scorer: TeacherScorer) -> type[BaseHTTPRequestHandler]:
    class TeacherRequestHandler(BaseHTTPRequestHandler):
        server: TeacherHTTPServer

        def log_message(self, format: str, *args: object) -> None:
            del format, args

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
            elif self.path == "/metadata":
                self._write_json(HTTPStatus.OK, scorer.metadata.to_dict())
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
                raw_requests = body.get("requests")
                if not isinstance(raw_requests, list) or not raw_requests:
                    raise ValueError("requests must be a non-empty list")
                requests = [TeacherScoreRequest.from_dict(item) for item in raw_requests]
                indexed = sorted(
                    enumerate(requests),
                    key=lambda item: (
                        item[1].condition.value,
                        len(item[1].response_token_ids),
                    ),
                )
                sorted_responses = scorer.score_batch([request for _, request in indexed])
                if len(sorted_responses) != len(indexed):
                    raise RuntimeError("scorer response count does not match request count")
                responses = [None] * len(requests)
                for (original_index, _), response in zip(
                    indexed,
                    sorted_responses,
                    strict=True,
                ):
                    responses[original_index] = response
                self._write_json(
                    HTTPStatus.OK,
                    {"responses": [response.to_dict() for response in responses]},
                )
            except Exception as exc:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": type(exc).__name__, "message": str(exc)},
                )

    return TeacherRequestHandler


def create_teacher_server(
    scorer: TeacherScorer,
    host: str = "127.0.0.1",
    port: int = 0,
) -> TeacherHTTPServer:
    server = TeacherHTTPServer((host, port), _handler_for(scorer))
    server.scorer = scorer
    return server


@contextmanager
def running_teacher_server(
    scorer: TeacherScorer,
    host: str = "127.0.0.1",
    port: int = 0,
) -> Iterator[TeacherHTTPServer]:
    server = create_teacher_server(scorer, host=host, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _build_scorer(args: argparse.Namespace) -> TeacherScorer:
    if args.backend == "synthetic":
        return SyntheticTeacherScorer(vocab_size=args.vocab_size, top_k=args.top_k)
    from .teacher_transformers import TransformersTeacherScorer

    return TransformersTeacherScorer(
        model_id=args.model,
        top_k=args.top_k,
        dtype=args.dtype,
        device=args.device,
        revision=args.revision,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--backend", choices=("synthetic", "transformers"), default="synthetic")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    scorer = _build_scorer(args)
    server = create_teacher_server(scorer, host=args.host, port=args.port)
    print(
        json.dumps({"address": server.server_address, "metadata": scorer.metadata.to_dict()}),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
