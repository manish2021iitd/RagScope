"""Built-in demo traces: a fictional docs assistant for "Nimbus", an API platform.

Each trace shows a different failure mode ragscope is built to catch.
"""

from __future__ import annotations

from typing import List

from .trace import Chunk, Trace

_RATE = Chunk(
    id="docs/limits.md#rate-limits", source="docs/limits.md", score=0.89,
    text=("Rate limits depend on your plan. The Free tier allows 60 requests per minute "
          "and 10,000 requests per day. The Pro tier allows 600 requests per minute with no "
          "daily cap. Requests over the limit receive HTTP 429 with a Retry-After header."),
)
_PRICING = Chunk(
    id="docs/pricing.md#plans", source="docs/pricing.md", score=0.71,
    text=("Nimbus offers two plans. The Free tier costs nothing and includes community support. "
          "The Pro tier costs $49 per month and includes email support and higher rate limits."),
)
_AUTH = Chunk(
    id="docs/auth.md#tokens", source="docs/auth.md", score=0.42,
    text=("All API calls must include a bearer token in the Authorization header. "
          "Tokens are created in the dashboard under Settings > API keys."),
)
_LOGS = Chunk(
    id="docs/logging.md#retention", source="docs/logging.md", score=0.86,
    text=("Request logs are retained for 30 days on all plans. Logs can be searched in the "
          "dashboard and downloaded as CSV files."),
)
_LOGS_FORMAT = Chunk(
    id="docs/logging.md#format", source="docs/logging.md", score=0.64,
    text=("Each log entry records the timestamp, endpoint, status code and latency of the "
          "request. Request bodies are never logged."),
)
_KEYS_CREATE = Chunk(
    id="docs/auth.md#creating-keys", source="docs/auth.md", score=0.58,
    text=("To create an API key, open the dashboard, go to Settings > API keys and click "
          "New key. Each key is shown only once, so store it somewhere safe."),
)
_KEYS_SCOPES = Chunk(
    id="docs/auth.md#scopes", source="docs/auth.md", score=0.51,
    text=("API keys can be limited to read-only or read-write scopes. Read-only keys cannot "
          "create or delete resources."),
)
_WEBHOOK_INTRO = Chunk(
    id="blog/2024-launch.md", source="blog/2024-launch.md", score=0.81,
    text=("We are excited to launch Nimbus webhooks! Webhooks are the easiest way to react to "
          "events in real time, and our customers love them."),
)
_WEBHOOK_INTRO_DUP = Chunk(
    id="blog/2024-launch-copy.md", source="blog/2024-launch-copy.md", score=0.80,
    text=("We are excited to launch Nimbus webhooks! Webhooks are the easiest way to react to "
          "events in real time, and our customers love them so much."),
)
_REGIONS = Chunk(
    id="docs/regions.md", source="docs/regions.md", score=0.62,
    text=("Nimbus runs in three regions: us-east, eu-west and ap-south. Data never leaves the "
          "region you choose when creating a project."),
)
_STATUS = Chunk(
    id="docs/status.md", source="docs/status.md", score=0.6,
    text="Service status and incident history are published at status.nimbus.example.",
)
_WEBHOOK_RETRY = Chunk(
    id="docs/webhooks.md#retries", source="docs/webhooks.md", score=0.55,
    text=("If your endpoint does not return a 2xx status, Nimbus retries the webhook delivery "
          "up to 5 times with exponential backoff. After the final attempt the event is marked "
          "as failed in the dashboard."),
)


def demo_traces() -> List[Trace]:
    return [
        Trace(
            id="demo-grounded",
            query="What is the rate limit on the free tier?",
            chunks=[_RATE, _PRICING, _AUTH],
            answer=("The Free tier allows 60 requests per minute and 10,000 requests per day. "
                    "If you go over the limit you receive HTTP 429 with a Retry-After header. "
                    "Upgrading to Pro raises the limit to 600 requests per minute. Hope this helps!"),
            metadata={"scenario": "Grounded answer"},
        ),
        Trace(
            id="demo-wrong-number",
            query="How long are request logs kept?",
            chunks=[_LOGS, _LOGS_FORMAT],
            answer=("Request logs are retained for 90 days on all plans. "
                    "You can search them in the dashboard and download them as CSV files. "
                    "Logs can also be streamed automatically to your own S3 bucket."),
            metadata={"scenario": "Hallucinated number + invented feature"},
        ),
        Trace(
            id="demo-retrieval-gap",
            query="How do I rotate an API key without downtime?",
            chunks=[_KEYS_CREATE, _KEYS_SCOPES, _AUTH],
            answer=("To rotate a key, create a new key under Settings > API keys. "
                    "Deploy the new key alongside the old one, then revoke the old key "
                    "once traffic has moved over. Nimbus keeps revoked keys valid for a "
                    "24-hour grace period."),
            metadata={"scenario": "Retriever missed the relevant doc"},
        ),
        Trace(
            id="demo-noisy-context",
            query="How many times are failed webhook deliveries retried?",
            chunks=[_WEBHOOK_INTRO, _WEBHOOK_INTRO_DUP, _REGIONS, _STATUS, _WEBHOOK_RETRY],
            answer=("Nimbus retries a failed webhook delivery up to 5 times with exponential "
                    "backoff. After the final attempt, the event is marked as failed in the "
                    "dashboard."),
            metadata={"scenario": "Noisy, duplicated context; best chunk ranked last"},
        ),
    ]
