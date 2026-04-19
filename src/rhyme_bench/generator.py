"""
Corpus generator: produces synthetic incidents from archetype templates.

When LLM-generated prose pools are available (data/prose_pools/*.json),
the generator samples from those pools instead of the hardcoded archetype
templates. This breaks the lexical regularity that lets keyword-based
retrievers cheat (spec §10 FM1).

Falls back to archetype templates if no prose pool exists for a class.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .archetypes import (
    ARCHETYPES,
    Archetype,
    AlertTemplate,
    LogTemplate,
    SERVICE_NAMES,
    DB_NAMES,
    CACHE_NAMES,
)
from .models import (
    Alert,
    Corpus,
    Fingerprint,
    Incident,
    IncidentLabels,
    IncidentPayload,
    LogLine,
    TopologyEdge,
    TopologyFragment,
    TopologyNode,
)
from .prose_pools import ProsePool, load_all_prose_pools
from .taxonomy import (
    CAUSE_FAMILY_MAP,
    CONFUSABLE_PAIRS,
    REMEDIATIONS,
    TAXONOMY_VERSION,
    CauseClass,
    ConfusabilityTier,
)


ERROR_TYPES = [
    "NullPointerException", "IndexOutOfBoundsException", "ConnectionRefusedException",
    "TimeoutException", "IllegalStateException", "RuntimeException",
]

ENDPOINTS = [
    "/api/v1/orders", "/api/v1/users", "/api/v1/payments", "/api/v2/search",
    "/api/v1/cart", "/api/v1/inventory", "/api/v1/recommendations",
    "/internal/health", "/api/v1/checkout", "/api/v1/catalog",
]

OBJECT_TYPES = [
    "HttpConnection", "RequestContext", "SessionState", "CacheEntry",
    "EventBuffer", "ResponseWrapper", "PooledConnection",
]


def _pick_services(
    archetype: Archetype,
    used_services: set[str],
    rng: random.Random,
) -> tuple[str, str | None, str | None, list[str]]:
    """Pick origin, downstream, db, and affected services for an incident."""
    available = [s for s in SERVICE_NAMES if s not in used_services]
    if len(available) < 4:
        available = list(SERVICE_NAMES)

    rng.shuffle(available)
    origin = available[0]
    downstream = available[1] if archetype.requires_downstream else None
    db = rng.choice(DB_NAMES) if archetype.requires_db else None

    num_affected = rng.randint(*archetype.num_affected_range)
    affected = [origin] + available[1 : 1 + num_affected]
    if downstream and downstream not in affected:
        affected.append(downstream)

    return origin, downstream, db, affected


def _make_timestamps(rng: random.Random) -> tuple[datetime, datetime]:
    """Generate a plausible incident time window."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=rng.randint(0, 365),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
    )
    duration = timedelta(minutes=rng.randint(15, 240))
    return base, base + duration


def _interpolate(template: str, ctx: dict[str, str]) -> str:
    """Best-effort interpolation of template variables."""
    result = template
    for key, val in ctx.items():
        result = result.replace("{" + key + "}", str(val))
    return result


def _randomize_summary_style(summary: str, rng: random.Random) -> str:
    """Apply random style transformations to a summary to break structural patterns.

    Variations:
    - Split long sentences into shorter ones
    - Merge short sentences into compound ones
    - Add/remove filler phrases
    - Vary punctuation style
    """
    sentences = [s.strip() for s in summary.split(". ") if s.strip()]
    if not sentences:
        return summary

    variant = rng.randint(0, 4)

    if variant == 0:
        # Terse: keep only first 2-3 sentences, trim each
        max_keep = max(2, min(3, len(sentences)))
        kept = sentences[:rng.randint(1, max_keep)]
        return ". ".join(kept) + "."

    elif variant == 1:
        # Split long sentences at commas/semicolons
        new_sentences = []
        for s in sentences:
            if len(s.split()) > 15 and ("," in s or "—" in s or ";" in s):
                for splitter in [" — ", "; ", ", "]:
                    if splitter in s:
                        parts = s.split(splitter, 1)
                        new_sentences.extend(p.strip().capitalize() for p in parts if p.strip())
                        break
                else:
                    new_sentences.append(s)
            else:
                new_sentences.append(s)
        return ". ".join(new_sentences) + "."

    elif variant == 2:
        # Merge pairs of short sentences with conjunctions
        conjunctions = [" and ", " while ", " — ", "; "]
        merged = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and len(sentences[i].split()) < 12 and rng.random() < 0.5:
                conj = rng.choice(conjunctions)
                merged.append(sentences[i] + conj + sentences[i + 1][0].lower() + sentences[i + 1][1:])
                i += 2
            else:
                merged.append(sentences[i])
                i += 1
        return ". ".join(merged) + "."

    elif variant == 3:
        # Add contextual filler at start or end
        prefixes = [
            "Around {deploy_time},",
            "During routine monitoring,",
            "The on-call engineer noticed that",
            "At approximately {deploy_time},",
            "Following a page,",
        ]
        suffixes = [
            " Further investigation is ongoing.",
            " The team is monitoring for recurrence.",
            " Impact was contained to {service}.",
            " No data loss was reported.",
        ]
        result = rng.choice(prefixes) + " " + sentences[0][0].lower() + sentences[0][1:]
        if len(sentences) > 1:
            result += ". " + ". ".join(sentences[1:])
        result += rng.choice(suffixes) if rng.random() < 0.5 else "."
        return result

    else:
        # Pass through unchanged
        return summary


def _generate_alerts(
    archetype: Archetype,
    ctx: dict[str, str],
    start: datetime,
    end: datetime,
    rng: random.Random,
    prose_pool: ProsePool | None = None,
    all_pools: dict | None = None,
) -> list[Alert]:
    """Generate 5-15 alerts from archetype templates or prose pool.

    Injects noise alerts from other cause classes (20-40% of total) to
    break structural patterns that correlate with cause class.
    """
    num_alerts = rng.randint(5, 15)
    # Determine noise ratio: 20-40% of alerts come from other classes
    noise_count = int(num_alerts * rng.uniform(0.2, 0.4)) if all_pools else 0
    signal_count = num_alerts - noise_count

    alerts = []
    # Signal alerts (from this incident's cause class)
    for _ in range(signal_count):
        if prose_pool and prose_pool.alerts:
            pool_entry = rng.choice(prose_pool.alerts)
            severity = pool_entry.severity
            msg_template = rng.choice(pool_entry.messages)
        else:
            template = rng.choice(archetype.alerts)
            severity = template.severity
            msg_template = rng.choice(template.message_templates)
        ts = start + timedelta(
            seconds=rng.randint(0, max(1, int((end - start).total_seconds())))
        )
        alerts.append(Alert(
            timestamp=ts.isoformat(),
            severity=severity,
            service=ctx.get("service", "unknown"),
            message=_interpolate(msg_template, ctx),
        ))

    # Noise alerts (from random other cause classes)
    if all_pools and noise_count > 0:
        other_pools = [p for cc, p in all_pools.items() if p != prose_pool]
        for _ in range(noise_count):
            noise_pool = rng.choice(other_pools) if other_pools else prose_pool
            if noise_pool and noise_pool.alerts:
                pool_entry = rng.choice(noise_pool.alerts)
                # Noise alerts are typically lower severity
                severity = rng.choice(["info", "warning"])
                msg_template = rng.choice(pool_entry.messages)
            else:
                # Fallback: generic infrastructure noise
                severity = "info"
                msg_template = rng.choice([
                    "Health check passed for {service}",
                    "Metrics collection completed in {duration}ms",
                    "Certificate renewal check: all certs valid",
                    "Scheduled backup completed successfully",
                    "Auto-scaling evaluation: no action needed",
                    "DNS cache refreshed: {cache_miss_pct}% miss rate",
                    "Node {node} resource utilization normal",
                ])
            ts = start + timedelta(
                seconds=rng.randint(0, max(1, int((end - start).total_seconds())))
            )
            noise_service = rng.choice(SERVICE_NAMES)
            alerts.append(Alert(
                timestamp=ts.isoformat(),
                severity=severity,
                service=noise_service,
                message=_interpolate(msg_template, ctx),
            ))

    alerts.sort(key=lambda a: a.timestamp)
    return alerts


def _generate_logs(
    archetype: Archetype,
    ctx: dict[str, str],
    start: datetime,
    end: datetime,
    rng: random.Random,
    prose_pool: ProsePool | None = None,
    all_pools: dict | None = None,
) -> list[LogLine]:
    """Generate 20-100 log lines from archetype templates or prose pool.

    Injects noise log lines from other cause classes and generic infrastructure
    logs (20-40% of total) to break structural patterns.
    """
    num_logs = rng.randint(20, 60)
    noise_count = int(num_logs * rng.uniform(0.2, 0.4)) if all_pools else 0
    signal_count = num_logs - noise_count

    logs = []
    # Signal logs
    for _ in range(signal_count):
        if prose_pool and prose_pool.logs:
            pool_entry = rng.choice(prose_pool.logs)
            level = pool_entry.level
            msg_template = rng.choice(pool_entry.messages)
        else:
            template = rng.choice(archetype.logs)
            level = template.level
            msg_template = rng.choice(template.message_templates)
        ts = start + timedelta(
            seconds=rng.randint(0, max(1, int((end - start).total_seconds())))
        )
        service = ctx.get("service", "unknown")
        if level in ("ERROR", "WARN") and "downstream" in ctx and rng.random() < 0.3:
            service = ctx["downstream"]
        logs.append(LogLine(
            timestamp=ts.isoformat(),
            service=service,
            level=level,
            message=_interpolate(msg_template, ctx),
        ))

    # Noise logs (from other classes + generic infrastructure)
    if all_pools and noise_count > 0:
        other_pools = [p for cc, p in all_pools.items() if p != prose_pool]
        generic_logs = [
            ("INFO", "GET /healthz 200 {duration}ms"),
            ("INFO", "POST {endpoint} 200 {duration}ms"),
            ("INFO", "Connection established to {downstream}:{port}"),
            ("DEBUG", "Request {request_id} completed in {duration}ms"),
            ("INFO", "Worker thread started: pool={pool_name}, id={conn_id}"),
            ("DEBUG", "Cache hit for key={record_id}, ttl={tolerance}s remaining"),
            ("INFO", "Metrics flushed: {rps} datapoints in {window}s"),
            ("INFO", "Config loaded from /etc/config/{configmap_name}"),
            ("DEBUG", "GC completed in {gc_baseline}ms, heap={heap_mb}MB"),
            ("INFO", "Scheduled task completed: cleanup in {duration}ms"),
            ("WARN", "Slow request: {method} {endpoint} took {duration}ms"),
            ("INFO", "TLS handshake completed with {downstream} in {conn_time}ms"),
        ]
        for _ in range(noise_count):
            if rng.random() < 0.5 and other_pools:
                # Log from another cause class's pool
                noise_pool = rng.choice(other_pools)
                if noise_pool and noise_pool.logs:
                    pool_entry = rng.choice(noise_pool.logs)
                    level = pool_entry.level
                    # Downgrade most to INFO/DEBUG
                    if rng.random() < 0.7:
                        level = rng.choice(["INFO", "DEBUG"])
                    msg_template = rng.choice(pool_entry.messages)
                else:
                    level, msg_template = rng.choice(generic_logs)
            else:
                # Generic infrastructure log
                level, msg_template = rng.choice(generic_logs)

            ts = start + timedelta(
                seconds=rng.randint(0, max(1, int((end - start).total_seconds())))
            )
            noise_service = rng.choice(SERVICE_NAMES)
            logs.append(LogLine(
                timestamp=ts.isoformat(),
                service=noise_service,
                level=level,
                message=_interpolate(msg_template, ctx),
            ))

    logs.sort(key=lambda l: l.timestamp)
    return logs


def _generate_topology(
    origin: str,
    downstream: str | None,
    db: str | None,
    affected: list[str],
) -> TopologyFragment:
    nodes = [TopologyNode(service=s, kind="http") for s in affected]
    edges = []
    if downstream:
        edges.append(TopologyEdge(source=origin, target=downstream))
        if not any(n.service == downstream for n in nodes):
            nodes.append(TopologyNode(service=downstream, kind="http"))
    if db:
        nodes.append(TopologyNode(service=db, kind="database"))
        target_service = downstream or origin
        edges.append(TopologyEdge(source=target_service, target=db, protocol="tcp"))
    # Add some edges between affected services for realism
    for s in affected:
        if s != origin and s != downstream:
            edges.append(TopologyEdge(source=origin, target=s))
    return TopologyFragment(nodes=nodes, edges=edges)


def _build_context(
    archetype: Archetype,
    origin: str,
    downstream: str | None,
    db: str | None,
    rng: random.Random,
) -> dict[str, str]:
    """Build template interpolation context with realistic values."""
    ctx: dict[str, str] = {
        "service": origin,
        "pod": f"{origin}-{rng.randint(1,9)}",
        "endpoint": rng.choice(ENDPOINTS),
        "method": rng.choice(["GET", "POST", "PUT"]),
        "request_id": uuid.uuid4().hex[:12],
        "trace_id": uuid.uuid4().hex[:16],
        "error_pct": str(rng.randint(5, 65)),
        "latency": str(rng.randint(500, 15000)),
        "p50": str(rng.randint(200, 2000)),
        "p99": str(rng.randint(2000, 30000)),
        "slo": str(rng.choice([200, 500, 1000, 2000])),
        "baseline": str(rng.randint(20, 200)),
        "timeout": str(rng.choice([3000, 5000, 10000, 30000])),
        "timeout_pct": str(rng.randint(5, 40)),
        "error": rng.choice(["connection refused", "timeout", "503 Service Unavailable", "reset by peer"]),
        "error_type": rng.choice(ERROR_TYPES),
        "error_msg": "unexpected null value in response field",
        "port": str(rng.choice([8080, 8443, 9090, 3000, 5432])),
        "conn_count": str(rng.randint(90, 200)),
        "conn_max": str(rng.choice([100, 200])),
        "queue_depth": str(rng.randint(100, 5000)),
        "rate": str(rng.randint(500, 10000)),
        "normal_rate": str(rng.randint(50, 500)),
        "retry_pct": str(rng.randint(30, 80)),
        "rps": str(rng.randint(500, 10000)),
        "baseline_rps": str(rng.randint(100, 500)),
        "capacity": str(rng.randint(300, 2000)),
        "cpu_pct": str(rng.randint(70, 99)),
        "mem_pct": str(rng.randint(80, 99)),
        "mem_used": f"{rng.randint(900, 2000)}Mi",
        "mem_limit": f"{rng.choice([1024, 2048, 4096])}Mi",
        "heap_mb": str(rng.randint(500, 3000)),
        "start_heap": str(rng.randint(100, 300)),
        "max_mb": str(rng.choice([1024, 2048, 4096])),
        "rss_mb": str(rng.randint(800, 3000)),
        "cache_mb": str(rng.randint(50, 200)),
        "limit_mb": str(rng.choice([1024, 2048, 4096])),
        "freed_mb": str(rng.randint(10, 200)),
        "nonheap_mb": str(rng.randint(50, 200)),
        "alloc_mb": str(rng.randint(50, 500)),
        "gc_pause": str(rng.randint(100, 2000)),
        "gc_baseline": str(rng.randint(5, 30)),
        "gc_pct": str(rng.randint(30, 80)),
        "per_req_mb": f"{rng.uniform(0.5, 5.0):.1f}",
        "hours": str(rng.randint(2, 48)),
        "restart_count": str(rng.randint(3, 20)),
        "window": str(rng.choice([1, 5, 10, 30, 60])),
        "from_replicas": str(rng.randint(2, 5)),
        "to_replicas": str(rng.randint(6, 20)),
        "current": str(rng.randint(3, 8)),
        "target": str(rng.randint(8, 20)),
        "deploy_id": f"deploy-{uuid.uuid4().hex[:8]}",
        "prev_deploy_id": f"deploy-{uuid.uuid4().hex[:8]}",
        "deploy_time": "2024-06-15T14:30:00Z",
        "image": f"registry.internal/{origin}",
        "tag": f"v{rng.randint(1,9)}.{rng.randint(0,99)}.{rng.randint(0,999)}",
        "version": f"v{rng.randint(2,5)}.{rng.randint(0,20)}",
        "pod_count": str(rng.randint(3, 10)),
        "ready_count": str(rng.randint(3, 10)),
        "object_type": rng.choice(OBJECT_TYPES),
        "pool_name": rng.choice(["http-conn-pool", "db-conn-pool", "thread-pool"]),
        "active": str(rng.randint(50, 200)),
        "idle": str(rng.randint(0, 10)),
        "waiting": str(rng.randint(0, 100)),
        "created": str(rng.randint(200, 5000)),
        "class_name": rng.choice(["OrderController", "PaymentHandler", "UserService", "CartManager"]),
        "method_name": rng.choice(["processRequest", "handlePayment", "getUser", "updateCart"]),
        "file": rng.choice(["Handler.java", "Controller.go", "service.py", "handler.ts"]),
        "line": str(rng.randint(50, 500)),
        "property": rng.choice(["id", "name", "status", "amount", "userId"]),
        "index": str(rng.randint(0, 10)),
        "length": str(rng.randint(0, 5)),
        "config_key": rng.choice(["max_connections", "timeout_ms", "retry_count", "batch_size"]),
        "expected": str(rng.randint(1, 100)),
        "actual": str(rng.randint(100, 1000)),
        "canary_err": f"{rng.uniform(1, 10):.1f}",
        "baseline_err": f"{rng.uniform(0, 0.5):.2f}",
        "affected_pct": str(rng.randint(10, 80)),
        "attempt": str(rng.randint(1, 5)),
        "threshold": str(rng.choice([5, 10, 20])),
        "failures": str(rng.randint(5, 50)),
        "queue_pct": str(rng.randint(70, 100)),
        "backoff": str(rng.choice([100, 500, 1000, 2000, 5000])),
        "delay": str(rng.choice([100, 200, 500, 1000])),
        "backlog": str(rng.randint(50, 500)),
        "wait_ms": str(rng.randint(100, 5000)),
        "util_pct": str(rng.randint(80, 100)),
        "traffic_mult": f"{rng.uniform(2, 10):.1f}",
        "amp_ratio": f"{rng.uniform(3, 15):.1f}",
        "query_count": str(rng.randint(50, 500)),
        "normal_queries": str(rng.randint(5, 30)),
        "hc_latency": str(rng.randint(500, 5000)),
        "hc_threshold": str(rng.choice([100, 200, 500])),
        "conn_time": str(rng.randint(500, 5000)),
        "query_time": str(rng.randint(1000, 30000)),
        "table": rng.choice(["orders", "users", "payments", "inventory", "sessions"]),
        "duration": str(rng.randint(1000, 30000)),
        "total": str(rng.randint(2000, 60000)),
        "avg": str(rng.randint(500, 5000)),
        "latency_mult": f"{rng.uniform(2, 20):.0f}",
        "mem_mb": str(rng.randint(500, 3000)),

        # --- DNS / certificate ---
        "dns_server": rng.choice(["10.96.0.10:53", "169.254.169.254:53", "kube-dns.kube-system"]),
        "ttl_ago": str(rng.randint(30, 300)),
        "dns_latency": str(rng.randint(50, 5000)),
        "namespace": rng.choice(["default", "production", "platform", "services"]),
        "cache_miss_pct": str(rng.randint(50, 95)),
        "affected_count": str(rng.randint(2, 8)),
        "cert_expiry": f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T00:00:00Z",
        "cert_name": rng.choice(["tls-api-cert", "tls-internal", "mtls-service-cert", "wildcard-cert"]),
        "cert_subject": rng.choice(["*.internal.svc", "api.example.com", "payments.internal"]),
        "cert_issuer": rng.choice(["letsencrypt", "internal-ca", "vault-pki"]),
        "last_renewal": f"2024-{rng.randint(1,6):02d}-{rng.randint(1,28):02d}T12:00:00Z",

        # --- Connection pool ---
        "pool_wait": str(rng.randint(5000, 30000)),
        "pool_wait_threshold": str(rng.choice([1000, 3000, 5000])),
        "min_idle": str(rng.choice([2, 5, 10])),
        "pool_util_pct": str(rng.randint(90, 100)),
        "db_conn_count": str(rng.randint(50, 200)),
        "db_conn_max": str(rng.choice([50, 100, 200])),
        "stale_seconds": str(rng.randint(300, 3600)),
        "conn_id": str(rng.randint(1, 999)),
        "conn_duration": str(rng.randint(100, 30000)),

        # --- CPU throttling ---
        "throttle_count": str(rng.randint(100, 10000)),
        "throttle_pct": str(rng.randint(20, 80)),
        "cpu_limit": f"{rng.choice([0.5, 1.0, 2.0, 4.0])}",
        "cpu_request": f"{rng.choice([0.25, 0.5, 1.0, 2.0])}",
        "cpu_actual": f"{rng.uniform(0.8, 4.0):.1f}",
        "cpu_user": str(rng.randint(500, 5000)),
        "cpu_system": str(rng.randint(50, 500)),
        "throttle_ms": str(rng.randint(100, 5000)),
        "quota": str(rng.choice([50000, 100000, 200000])),
        "period": "100000",
        "expected_ms": str(rng.choice([10, 50, 100, 200])),
        "thread_max": str(rng.choice([50, 100, 200, 400])),
        "blocked_ms": str(rng.randint(100, 5000)),
        "batch_size": str(rng.randint(10, 500)),

        # --- Disk pressure ---
        "disk_pct": str(rng.randint(95, 100)),
        "disk_used": str(rng.randint(40, 200)),
        "disk_total": str(rng.choice([50, 100, 200, 500])),
        "volume": rng.choice(["/data", "/var/log", "/tmp", "/var/lib/postgresql"]),
        "node": f"node-{rng.randint(1,20)}",
        "growth_rate": str(rng.randint(10, 500)),
        "pvc_name": f"data-{origin}-0",
        "wal_segment": f"00000001000000{rng.randint(0,99):02d}",
        "log_count": str(rng.randint(100, 10000)),
        "log_size": str(rng.randint(10, 100)),
        "tmp_dir": "/tmp",
        "tmp_size": str(rng.randint(5, 50)),
        "retention_days": str(rng.choice([7, 14, 30])),

        # --- Config regression ---
        "config_time": "2024-06-15T14:30:00Z",
        "feature_flag": rng.choice(["enable_new_checkout", "use_v2_api", "dark_launch_payments", "enable_caching"]),
        "configmap_name": f"{origin}-config",
        "old_value": rng.choice(["true", "100", "v1", "enabled", "3000"]),
        "new_value": rng.choice(["false", "0", "v2", "disabled", "null", "-1"]),
        "config_type": rng.choice(["integer", "boolean", "string", "duration"]),
        "changed_keys": str(rng.randint(1, 5)),
        "last_deploy_ago": rng.choice(["3 days", "1 week", "2 weeks", "5 days"]),
        "segment": rng.choice(["all_users", "beta", "internal", "us-east-1"]),

        # --- Network ---
        "az": rng.choice(["us-east-1a", "us-east-1b", "us-west-2a", "eu-west-1c"]),
        "downstream_az": rng.choice(["us-east-1c", "us-west-2b", "eu-west-1a"]),
        "fallback_az": rng.choice(["us-east-1b", "us-west-2c", "eu-west-1b"]),
        "packet_loss_pct": str(rng.randint(50, 100)),
        "source_ip": f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
        "dest_ip": f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
        "sg_id": f"sg-{uuid.uuid4().hex[:8]}",
        "hop_num": str(rng.randint(3, 8)),

        # --- Load balancer ---
        "lb_name": rng.choice(["ingress-nginx", "aws-alb", "envoy-proxy", "traefik"]),
        "lb_error_code": rng.choice(["502", "503"]),
        "unhealthy_count": str(rng.randint(1, 10)),
        "total_backends": str(rng.randint(3, 10)),
        "hot_backend": f"{origin}-{rng.randint(1,5)}",
        "hot_pct": str(rng.randint(60, 95)),
        "cold_pct": str(rng.randint(1, 10)),
        "hc_error": rng.choice(["timeout", "connection refused", "404 Not Found"]),
        "hc_path": rng.choice(["/healthz", "/ready", "/health", "/status"]),
        "hc_timeout": str(rng.choice([3000, 5000, 10000])),
        "hc_failures": str(rng.randint(3, 10)),
        "ingress_name": f"{origin}-ingress",
        "healthy_count": str(rng.randint(0, 8)),
        "draining_count": str(rng.randint(0, 2)),
        "avail_pct": str(rng.randint(10, 60)),

        # --- Traffic spike / bot ---
        "campaign_name": rng.choice(["summer-sale", "product-launch", "flash-deal", "holiday-promo"]),
        "organic_pct": str(rng.randint(90, 99)),
        "endpoint_count": str(rng.randint(5, 20)),
        "unique_ips": str(rng.randint(5000, 50000)),
        "unique_uas": str(rng.randint(50, 200)),
        "rate_limit": str(rng.randint(1000, 10000)),
        "ramp_minutes": str(rng.randint(5, 60)),
        "bot_pct": str(rng.randint(60, 95)),
        "bot_ips": str(rng.randint(3, 50)),
        "bot_rps": str(rng.randint(500, 5000)),
        "bot_ua": rng.choice(["python-requests/2.28", "curl/7.88", "Go-http-client/1.1", ""]),
        "bot_ua_pct": str(rng.randint(70, 99)),
        "bot_ua_count": str(rng.randint(1, 5)),
        "no_session_pct": str(rng.randint(80, 100)),
        "endpoint_pct": str(rng.randint(60, 95)),
        "endpoint_rps": str(rng.randint(500, 5000)),
        "normal_endpoint_rps": str(rng.randint(10, 100)),
        "top_ips": str(rng.randint(3, 10)),
        "top_ip_pct": str(rng.randint(60, 95)),
        "blocked_ips": str(rng.randint(3, 50)),
        "other_pct": str(rng.randint(5, 40)),
        "rate_limited": str(rng.randint(500, 5000)),

        # --- Cascading timeout ---
        "deep_downstream": rng.choice(SERVICE_NAMES),
        "downstream_timeout": str(rng.choice([3000, 5000, 10000])),
        "total_timeout": str(rng.choice([15000, 30000, 60000])),

        # --- Schema migration / data corruption ---
        "migration_id": f"migration-{rng.randint(100,999)}",
        "migration_time": "2024-06-15T14:30:00Z",
        "migration_action": rng.choice(["DROP COLUMN", "RENAME COLUMN", "ALTER TYPE", "ADD NOT NULL"]),
        "column": rng.choice(["status", "email", "amount", "created_at", "metadata"]),
        "entity": rng.choice(["Order", "User", "Payment", "Inventory"]),
        "expected_schema": f"v{rng.randint(10,50)}",
        "actual_schema": f"v{rng.randint(51,60)}",
        "schema_version": f"v{rng.randint(51,60)}",
        "failed_queries": str(rng.randint(100, 5000)),
        "total_queries": str(rng.randint(5000, 20000)),
        "query_fragment": rng.choice(["SELECT * FROM orders WHERE", "INSERT INTO users", "UPDATE payments SET"]),
        "sql_error": rng.choice(["column does not exist", "relation does not exist", "type mismatch"]),
        "record_id": str(rng.randint(10000, 99999)),
        "corrupted_records": str(rng.randint(10, 500)),
        "corrupted_ids": f"{rng.randint(10000,50000)}, {rng.randint(50001,99999)}",
        "deserialize_error": rng.choice(["invalid JSON", "unexpected EOF", "invalid UTF-8", "checksum mismatch"]),
        "expected_checksum": uuid.uuid4().hex[:8],
        "actual_checksum": uuid.uuid4().hex[:8],
        "ref_id": str(rng.randint(10000, 99999)),
        "position": str(rng.randint(0, 1000)),
        "bad_value": rng.choice(["null", "NaN", "\x00\x00", "undefined"]),
        "constraint": rng.choice(["NOT NULL", "UNIQUE", "CHECK(amount > 0)", "FOREIGN KEY"]),
        "byte_offset": str(rng.randint(0, 4096)),
        "actual_len": str(rng.randint(10, 100)),
        "expected_len": str(rng.randint(200, 500)),
        "scanned_records": str(rng.randint(10000, 100000)),
        "clean_records": str(rng.randint(9500, 99500)),
        "corrupt_start": f"2024-{rng.randint(1,6):02d}-{rng.randint(1,28):02d}",
        "corrupt_end": f"2024-{rng.randint(6,12):02d}-{rng.randint(1,28):02d}",
        "db_latency": str(rng.randint(1, 20)),

        # --- Clock skew / race condition ---
        "clock_offset": f"{rng.uniform(1, 30):.1f}",
        "clock_direction": rng.choice(["ahead of", "behind"]),
        "token_issued": "2024-06-15T14:30:00Z",
        "token_exp": "2024-06-15T15:30:00Z",
        "current_time": "2024-06-15T14:29:00Z",
        "tolerance": str(rng.choice([5, 10, 30, 60])),
        "lease_time": "2024-06-15T14:29:50Z",
        "last_sync_ago": rng.choice(["2 hours", "6 hours", "1 day", "3 days"]),
        "event_time": "2024-06-15T14:30:00Z",
        "later_time": "2024-06-15T14:29:58Z",
        "ntp_server": rng.choice(["0.pool.ntp.org", "time.google.com", "169.254.169.123"]),
        "jitter": str(rng.randint(1, 500)),
        "stratum": str(rng.choice([1, 2, 3, 16])),
        "duplicate_count": str(rng.randint(10, 500)),
        "lost_updates": str(rng.randint(5, 100)),
        "constraint_violations": str(rng.randint(10, 200)),
        "concurrency": str(rng.randint(20, 200)),
        "baseline_concurrency": str(rng.randint(2, 10)),
        "deadlock_count": str(rng.randint(1, 50)),
        "txn_id": str(rng.randint(1000, 9999)),
        "blocking_txn_id": str(rng.randint(1000, 9999)),
        "expected_version": str(rng.randint(1, 50)),
        "actual_version": str(rng.randint(51, 100)),
        "concurrent_count": str(rng.randint(2, 10)),
        "value": str(rng.randint(10000, 99999)),
        "isolation_level": rng.choice(["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"]),
        "lock_wait": str(rng.randint(10, 500)),
        "max_lock_wait": str(rng.randint(500, 10000)),
    }

    if downstream:
        ctx["downstream"] = downstream
    if db:
        ctx["db"] = db

    return ctx


def _assign_confusability(cause_class: CauseClass, rng: random.Random) -> ConfusabilityTier:
    """Assign confusability tier based on the cause class's confusable pairs."""
    hard_partners = [
        (a, b) for a, b, tier in CONFUSABLE_PAIRS
        if tier == ConfusabilityTier.HARD and (a == cause_class or b == cause_class)
    ]
    if hard_partners and rng.random() < 0.4:
        return ConfusabilityTier.HARD
    medium_partners = [
        (a, b) for a, b, tier in CONFUSABLE_PAIRS
        if tier == ConfusabilityTier.MEDIUM and (a == cause_class or b == cause_class)
    ]
    if medium_partners and rng.random() < 0.4:
        return ConfusabilityTier.MEDIUM
    return rng.choice([ConfusabilityTier.EASY, ConfusabilityTier.MEDIUM])


def generate_incident(
    cause_class: CauseClass,
    incident_num: int,
    rng: random.Random,
    used_services: set[str] | None = None,
    prose_pool: ProsePool | None = None,
    all_pools: dict[CauseClass, ProsePool] | None = None,
) -> Incident:
    """Generate a single synthetic incident from an archetype."""
    archetype = ARCHETYPES[cause_class]
    used = used_services or set()

    origin, downstream, db, affected = _pick_services(archetype, used, rng)
    start, end = _make_timestamps(rng)
    ctx = _build_context(archetype, origin, downstream, db, rng)

    # Update time-correlated fields for deploy/config classes
    if cause_class in (CauseClass.CODE_REGRESSION, CauseClass.CONFIG_REGRESSION):
        ctx["deploy_time"] = start.isoformat()
        ctx["config_time"] = start.isoformat()
    if cause_class == CauseClass.SCHEMA_MIGRATION_FAILURE:
        ctx["migration_time"] = start.isoformat()

    incident_id = f"INC-{cause_class.value}-{incident_num:03d}"

    # Summary: prefer prose pool, fall back to archetype templates
    if prose_pool and prose_pool.summaries:
        summary_template = rng.choice(prose_pool.summaries)
    else:
        summary_template = rng.choice(archetype.summary.templates)
    summary = _interpolate(summary_template, ctx)
    summary = _randomize_summary_style(summary, rng)

    alerts = _generate_alerts(archetype, ctx, start, end, rng, prose_pool, all_pools)
    logs = _generate_logs(archetype, ctx, start, end, rng, prose_pool, all_pools)
    topology = _generate_topology(origin, downstream, db, affected)

    fp_template = archetype.fingerprint
    amp_ratio = rng.uniform(*fp_template.amplification_ratio_range)

    # DB-side metrics — populated for classes where DB interaction is discriminating
    db_query_time: float | None = None
    db_pool_util: float | None = None
    db_lock: float | None = None

    if cause_class == CauseClass.DOWNSTREAM_SLOWDOWN:
        db_query_time = round(rng.uniform(500, 5000), 1)  # slow queries
        db_pool_util = round(rng.uniform(0.3, 0.6), 2)  # pool is fine
    elif cause_class == CauseClass.CONNECTION_POOL_EXHAUSTION:
        db_query_time = round(rng.uniform(5, 50), 1)  # queries are fast
        db_pool_util = round(rng.uniform(0.95, 1.0), 2)  # pool is maxed
    elif cause_class == CauseClass.SCHEMA_MIGRATION_FAILURE:
        db_query_time = round(rng.uniform(1, 20), 1)
        db_pool_util = round(rng.uniform(0.2, 0.5), 2)
    elif cause_class == CauseClass.DATA_CORRUPTION:
        db_query_time = round(rng.uniform(5, 100), 1)
        db_pool_util = round(rng.uniform(0.2, 0.5), 2)
    elif cause_class == CauseClass.RACE_CONDITION:
        db_query_time = round(rng.uniform(10, 100), 1)
        db_pool_util = round(rng.uniform(0.4, 0.7), 2)
        db_lock = round(rng.uniform(0.3, 0.8), 2)  # elevated lock contention

    # Deploy type — discriminates code vs config regression
    deploy_type: str | None = None
    if cause_class == CauseClass.CODE_REGRESSION:
        deploy_type = "code"
    elif cause_class == CauseClass.CONFIG_REGRESSION:
        deploy_type = "config"

    fingerprint = Fingerprint(
        latency_shift=rng.choice(fp_template.latency_shift),
        error_rate_pattern=rng.choice(fp_template.error_rate_pattern),
        traffic_correlation=rng.choice(fp_template.traffic_correlation),
        origin_service=origin,
        affected_services=affected,
        topology_pattern=rng.choice(fp_template.topology_pattern),
        onset_shape=rng.choice(fp_template.onset_shape),
        duration_pattern=rng.choice(fp_template.duration_pattern),
        deploy_correlation=fp_template.deploy_correlation,
        amplification_ratio=round(amp_ratio, 2),
        db_query_time_ms=db_query_time,
        db_connection_pool_utilization=db_pool_util,
        db_lock_contention=db_lock,
        deploy_type=deploy_type,
    )

    remediation = REMEDIATIONS[cause_class]
    tier = _assign_confusability(cause_class, rng)

    return Incident(
        payload=IncidentPayload(
            incident_id=incident_id,
            summary=summary,
            alerts=alerts,
            log_lines=logs,
            topology=topology,
            incident_start=start.isoformat(),
            incident_end=end.isoformat(),
        ),
        labels=IncidentLabels(
            cause_class=cause_class,
            confusability_tier=tier,
            remediation_canonical=remediation.canonical,
            remediation_masks_symptom=remediation.masks_symptom,
            remediation_would_worsen=remediation.would_worsen,
            fingerprint=fingerprint,
        ),
    )


def generate_corpus(
    incidents_per_class: int = 10,
    seed: int = 42,
    prose_pools_dir: Path | None = None,
) -> Corpus:
    """Generate the benchmark corpus.

    Args:
        incidents_per_class: Number of incidents per cause class.
        seed: Random seed for reproducibility.
        prose_pools_dir: Directory containing LLM-generated prose pool JSON files.
            If provided, the generator samples from these pools for varied prose.
            Falls back to archetype templates for classes without a pool.

    Returns:
        A Corpus with labeled incidents.
    """
    rng = random.Random(seed)

    # Load prose pools if available
    pools: dict[CauseClass, ProsePool] = {}
    if prose_pools_dir and prose_pools_dir.exists():
        pools = load_all_prose_pools(prose_pools_dir)
        if pools:
            print(f"  Loaded prose pools for {len(pools)}/{len(CauseClass)} classes")

    incidents: list[Incident] = []

    for cause_class in CauseClass:
        pool = pools.get(cause_class)
        for i in range(incidents_per_class):
            incident = generate_incident(
                cause_class, i, rng, prose_pool=pool,
                all_pools=pools if pools else None,
            )
            incidents.append(incident)

    # Shuffle so incidents aren't grouped by class
    rng.shuffle(incidents)

    return Corpus(
        version="1.0.0",
        taxonomy_version=TAXONOMY_VERSION,
        incidents=incidents,
    )
