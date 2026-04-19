"""
Archetype templates for each cause class.

Each archetype defines the structural skeleton of an incident: what services are
involved, what the fingerprint looks like, what alerts/logs are characteristic.
The generator uses these to produce varied incidents by sampling from the template
distributions and filling prose with variation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .taxonomy import CauseClass, ConfusabilityTier


SERVICE_NAMES = [
    "api-gateway", "order-service", "payment-service", "inventory-service",
    "user-service", "notification-service", "search-service", "cart-service",
    "shipping-service", "recommendation-service", "auth-service", "catalog-service",
    "pricing-service", "analytics-service", "billing-service", "email-service",
]

DB_NAMES = ["orders-db", "users-db", "inventory-db", "payments-db", "catalog-db"]
CACHE_NAMES = ["redis-sessions", "redis-cache", "memcached-catalog"]
QUEUE_NAMES = ["kafka-events", "rabbitmq-tasks", "sqs-notifications"]


@dataclass
class AlertTemplate:
    severity: str
    message_templates: list[str]


@dataclass
class LogTemplate:
    level: str
    message_templates: list[str]


@dataclass
class FingerprintTemplate:
    latency_shift: list[str]
    error_rate_pattern: list[str]
    traffic_correlation: list[str]
    topology_pattern: list[str]
    onset_shape: list[str]
    duration_pattern: list[str]
    deploy_correlation: bool
    amplification_ratio_range: tuple[float, float]


@dataclass
class SummaryTemplate:
    """Templates for the responder summary. {service}, {downstream}, {db} are interpolated."""
    templates: list[str]


@dataclass
class Archetype:
    cause_class: CauseClass
    fingerprint: FingerprintTemplate
    alerts: list[AlertTemplate]
    logs: list[LogTemplate]
    summary: SummaryTemplate
    num_affected_range: tuple[int, int] = (1, 3)
    requires_downstream: bool = False
    requires_db: bool = False


# ---------------------------------------------------------------------------
# DEPENDENCY FAMILY
# ---------------------------------------------------------------------------

RETRY_STORM_ARCHETYPE = Archetype(
    cause_class=CauseClass.RETRY_STORM,
    fingerprint=FingerprintTemplate(
        latency_shift=["bimodal"],
        error_rate_pattern=["step_increase", "bursty"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["fan_out", "cascade"],
        onset_shape=["step", "ramp"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(3.0, 15.0),
    ),
    alerts=[
        AlertTemplate("critical", [
            "High error rate on {service}: {error_pct}% of requests failing (5xx)",
            "{service} error budget burn rate exceeding 10x threshold",
        ]),
        AlertTemplate("critical", [
            "Connection pool exhausted on {downstream}: {conn_count}/{conn_max} connections in use",
            "{downstream} request queue depth at {queue_depth}, exceeding threshold of 100",
        ]),
        AlertTemplate("warning", [
            "{service} outbound request rate to {downstream} is {rate}/s (normal: {normal_rate}/s)",
            "Retry rate on {service}->{downstream} at {retry_pct}% of total requests",
        ]),
        AlertTemplate("warning", [
            "p99 latency on {service} at {latency}ms (SLO: {slo}ms)",
            "{service} circuit breaker OPEN for {downstream} dependency",
        ]),
        AlertTemplate("info", [
            "Auto-scaler triggered for {downstream}: scaling from {from_replicas} to {to_replicas} replicas",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "POST {endpoint} failed: connection refused to {downstream}:{port}",
            "Timeout after {timeout}ms waiting for response from {downstream}",
            "Retry attempt {attempt}/5 for {downstream} request {request_id}: {error}",
        ]),
        LogTemplate("WARN", [
            "Circuit breaker for {downstream} tripped: {failures}/{threshold} failures in {window}s",
            "Request queue for {downstream} at {queue_pct}% capacity",
            "Exponential backoff: next retry for {downstream} in {backoff}ms",
        ]),
        LogTemplate("INFO", [
            "Retrying request to {downstream}: attempt {attempt}, delay {delay}ms",
            "Connection pool stats for {downstream}: active={active}, idle={idle}, waiting={waiting}",
        ]),
        LogTemplate("ERROR", [
            "upstream connect error or disconnect/reset before headers. reset reason: overflow",
            "net/http: request canceled (Client.Timeout exceeded while awaiting headers)",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} experienced cascading failures after {downstream} became partially unreachable. "
        "Outbound request rate from {service} to {downstream} was observed at {amp_ratio}x normal "
        "levels, suggesting aggressive retry behavior. Error rates on {service} climbed to {error_pct}% "
        "and latency became bimodal with a spike at the timeout ceiling.",

        "Elevated 5xx errors on {service} traced to retry amplification against {downstream}. "
        "The downstream service showed connection pool saturation despite modest inbound traffic from "
        "other callers. {service} retry logic was creating a {amp_ratio}x amplification loop.",

        "Pager fired for {service} error rate exceeding SLO. Investigation revealed {service} was "
        "hammering {downstream} with retries after an initial transient error. Downstream connection "
        "pools were exhausted and the retry storm was self-sustaining.",
    ]),
    requires_downstream=True,
)

DOWNSTREAM_SLOWDOWN_ARCHETYPE = Archetype(
    cause_class=CauseClass.DOWNSTREAM_SLOWDOWN,
    fingerprint=FingerprintTemplate(
        latency_shift=["uniform_increase", "p99_spike"],
        error_rate_pattern=["gradual"],
        traffic_correlation=["correlated"],
        topology_pattern=["cascade"],
        onset_shape=["ramp"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.8, 1.2),
    ),
    alerts=[
        AlertTemplate("critical", [
            "p99 latency on {service} at {latency}ms, exceeding SLO of {slo}ms",
            "{service} latency SLO violation: p50={p50}ms p99={p99}ms (target p99<{slo}ms)",
        ]),
        AlertTemplate("critical", [
            "{downstream} response time degraded: p50={p50}ms (baseline {baseline}ms)",
            "{downstream} internal processing time elevated: avg={avg}ms",
        ]),
        AlertTemplate("warning", [
            "{service} timeout rate to {downstream}: {timeout_pct}% of requests",
            "Upstream services of {downstream} reporting elevated latency",
        ]),
        AlertTemplate("warning", [
            "{downstream} CPU utilization at {cpu_pct}% across all pods",
            "{downstream} active {db} query count: {query_count} (normal: {normal_queries})",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "Request to {downstream} timed out after {timeout}ms: {endpoint}",
            "Slow response from {downstream}: {duration}ms for {method} {endpoint}",
        ]),
        LogTemplate("WARN", [
            "{downstream} health check latency elevated: {hc_latency}ms (threshold: {hc_threshold}ms)",
            "Connection to {downstream} slow to establish: {conn_time}ms",
            "{downstream} response time p99 trending upward: {p99}ms over last {window}m",
        ]),
        LogTemplate("INFO", [
            "Request trace {trace_id}: {downstream} took {duration}ms of {total}ms total",
            "{downstream} returning partial results due to internal timeout",
        ]),
        LogTemplate("WARN", [
            "Slow query detected on {db}: {query_time}ms - SELECT * FROM {table} WHERE ...",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{downstream} response times degraded significantly, causing cascading latency increases "
        "across {service} and other upstream callers. {downstream} showed elevated CPU usage and "
        "slow database queries. Latency distribution shifted uniformly rather than showing bimodal "
        "behavior.",

        "Users reported slow page loads. Traced to {downstream} p99 latency increasing from "
        "{baseline}ms to {p99}ms. {service} was timing out on {timeout_pct}% of requests to "
        "{downstream}. Root cause was a query regression in {downstream} after data volume growth.",

        "Gradual latency degradation across multiple services traced to {downstream} slowdown. "
        "The downstream service's internal processing time increased due to resource saturation. "
        "No retry amplification observed — request rates to {downstream} matched inbound traffic.",
    ]),
    requires_downstream=True,
)

DEPENDENCY_OUTAGE_ARCHETYPE = Archetype(
    cause_class=CauseClass.DEPENDENCY_OUTAGE,
    fingerprint=FingerprintTemplate(
        latency_shift=["bimodal"],
        error_rate_pattern=["step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["fan_out"],
        onset_shape=["step"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    num_affected_range=(1, 4),
    requires_downstream=True,
    alerts=[
        AlertTemplate("critical", [
            "{downstream} returning 503 for all requests — dependency status page confirms outage",
            "All calls to {downstream} failing: {error_pct}% error rate since {deploy_time}",
        ]),
        AlertTemplate("critical", [
            "{service} degraded: upstream dependency {downstream} is down (confirmed on their status page)",
            "{downstream} health endpoint returning 503 — external outage confirmed",
        ]),
        AlertTemplate("warning", [
            "{downstream} error rate at 100% — not a network issue, dependency is down",
            "{service} fallback mode activated for {downstream} dependency",
        ]),
        AlertTemplate("warning", [
            "Third-party status: {downstream} reporting major outage since {deploy_time}",
            "{service} circuit breaker OPEN for {downstream} — all requests fast-failing",
        ]),
        AlertTemplate("info", [
            "{downstream} status page updated: investigating elevated error rates",
            "Graceful degradation active: {service} serving cached/default responses for {downstream} features",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "HTTP 503 from {downstream}: Service Unavailable",
            "POST {endpoint} to {downstream} failed: connection refused",
            "{downstream} returned 503: {error} (dependency outage confirmed)",
            "All {conn_count} requests to {downstream} failing with 503",
        ]),
        LogTemplate("ERROR", [
            "Circuit breaker OPEN for {downstream}: 100% failure rate over last {window}s",
            "Fallback triggered for {downstream}: returning cached response for {endpoint}",
        ]),
        LogTemplate("WARN", [
            "{downstream} status page reports outage: 'investigating elevated error rates in {az}'",
            "Third-party dependency {downstream} unavailable — degraded mode enabled for {service}",
            "All endpoints on {downstream} returning 503 — this is not a partial degradation",
        ]),
        LogTemplate("INFO", [
            "Checked {downstream} status: confirmed external outage, not our infrastructure",
            "Network connectivity to other services verified healthy — issue is {downstream}-specific",
            "{downstream} status page: 'identified root cause, working on fix'",
            "Graceful degradation serving {rps} req/s without {downstream} features",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{downstream} went completely down, confirmed by their status page. All requests from "
        "{service} to {downstream} returned 503 immediately — not timeouts, not intermittent, "
        "but hard failures. Our network was healthy; other dependencies were unaffected. "
        "Activated graceful degradation until {downstream} recovered.",

        "External outage on {downstream} impacted {service}. Error rate jumped to {error_pct}% "
        "for all {downstream}-dependent operations. Circuit breaker opened within {window}s. "
        "The failure was confirmed external — {downstream}'s own status page showed a major "
        "incident at the same time.",

        "{service} experienced {error_pct}% failures on features requiring {downstream}. "
        "Initial investigation ruled out our infrastructure — network was healthy, other "
        "dependencies responded normally. {downstream}'s status page confirmed an ongoing "
        "outage. Switched to fallback mode and waited for their recovery.",
    ]),
)

DNS_RESOLUTION_FAILURE_ARCHETYPE = Archetype(
    cause_class=CauseClass.DNS_RESOLUTION_FAILURE,
    fingerprint=FingerprintTemplate(
        latency_shift=["bimodal"],
        error_rate_pattern=["step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["fan_out"],
        onset_shape=["step"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    num_affected_range=(2, 5),
    alerts=[
        AlertTemplate("critical", [
            "{service} unable to resolve hostname {downstream}.svc.cluster.local",
            "DNS resolution failures on {service}: {error_pct}% of lookups failing",
        ]),
        AlertTemplate("critical", [
            "CoreDNS pod {pod} not responding: health check failed",
            "kube-dns service returning SERVFAIL for {downstream}.svc.cluster.local",
        ]),
        AlertTemplate("warning", [
            "Multiple services reporting DNS resolution failures in namespace {namespace}",
            "{service} connection errors: getaddrinfo ENOTFOUND {downstream}",
        ]),
        AlertTemplate("info", [
            "CoreDNS cache miss rate at {cache_miss_pct}% (normal: <5%)",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "dns: lookup {downstream}.svc.cluster.local: no such host",
            "getaddrinfo ENOTFOUND {downstream}.svc.cluster.local",
            "Failed to resolve '{downstream}': NXDOMAIN",
            "dial tcp: lookup {downstream} on {dns_server}: server misbehaving",
        ]),
        LogTemplate("ERROR", [
            "Could not connect to {downstream}: name resolution failed",
            "HTTP request to {downstream} failed: DNS resolution error",
        ]),
        LogTemplate("WARN", [
            "DNS cache expired for {downstream}, re-resolution failing",
            "Falling back to stale DNS entry for {downstream} (TTL expired {ttl_ago}s ago)",
        ]),
        LogTemplate("INFO", [
            "CoreDNS: reloading configuration",
            "DNS query latency to kube-dns: {dns_latency}ms (normal: <5ms)",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "Multiple services lost connectivity to {downstream} simultaneously due to DNS resolution "
        "failures. All hostname-based connections returned NXDOMAIN or SERVFAIL. Services using "
        "cached DNS entries or IP addresses were unaffected. CoreDNS pods showed degraded health.",

        "Outage across {affected_count} services caused by DNS resolution failure for "
        "{downstream}.svc.cluster.local. Errors appeared simultaneously across all callers — not "
        "a gradual degradation. Investigation pointed to CoreDNS pod instability.",

        "Sudden connection failures to {downstream} from {service} and other callers. Error logs "
        "showed 'no such host' and 'NXDOMAIN' responses. Direct IP-based connections succeeded. "
        "Root cause was a DNS infrastructure issue, not a problem with {downstream} itself.",
    ]),
    requires_downstream=True,
)

CERTIFICATE_EXPIRY_ARCHETYPE = Archetype(
    cause_class=CauseClass.CERTIFICATE_EXPIRY,
    fingerprint=FingerprintTemplate(
        latency_shift=["bimodal"],
        error_rate_pattern=["step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["fan_out"],
        onset_shape=["step"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    num_affected_range=(1, 4),
    alerts=[
        AlertTemplate("critical", [
            "TLS handshake failure to {downstream}: certificate has expired",
            "{service} unable to connect to {downstream}: x509 certificate expired at {cert_expiry}",
        ]),
        AlertTemplate("critical", [
            "All HTTPS connections to {downstream} failing: SSL_ERROR_EXPIRED_CERT_ALERT",
            "{service} error rate at {error_pct}%: TLS handshake failures to {downstream}",
        ]),
        AlertTemplate("warning", [
            "Certificate for {downstream} expired at {cert_expiry}",
            "cert-manager: renewal failed for certificate {cert_name} in namespace {namespace}",
        ]),
        AlertTemplate("info", [
            "Certificate details: subject={cert_subject}, expired={cert_expiry}, issuer={cert_issuer}",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "tls: failed to verify certificate: x509: certificate has expired or is not yet valid",
            "SSL routines:ssl3_read_bytes:sslv3 alert certificate expired",
            "HTTPS request to {downstream} failed: certificate expired on {cert_expiry}",
            "mTLS handshake failed: peer certificate expired",
        ]),
        LogTemplate("ERROR", [
            "Connection to {downstream}:443 rejected: TLS handshake error",
            "Failed to establish secure connection: certificate not valid after {cert_expiry}",
        ]),
        LogTemplate("WARN", [
            "cert-manager: certificate {cert_name} renewal deadline exceeded",
            "Attempting fallback to HTTP for {downstream} (TLS unavailable)",
        ]),
        LogTemplate("INFO", [
            "Certificate {cert_name} status: expired since {cert_expiry}",
            "Issuer {cert_issuer} last renewal attempt: {last_renewal} (failed)",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "All HTTPS connections to {downstream} began failing simultaneously with TLS handshake "
        "errors. Certificate inspection showed the TLS cert expired at {cert_expiry}. The "
        "service itself was healthy and reachable over HTTP. Multiple upstream services affected.",

        "{service} and other callers lost connectivity to {downstream} due to an expired TLS "
        "certificate. Error logs uniformly showed 'x509: certificate has expired'. No deploy or "
        "traffic change preceded the incident — onset coincided with the certificate's not-after "
        "date.",

        "Sudden {error_pct}% error rate on {service} traced to TLS certificate expiry on "
        "{downstream}. cert-manager had failed to renew the certificate. The downstream service "
        "was running normally but all encrypted connections were rejected.",
    ]),
    requires_downstream=True,
)

CONNECTION_POOL_EXHAUSTION_ARCHETYPE = Archetype(
    cause_class=CauseClass.CONNECTION_POOL_EXHAUSTION,
    fingerprint=FingerprintTemplate(
        latency_shift=["p99_spike"],
        error_rate_pattern=["gradual", "step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["single_service", "cascade"],
        onset_shape=["ramp"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    requires_db=True,
    requires_downstream=True,
    alerts=[
        AlertTemplate("critical", [
            "{service} connection pool to {db} exhausted: {conn_count}/{conn_max} active",
            "{service} request failures due to connection pool timeout: {error_pct}%",
        ]),
        AlertTemplate("warning", [
            "{service} connection pool wait time: avg {pool_wait}ms (threshold: {pool_wait_threshold}ms)",
            "{service} connection pool idle count: 0 (min healthy: {min_idle})",
        ]),
        AlertTemplate("warning", [
            "{service} p99 latency at {latency}ms — connection acquisition is the bottleneck",
            "{db} active connection count: {db_conn_count} from {service} (limit: {db_conn_max})",
        ]),
        AlertTemplate("info", [
            "{service} connection pool stats: active={active}, idle={idle}, waiting={waiting}, max={conn_max}",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "Failed to acquire connection from pool after {pool_wait}ms: pool exhausted",
            "Connection pool timeout: waited {pool_wait}ms for {db} connection (max pool size: {conn_max})",
            "Cannot get a connection, pool error: Timeout waiting for idle object",
        ]),
        LogTemplate("WARN", [
            "Connection pool approaching limit: {conn_count}/{conn_max} active connections to {db}",
            "Connection leak suspected: {conn_count} active connections, {created} total created, 0 idle",
            "Stale connection detected in pool: last activity {stale_seconds}s ago",
        ]),
        LogTemplate("INFO", [
            "Connection pool status: active={active}, idle={idle}, pending={waiting}, maxSize={conn_max}",
            "New connection created to {db}: total={created}, pool utilization {pool_util_pct}%",
        ]),
        LogTemplate("DEBUG", [
            "Connection #{conn_id} checked out from pool, active={active}",
            "Connection #{conn_id} returned to pool after {conn_duration}ms, idle={idle}",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} experienced escalating latency and timeouts due to connection pool exhaustion. "
        "All {conn_max} connections to {db} were in use with zero idle connections. New requests "
        "queued for a connection and timed out. The database itself was responsive — the bottleneck "
        "was in the application-side pool.",

        "Latency spike on {service} traced to database connection pool saturation. Active connection "
        "count was stuck at {conn_max} with {waiting} requests waiting. Heap dump showed connections "
        "being checked out but never returned, suggesting a connection leak in the request handler.",

        "Gradual degradation on {service}: p99 latency climbed from {baseline}ms to {latency}ms "
        "over {hours} hours. Root cause was connection pool exhaustion to {db}. Connections were "
        "being acquired but not released on error paths, slowly depleting the pool.",
    ]),
)

# ---------------------------------------------------------------------------
# RESOURCE FAMILY
# ---------------------------------------------------------------------------

MEMORY_LEAK_ARCHETYPE = Archetype(
    cause_class=CauseClass.MEMORY_LEAK,
    fingerprint=FingerprintTemplate(
        latency_shift=["p99_spike"],
        error_rate_pattern=["bursty"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["single_service"],
        onset_shape=["ramp"],
        duration_pattern=["sawtooth"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    alerts=[
        AlertTemplate("critical", [
            "{service} pod {pod} OOMKilled: memory usage {mem_used}/{mem_limit}",
            "{service} container restarted due to OOM: restart count {restart_count} in {window}h",
        ]),
        AlertTemplate("warning", [
            "{service} memory usage at {mem_pct}% of limit and rising",
            "{service} heap size growing: {heap_mb}MB (started at {start_heap}MB {hours}h ago)",
        ]),
        AlertTemplate("warning", [
            "{service} GC pause time increasing: {gc_pause}ms (baseline {gc_baseline}ms)",
            "{service} pod {pod} approaching memory limit: {mem_pct}% utilized",
        ]),
        AlertTemplate("info", [
            "{service} pod {pod} restarted by kubelet: reason=OOMKilled",
            "Kubernetes restarting {service} pod {pod}: exit code 137",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "java.lang.OutOfMemoryError: Java heap space",
            "runtime: out of memory: cannot allocate {alloc_mb}MB region",
            "OOMKilled: {service} container exceeded {mem_limit} memory limit",
        ]),
        LogTemplate("WARN", [
            "GC overhead exceeded: spent {gc_pct}% of time in garbage collection",
            "Memory usage {mem_pct}%: heap={heap_mb}MB, non-heap={nonheap_mb}MB",
            "Large allocation detected: {object_type} allocating {alloc_mb}MB",
        ]),
        LogTemplate("INFO", [
            "GC cycle completed: freed {freed_mb}MB, heap now {heap_mb}MB/{max_mb}MB",
            "Container memory: RSS={rss_mb}MB, cache={cache_mb}MB, limit={limit_mb}MB",
        ]),
        LogTemplate("DEBUG", [
            "Object pool stats: {pool_name} active={active} idle={idle} created={created}",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} experienced repeated OOMKill events. Memory usage was observed growing "
        "monotonically over {hours} hours regardless of traffic levels. Each restart temporarily "
        "resolved the issue but memory growth resumed immediately. Heap profiling showed "
        "unbounded growth in {object_type} objects.",

        "Recurring OOM kills on {service} pods. Memory consumption climbed steadily from "
        "{start_heap}MB to {mem_limit}MB over {hours} hours with no correlation to request "
        "volume. The pattern repeated identically after each pod restart, suggesting a "
        "deterministic leak.",

        "Alert fired for {service} OOM kills. Investigation showed sawtooth memory pattern: "
        "linear growth over {hours}h, OOM kill, restart, repeat. Traffic was stable throughout. "
        "No other services affected. Leak traced to unreleased {object_type} references in "
        "request handling path.",
    ]),
)

TRAFFIC_ALLOCATION_ARCHETYPE = Archetype(
    cause_class=CauseClass.TRAFFIC_ALLOCATION,
    fingerprint=FingerprintTemplate(
        latency_shift=["uniform_increase"],
        error_rate_pattern=["step_increase", "gradual"],
        traffic_correlation=["correlated"],
        topology_pattern=["single_service"],
        onset_shape=["ramp", "step"],
        duration_pattern=["self_resolving", "persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    alerts=[
        AlertTemplate("critical", [
            "{service} pod {pod} OOMKilled during traffic peak: memory {mem_used}/{mem_limit}",
            "{service} error rate at {error_pct}% during peak traffic ({rps} req/s)",
        ]),
        AlertTemplate("warning", [
            "{service} memory usage at {mem_pct}% — correlates with traffic spike to {rps} req/s",
            "{service} request rate {rps}/s exceeds provisioned capacity of {capacity}/s",
        ]),
        AlertTemplate("warning", [
            "HPA triggered for {service}: scaling from {from_replicas} to {to_replicas} (CPU: {cpu_pct}%)",
            "{service} request queue depth at {queue_depth}: request rate exceeding processing capacity",
        ]),
        AlertTemplate("info", [
            "{service} auto-scaling in progress: {current}/{target} replicas ready",
            "Traffic spike detected: {service} inbound rate {rps}/s (baseline {baseline_rps}/s)",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "OOMKilled: container memory {mem_used} exceeded limit {mem_limit} during peak load",
            "Request rejected: {service} at capacity ({rps} req/s, limit {capacity} req/s)",
        ]),
        LogTemplate("WARN", [
            "Memory pressure: {mem_pct}% utilized at current traffic level of {rps} req/s",
            "Request processing backlog: {backlog} requests queued, avg wait {wait_ms}ms",
            "Per-request memory allocation elevated: {per_req_mb}MB/req at {rps} req/s",
        ]),
        LogTemplate("INFO", [
            "Traffic level: {rps} req/s (capacity: {capacity} req/s, utilization: {util_pct}%)",
            "Horizontal pod autoscaler adjusting replicas: {from_replicas} -> {to_replicas}",
            "Memory usage tracking request volume: {mem_mb}MB at {rps} req/s",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} hit OOM kills during a traffic spike. Memory usage directly correlated with "
        "inbound request rate, peaking at {rps} req/s against a provisioned capacity of {capacity} "
        "req/s. Adding replicas immediately stabilized the service. Memory returned to baseline "
        "when traffic subsided.",

        "Pager fired for {service} error rate during peak traffic. Request volume reached {rps} "
        "req/s ({traffic_mult}x normal). OOM kills occurred on pods that couldn't handle the "
        "per-request memory footprint at that volume. HPA scaled the service and errors resolved.",

        "{service} experienced capacity-driven OOM events during a {traffic_mult}x traffic surge. "
        "Unlike a memory leak, memory usage dropped back to baseline immediately when traffic "
        "normalized. The service was simply under-provisioned for the peak load.",
    ]),
)

CPU_THROTTLING_ARCHETYPE = Archetype(
    cause_class=CauseClass.CPU_THROTTLING,
    fingerprint=FingerprintTemplate(
        latency_shift=["uniform_increase"],
        error_rate_pattern=["gradual"],
        traffic_correlation=["correlated"],
        topology_pattern=["single_service"],
        onset_shape=["ramp"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    alerts=[
        AlertTemplate("critical", [
            "{service} p99 latency at {latency}ms (SLO: {slo}ms) — CPU throttling detected",
            "{service} SLO violation: {error_pct}% of requests exceeding latency target",
        ]),
        AlertTemplate("warning", [
            "{service} CPU throttle count: {throttle_count} periods/s (container limit: {cpu_limit})",
            "{service} CPU utilization at {cpu_pct}% of cgroup limit across all pods",
        ]),
        AlertTemplate("warning", [
            "{service} container CPU capped at {cpu_limit} cores, throttled {throttle_pct}% of cycles",
            "HPA for {service} cannot scale further: max replicas ({to_replicas}) reached",
        ]),
        AlertTemplate("info", [
            "{service} CPU request/limit: {cpu_request}/{cpu_limit} cores, actual: {cpu_actual} cores",
        ]),
    ],
    logs=[
        LogTemplate("WARN", [
            "Request processing slow: handler took {duration}ms (expected <{expected_ms}ms), possible CPU throttling",
            "Thread pool saturation: {active}/{thread_max} threads active, {waiting} tasks queued",
            "GC pause elevated: {gc_pause}ms — CPU scheduling delays suspected",
        ]),
        LogTemplate("INFO", [
            "CPU stats: user={cpu_user}ms, system={cpu_system}ms, throttled={throttle_ms}ms",
            "Request processing time: p50={p50}ms p99={p99}ms (target p99<{slo}ms)",
            "Container cgroup: cpu.cfs_quota_us={quota}, cpu.cfs_period_us={period}, nr_throttled={throttle_count}",
        ]),
        LogTemplate("WARN", [
            "Slow event loop: blocked for {blocked_ms}ms processing batch of {batch_size} items",
            "Worker thread starvation: task waited {wait_ms}ms in queue before execution",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} experienced progressive latency degradation due to CPU throttling. Container CPU "
        "usage was pegged at the {cpu_limit}-core cgroup limit with {throttle_pct}% of CPU cycles "
        "throttled. The service never crashed or OOM-killed — it simply processed requests slower "
        "as throttling increased.",

        "Latency SLO violation on {service}: p99 climbed to {latency}ms from a {baseline}ms baseline. "
        "No memory pressure or downstream issues detected. Container metrics showed CPU throttle count "
        "rising steadily, confirming the cgroup CPU limit as the bottleneck.",

        "Gradual degradation on {service} correlated with increased request volume. CPU throttle "
        "metrics confirmed the container was hitting its CPU ceiling. Unlike memory-based failures, "
        "the service remained healthy but slow — no restarts, no OOM kills, just increased latency.",
    ]),
)

DISK_PRESSURE_ARCHETYPE = Archetype(
    cause_class=CauseClass.DISK_PRESSURE,
    fingerprint=FingerprintTemplate(
        latency_shift=["p99_spike"],
        error_rate_pattern=["step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["single_service"],
        onset_shape=["ramp"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    alerts=[
        AlertTemplate("critical", [
            "{service} disk usage at {disk_pct}%: volume {volume} nearly full",
            "{service} pod {pod} evicted: DiskPressure condition on node {node}",
        ]),
        AlertTemplate("critical", [
            "{service} write failures: no space left on device ({volume})",
            "Node {node} disk pressure: ephemeral storage at {disk_pct}%",
        ]),
        AlertTemplate("warning", [
            "{service} log volume growing at {growth_rate}MB/h: {disk_used}GB/{disk_total}GB used",
            "PersistentVolumeClaim {pvc_name} utilization at {disk_pct}%",
        ]),
        AlertTemplate("info", [
            "Kubelet eviction threshold reached: nodefs.available < 10%",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "write /data/{file}: no space left on device",
            "ENOSPC: Failed to write to {volume}: disk full",
            "WAL write failed: no space left on device, segment={wal_segment}",
            "Log rotation failed: cannot create new log file, disk full",
        ]),
        LogTemplate("WARN", [
            "Disk usage at {disk_pct}%: {disk_used}GB of {disk_total}GB on {volume}",
            "Old log files accumulating: {log_count} files, {log_size}GB total in /var/log/{service}",
            "Temp directory {tmp_dir} consuming {tmp_size}GB",
        ]),
        LogTemplate("INFO", [
            "Disk usage report: {volume} {disk_used}GB/{disk_total}GB ({disk_pct}%)",
            "Starting emergency log cleanup: removing files older than {retention_days} days",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} failed after disk volume {volume} filled to {disk_pct}% capacity. Write "
        "operations returned 'no space left on device' errors. Read operations continued to "
        "succeed. Investigation showed accumulated log files and WAL segments consuming "
        "{disk_used}GB of the {disk_total}GB volume.",

        "Pod evictions on {service} due to disk pressure. Node {node} ephemeral storage "
        "exceeded the kubelet eviction threshold. The service was writing log files at "
        "{growth_rate}MB/h without rotation, gradually filling the volume over {hours} hours.",

        "{service} began throwing write errors after the persistent volume reached capacity. "
        "No memory or CPU pressure observed. The failure was purely I/O: database WAL segments "
        "and application logs had accumulated beyond the provisioned disk size.",
    ]),
)

# ---------------------------------------------------------------------------
# DEPLOY FAMILY
# ---------------------------------------------------------------------------

CODE_REGRESSION_ARCHETYPE = Archetype(
    cause_class=CauseClass.CODE_REGRESSION,
    fingerprint=FingerprintTemplate(
        latency_shift=["uniform_increase", "p99_spike"],
        error_rate_pattern=["step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["single_service"],
        onset_shape=["step"],
        duration_pattern=["persistent"],
        deploy_correlation=True,
        amplification_ratio_range=(0.9, 1.1),
    ),
    alerts=[
        AlertTemplate("critical", [
            "{service} error rate jumped to {error_pct}% after deploy {deploy_id}",
            "New error type detected on {service}: {error_type} (first seen at {deploy_time})",
        ]),
        AlertTemplate("critical", [
            "{service} p99 latency increased {latency_mult}x after deploy {deploy_id}",
            "{service} 5xx rate: {error_pct}% (baseline <0.1%), started at {deploy_time}",
        ]),
        AlertTemplate("warning", [
            "Deploy {deploy_id} for {service} completed at {deploy_time}",
            "{service} canary showing elevated error rate vs. baseline: {canary_err}% vs {baseline_err}%",
        ]),
        AlertTemplate("info", [
            "Rollback initiated for {service}: reverting {deploy_id} to {prev_deploy_id}",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "NullPointerException in {class_name}.{method_name}({file}:{line})",
            "panic: runtime error: index out of range [{index}] with length {length}",
            "TypeError: Cannot read properties of undefined (reading '{property}')",
            "Unhandled exception in {endpoint}: {error_type}: {error_msg}",
        ]),
        LogTemplate("ERROR", [
            "Failed to process request {request_id}: {error_type} at {class_name}.{method_name}",
            "HTTP 500 on {method} {endpoint}: {error_type}",
        ]),
        LogTemplate("WARN", [
            "Deprecation warning: method {method_name} behavior changed in {version}",
            "Configuration mismatch: expected {expected}, got {actual} for {config_key}",
        ]),
        LogTemplate("INFO", [
            "Deploy {deploy_id} started: image={image}:{tag}",
            "Deploy {deploy_id} completed: {pod_count} pods updated, {ready_count} ready",
            "Rollback {deploy_id} -> {prev_deploy_id}: {pod_count} pods reverted",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} began throwing {error_type} errors immediately after deploy {deploy_id} at "
        "{deploy_time}. Error rate jumped from <0.1% to {error_pct}%. No other services affected. "
        "Rollback to {prev_deploy_id} immediately resolved the issue.",

        "Post-deploy incident on {service}. Deploy {deploy_id} introduced a regression causing "
        "{error_type} on {affected_pct}% of requests to {endpoint}. The error was a new code path "
        "not covered by existing tests. Rollback confirmed the deploy as root cause.",

        "Step-function increase in {service} error rate coinciding exactly with deploy {deploy_id}. "
        "New stack traces appeared ({error_type} in {class_name}.{method_name}) that weren't present "
        "in the previous version. Dependencies all healthy. Rolled back and errors ceased immediately.",
    ]),
)

CONFIG_REGRESSION_ARCHETYPE = Archetype(
    cause_class=CauseClass.CONFIG_REGRESSION,
    fingerprint=FingerprintTemplate(
        latency_shift=["uniform_increase", "p99_spike"],
        error_rate_pattern=["step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["single_service"],
        onset_shape=["step"],
        duration_pattern=["persistent"],
        deploy_correlation=True,
        amplification_ratio_range=(0.9, 1.1),
    ),
    alerts=[
        AlertTemplate("critical", [
            "{service} error rate jumped to {error_pct}% after ConfigMap update at {config_time}",
            "{service} behavior change detected: feature flag '{feature_flag}' toggled at {config_time}",
        ]),
        AlertTemplate("critical", [
            "{service} 5xx rate at {error_pct}% — no code deploy detected, last deploy was {last_deploy_ago}",
            "{service} failing on {affected_pct}% of requests with configuration error",
        ]),
        AlertTemplate("warning", [
            "ConfigMap {configmap_name} updated in namespace {namespace} at {config_time}",
            "{service} reloaded configuration: {config_key} changed from '{old_value}' to '{new_value}'",
        ]),
        AlertTemplate("info", [
            "No deploy events for {service} in last {last_deploy_ago} — checking config changes",
            "Feature flag service: '{feature_flag}' set to {new_value} for {service} at {config_time}",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "Invalid configuration value for {config_key}: '{new_value}' (expected {expected})",
            "Failed to parse config: {config_key} value '{new_value}' is not a valid {config_type}",
            "Service misconfigured: {config_key}='{new_value}' caused {error_type}",
        ]),
        LogTemplate("ERROR", [
            "Request failed due to config: {config_key}={new_value} — {error_msg}",
            "Feature flag '{feature_flag}' enabled code path that throws {error_type}",
        ]),
        LogTemplate("WARN", [
            "Configuration reloaded: {changed_keys} keys changed",
            "Config validation warning: {config_key} value '{new_value}' is outside expected range",
        ]),
        LogTemplate("INFO", [
            "ConfigMap {configmap_name} mounted at /etc/config, watching for changes",
            "Configuration applied: {config_key}='{new_value}' (was '{old_value}')",
            "Feature flag evaluation: '{feature_flag}'={new_value} for user segment '{segment}'",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} error rate spiked to {error_pct}% after a ConfigMap update at {config_time}. "
        "No code deployment occurred — the container image was unchanged. The configuration "
        "change set {config_key} to '{new_value}', which caused request processing failures. "
        "Reverting the ConfigMap resolved the issue immediately.",

        "Incident on {service} caused by a feature flag change. The flag '{feature_flag}' was "
        "toggled at {config_time}, enabling a code path that produced {error_type} errors on "
        "{affected_pct}% of requests. Rolling back the code had no effect; reverting the flag did.",

        "Step-function error rate increase on {service} with no corresponding deploy event. "
        "Investigation found a ConfigMap update at {config_time} that changed {config_key} to "
        "an invalid value. The last code deploy was {last_deploy_ago} ago and had been stable.",
    ]),
)

# ---------------------------------------------------------------------------
# NETWORK FAMILY
# ---------------------------------------------------------------------------

NETWORK_PARTITION_ARCHETYPE = Archetype(
    cause_class=CauseClass.NETWORK_PARTITION,
    fingerprint=FingerprintTemplate(
        latency_shift=["bimodal"],
        error_rate_pattern=["step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["fan_out"],
        onset_shape=["step"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    num_affected_range=(2, 5),
    requires_downstream=True,
    alerts=[
        AlertTemplate("critical", [
            "{service} unable to reach {downstream}: connection timed out (no route to host)",
            "Network connectivity lost between {service} and {downstream} in {az}",
        ]),
        AlertTemplate("critical", [
            "Multiple services in {az} reporting connection failures to {downstream_az}",
            "{service} TCP SYN timeout to {downstream}:{port} — no response",
        ]),
        AlertTemplate("warning", [
            "Network probe: {service} -> {downstream} packet loss at {packet_loss_pct}%",
            "Security group / network policy may be blocking traffic: {service} -> {downstream}:{port}",
        ]),
        AlertTemplate("info", [
            "VPC flow log: REJECT {source_ip} -> {dest_ip}:{port} (security group {sg_id})",
            "Network diagnostic: traceroute from {service} to {downstream} fails at hop {hop_num}",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "dial tcp {dest_ip}:{port}: i/o timeout",
            "connect ETIMEDOUT {dest_ip}:{port}",
            "No route to host: {downstream} ({dest_ip})",
            "TCP connection to {downstream}:{port} failed: SYN sent, no SYN-ACK received",
        ]),
        LogTemplate("ERROR", [
            "Health check failed for {downstream}: connect timeout after {timeout}ms",
            "All endpoints for {downstream} unreachable from {az}",
        ]),
        LogTemplate("WARN", [
            "Intermittent connectivity to {downstream}: {packet_loss_pct}% packet loss",
            "DNS resolved {downstream} to {dest_ip}, but TCP connection fails",
        ]),
        LogTemplate("INFO", [
            "Network topology: {service} in {az}, {downstream} in {downstream_az}",
            "Failover: routing traffic to {downstream} replica in {fallback_az}",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "Services in {az} lost connectivity to {downstream} in {downstream_az}. TCP connections "
        "timed out at the SYN stage — not application-level errors. Services within {az} "
        "communicated normally. Traceroute showed a blackhole at the inter-AZ network hop.",

        "Network partition between {service} and {downstream}. All TCP connections from {az} "
        "to {downstream_az} failed with 'no route to host'. DNS resolution succeeded but "
        "packets were not reaching the destination. Affected {affected_count} services.",

        "Sudden connectivity loss between {service} and {downstream}. Unlike a service outage, "
        "{downstream} was healthy and serving traffic from other availability zones. The failure "
        "was purely network-level: a routing change dropped traffic between {az} and {downstream_az}.",
    ]),
)

LOAD_BALANCER_MISCONFIGURATION_ARCHETYPE = Archetype(
    cause_class=CauseClass.LOAD_BALANCER_MISCONFIGURATION,
    fingerprint=FingerprintTemplate(
        latency_shift=["bimodal"],
        error_rate_pattern=["step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["single_service"],
        onset_shape=["step"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    requires_downstream=True,
    alerts=[
        AlertTemplate("critical", [
            "{service} error rate at {error_pct}% — load balancer returning {lb_error_code}",
            "Load balancer {lb_name}: {unhealthy_count}/{total_backends} backends marked unhealthy",
        ]),
        AlertTemplate("critical", [
            "{service} receiving 502/503 from {lb_name} despite backends reporting healthy",
            "Traffic distribution anomaly: {hot_backend} receiving {hot_pct}% of traffic",
        ]),
        AlertTemplate("warning", [
            "Load balancer {lb_name} health check failing for {service}: {hc_error}",
            "Ingress {ingress_name} routing rule change detected at {config_time}",
        ]),
        AlertTemplate("info", [
            "Load balancer {lb_name} config updated: backend pool changed",
            "Direct request to {service} pod {pod} succeeded — LB routing suspected",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "502 Bad Gateway from {lb_name}: no healthy upstream",
            "503 Service Unavailable: {lb_name} reports all backends down",
            "Load balancer rejected request: routing rule mismatch for path {endpoint}",
        ]),
        LogTemplate("WARN", [
            "Health check probe from {lb_name} timing out: {hc_path} not responding within {hc_timeout}ms",
            "Uneven traffic distribution: pod {pod} at {hot_pct}% load, other pods at {cold_pct}%",
            "Backend {service} removed from {lb_name} pool: health check failed {hc_failures} times",
        ]),
        LogTemplate("INFO", [
            "Ingress controller reconciled: {ingress_name} rules updated",
            "Load balancer {lb_name}: healthy={healthy_count}, unhealthy={unhealthy_count}, draining={draining_count}",
            "Direct pod request succeeded: {method} {endpoint} -> 200 in {duration}ms",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "Clients received {lb_error_code} errors from {lb_name} despite {service} backends being "
        "healthy. Direct requests to service pods succeeded. The load balancer's health check path "
        "had been changed to {hc_path}, which didn't exist on the current service version, causing "
        "all backends to be marked unhealthy.",

        "Traffic to {service} was being routed to a single backend by {lb_name}, causing overload "
        "on one pod while others were idle. Investigation revealed a session affinity misconfiguration "
        "applied during a recent ingress rule change. Other {service} pods were healthy but receiving "
        "no traffic.",

        "{service} availability dropped to {avail_pct}% despite all pods being healthy and "
        "responsive. {lb_name} was returning 502 errors. The load balancer configuration had been "
        "updated and the routing rules no longer matched the service's endpoint paths. Direct "
        "pod-to-pod requests worked correctly.",
    ]),
)

# ---------------------------------------------------------------------------
# TRAFFIC FAMILY
# ---------------------------------------------------------------------------

TRAFFIC_SPIKE_ARCHETYPE = Archetype(
    cause_class=CauseClass.TRAFFIC_SPIKE,
    fingerprint=FingerprintTemplate(
        latency_shift=["uniform_increase"],
        error_rate_pattern=["gradual", "step_increase"],
        traffic_correlation=["correlated"],
        topology_pattern=["single_service", "cascade"],
        onset_shape=["ramp", "step"],
        duration_pattern=["self_resolving"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    alerts=[
        AlertTemplate("critical", [
            "{service} error rate at {error_pct}% during traffic surge to {rps} req/s",
            "{service} overwhelmed: {rps} req/s (provisioned for {capacity} req/s)",
        ]),
        AlertTemplate("warning", [
            "Traffic spike on {service}: {rps} req/s ({traffic_mult}x baseline of {baseline_rps} req/s)",
            "HPA scaling {service}: {from_replicas} -> {to_replicas} replicas (CPU: {cpu_pct}%)",
        ]),
        AlertTemplate("warning", [
            "{service} request queue depth at {queue_depth}, processing backlog",
            "{service} p99 latency at {latency}ms during traffic peak",
        ]),
        AlertTemplate("info", [
            "Marketing campaign '{campaign_name}' launched — expect elevated traffic",
            "Traffic source analysis: {organic_pct}% organic, distributed across {endpoint_count} endpoints",
        ]),
    ],
    logs=[
        LogTemplate("WARN", [
            "Rate limit approaching: {rps}/{rate_limit} req/s on {endpoint}",
            "Request queue backlog: {backlog} pending, avg wait {wait_ms}ms",
            "Thread pool exhausted: {active}/{thread_max} threads, {waiting} tasks queued",
        ]),
        LogTemplate("INFO", [
            "Traffic level: {rps} req/s across {endpoint_count} endpoints (normal distribution)",
            "Request pattern analysis: {unique_ips} unique IPs, {unique_uas} unique user agents",
            "Auto-scaling response: new pod {pod} ready, total replicas: {to_replicas}",
        ]),
        LogTemplate("ERROR", [
            "Request rejected: rate limit exceeded ({rps} req/s > {rate_limit} req/s)",
            "503 Service Unavailable: {service} at capacity during traffic peak",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} was overwhelmed by a {traffic_mult}x traffic spike reaching {rps} req/s "
        "against a provisioned capacity of {capacity} req/s. Traffic distribution was normal "
        "across endpoints with legitimate user agents and session patterns — this was organic "
        "traffic, likely correlated with a marketing event. HPA scaled the service and errors "
        "resolved.",

        "Elevated errors on {service} during a legitimate traffic surge. Request volume increased "
        "from {baseline_rps} to {rps} req/s over {ramp_minutes} minutes. User agent analysis "
        "confirmed normal browser/mobile distribution. Scaling resolved the issue; traffic "
        "subsided naturally.",

        "{service} capacity exceeded by organic traffic growth. Error rate reached {error_pct}% "
        "at peak. Traffic was distributed across {endpoint_count} endpoints with {unique_ips} "
        "unique source IPs. No bot signatures detected. Service recovered after HPA scaling.",
    ]),
)


# ---------------------------------------------------------------------------
# AMPLIFICATION FAMILY
# ---------------------------------------------------------------------------

CASCADING_TIMEOUT_ARCHETYPE = Archetype(
    cause_class=CauseClass.CASCADING_TIMEOUT,
    fingerprint=FingerprintTemplate(
        latency_shift=["uniform_increase"],
        error_rate_pattern=["gradual"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["cascade"],
        onset_shape=["ramp"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.8, 1.2),
    ),
    num_affected_range=(2, 5),
    requires_downstream=True,
    alerts=[
        AlertTemplate("critical", [
            "{service} p99 latency at {timeout}ms (equals configured timeout) — timeout cascade suspected",
            "{service} timeout rate to {downstream}: {timeout_pct}% of requests",
        ]),
        AlertTemplate("critical", [
            "Multiple tiers timing out: {service} -> {downstream} -> {deep_downstream} all at timeout ceiling",
            "{service} error rate at {error_pct}%: upstream callers also timing out",
        ]),
        AlertTemplate("warning", [
            "Latency at every tier matches timeout config: {service}={timeout}ms, {downstream}={downstream_timeout}ms",
            "Thread pool exhaustion on {service}: {active}/{thread_max} threads blocked on downstream calls",
        ]),
        AlertTemplate("info", [
            "Call chain depth: {service} -> {downstream} -> {deep_downstream} (3 hops, cumulative timeout: {total_timeout}ms)",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "Request to {downstream} timed out after {timeout}ms",
            "Upstream caller timed out waiting for {service}: cascade from {downstream} slowdown",
            "Thread blocked for {timeout}ms on {downstream} call, releasing with timeout error",
        ]),
        LogTemplate("WARN", [
            "All requests to {downstream} hitting timeout ceiling ({timeout}ms) — no fast responses",
            "Thread pool {pool_name}: {active}/{thread_max} threads blocked, {waiting} requests queued",
            "Timeout propagation: {deep_downstream} slow -> {downstream} timeout -> {service} timeout",
        ]),
        LogTemplate("INFO", [
            "Request rate to {downstream}: {rate} req/s (matches inbound rate — no amplification)",
            "Timeout configuration: {service}={timeout}ms, {downstream}={downstream_timeout}ms",
            "Connection count stable: {conn_count} to {downstream} (no retry-driven growth)",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "Timeouts cascaded up the call chain from {deep_downstream} through {downstream} to "
        "{service}. At each tier, latency equaled the configured timeout value — requests were "
        "simply waiting until timeout, then failing. No retry amplification was observed: request "
        "rates were 1:1 at each hop. The root cause was {deep_downstream} slowdown.",

        "{service} experienced {error_pct}% timeouts that propagated from a slow dependency chain. "
        "{deep_downstream} became slow, causing {downstream} to hold connections until timeout, "
        "which in turn exhausted {service}'s thread pool. Unlike a retry storm, request rates "
        "stayed stable — the problem was connection holding, not amplification.",

        "Cascading timeout incident across {affected_count} services. Latency at each tier was "
        "clamped at the timeout ceiling: {service} at {timeout}ms, {downstream} at {downstream_timeout}ms. "
        "Thread pools were saturated with blocked calls. Reducing {service}'s timeout value to "
        "fail fast would have limited the blast radius.",
    ]),
)

# ---------------------------------------------------------------------------
# DATA / SCHEMA FAMILY
# ---------------------------------------------------------------------------

SCHEMA_MIGRATION_FAILURE_ARCHETYPE = Archetype(
    cause_class=CauseClass.SCHEMA_MIGRATION_FAILURE,
    fingerprint=FingerprintTemplate(
        latency_shift=["p99_spike"],
        error_rate_pattern=["step_increase"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["single_service"],
        onset_shape=["step"],
        duration_pattern=["persistent"],
        deploy_correlation=True,
        amplification_ratio_range=(0.9, 1.1),
    ),
    requires_db=True,
    alerts=[
        AlertTemplate("critical", [
            "{service} error rate jumped to {error_pct}% — database query errors after migration",
            "{service} failing on all writes to {db}: column '{column}' does not exist",
        ]),
        AlertTemplate("critical", [
            "Migration {migration_id} on {db} completed at {migration_time} — errors followed",
            "{service} 5xx rate at {error_pct}%: SQL errors referencing schema mismatch",
        ]),
        AlertTemplate("warning", [
            "Database migration {migration_id} applied to {db} at {migration_time}",
            "{service} query failures: {failed_queries}/{total_queries} queries failing",
        ]),
        AlertTemplate("info", [
            "Migration {migration_id}: ALTER TABLE {table} {migration_action}",
            "Schema version: {db} now at version {schema_version}",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "PSQLException: column \"{column}\" of relation \"{table}\" does not exist",
            "ERROR: relation \"{table}\" does not exist",
            "SQL Error: Unknown column '{column}' in 'field list'",
            "constraint violation: NOT NULL constraint failed: {table}.{column}",
        ]),
        LogTemplate("ERROR", [
            "Failed to execute query: {query_fragment} — {sql_error}",
            "ORM mapping error: entity {entity} expects column {column} but it was {migration_action}",
        ]),
        LogTemplate("WARN", [
            "Migration {migration_id} may be incompatible with running application version",
            "Schema drift detected: application expects {expected_schema}, database has {actual_schema}",
        ]),
        LogTemplate("INFO", [
            "Migration {migration_id} executed: {migration_action} on {table}.{column}",
            "Database connection healthy: {db} responding in {db_latency}ms",
            "Schema version check: expected={expected_schema}, actual={actual_schema}",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} started failing after migration {migration_id} ran on {db} at {migration_time}. "
        "Queries referencing {table}.{column} returned schema errors. The database itself was "
        "healthy and responding — the application code expected a schema that no longer matched. "
        "Rolling back the migration restored service.",

        "Outage on {service} caused by an incompatible schema migration. Migration {migration_id} "
        "altered {table} in {db}, but the running application version expected the old schema. "
        "{failed_queries} queries per second were failing with column-not-found errors.",

        "{service} error rate spiked to {error_pct}% coinciding with database migration "
        "{migration_id}. No code deploy occurred — the failure was a schema mismatch between "
        "the application's ORM mappings and the updated database structure. The database was "
        "reachable and healthy; the schema was the problem.",
    ]),
)

DATA_CORRUPTION_ARCHETYPE = Archetype(
    cause_class=CauseClass.DATA_CORRUPTION,
    fingerprint=FingerprintTemplate(
        latency_shift=["p99_spike"],
        error_rate_pattern=["bursty", "gradual"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["single_service"],
        onset_shape=["ramp"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    requires_db=True,
    alerts=[
        AlertTemplate("critical", [
            "{service} deserialization errors on {error_pct}% of reads from {db}",
            "{service} data integrity check failed: {corrupted_records} records with invalid checksums",
        ]),
        AlertTemplate("warning", [
            "{service} intermittent errors: specific record IDs failing, others succeeding",
            "Data validation failures on {service}: {failed_validations} records rejected",
        ]),
        AlertTemplate("warning", [
            "{service} error rate elevated but not time-correlated — errors appear on specific record access",
            "{db} returning unexpected data: encoding mismatch on {table}.{column}",
        ]),
        AlertTemplate("info", [
            "Affected records: IDs {corrupted_ids} in {table} (total corrupted: {corrupted_records})",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "Failed to deserialize record {record_id} from {table}: {deserialize_error}",
            "Data integrity check failed: expected checksum {expected_checksum}, got {actual_checksum}",
            "Foreign key violation: {table}.{column} references non-existent record {ref_id}",
            "JSON parse error on record {record_id}: unexpected token at position {position}",
        ]),
        LogTemplate("ERROR", [
            "Invalid data in {table}: record {record_id} has {column}='{bad_value}' (constraint: {constraint})",
            "Encoding error reading {table}.{column}: invalid UTF-8 sequence at byte {byte_offset}",
        ]),
        LogTemplate("WARN", [
            "Data inconsistency detected: {table} has {corrupted_records} records failing validation",
            "Record {record_id} corrupted: {column} contains truncated data ({actual_len}/{expected_len} bytes)",
        ]),
        LogTemplate("INFO", [
            "Data validation scan: {scanned_records} checked, {corrupted_records} corrupted, {clean_records} clean",
            "Corruption pattern: affected records were last modified between {corrupt_start} and {corrupt_end}",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} experienced intermittent read failures from {db}. Errors were data-dependent: "
        "specific record IDs consistently failed deserialization while others succeeded. "
        "{corrupted_records} records in {table} had invalid checksums. The schema was correct; "
        "the data values themselves were corrupted.",

        "Data integrity issues on {service}. Read errors appeared sporadically, correlated with "
        "access to specific records rather than time or traffic. Investigation found "
        "{corrupted_records} records with encoding errors in {table}.{column}. The corruption "
        "was traced to a batch job that wrote truncated data.",

        "{service} error rate showed a bursty pattern — errors appeared when users accessed "
        "specific records from {table} in {db}. Unlike a schema migration failure (where all "
        "queries to the table fail), only {corrupted_records} specific records were affected. "
        "Deserialization errors confirmed the data payload was corrupted.",
    ]),
)

# ---------------------------------------------------------------------------
# TIMING FAMILY
# ---------------------------------------------------------------------------

CLOCK_SKEW_ARCHETYPE = Archetype(
    cause_class=CauseClass.CLOCK_SKEW,
    fingerprint=FingerprintTemplate(
        latency_shift=["p99_spike"],
        error_rate_pattern=["bursty"],
        traffic_correlation=["uncorrelated"],
        topology_pattern=["single_service"],
        onset_shape=["step"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    alerts=[
        AlertTemplate("critical", [
            "{service} authentication failures: {error_pct}% of tokens rejected as expired",
            "etcd leader election instability on node {node}: clock skew detected",
        ]),
        AlertTemplate("warning", [
            "NTP offset on node {node}: {clock_offset}s (threshold: 1s)",
            "{service} pod {pod} clock drift: {clock_offset}s behind cluster time",
        ]),
        AlertTemplate("warning", [
            "{service} intermittent token validation failures — only on pod {pod} (node {node})",
            "Cache TTL anomaly: entries expiring {clock_offset}s early/late on node {node}",
        ]),
        AlertTemplate("info", [
            "chrony status on node {node}: stratum={stratum}, offset={clock_offset}s, state=not synchronized",
            "Kubernetes lease renewal: node {node} lease expired due to clock drift",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "Token validation failed: token issued at {token_issued} is in the future (current time: {current_time})",
            "JWT expired: exp={token_exp}, current={current_time}, skew exceeds tolerance of {tolerance}s",
            "Distributed lock acquisition failed: lease timestamp {lease_time} conflicts with local clock {current_time}",
        ]),
        LogTemplate("WARN", [
            "Clock skew detected: node {node} is {clock_offset}s {clock_direction} cluster reference",
            "NTP synchronization lost on node {node}: last sync {last_sync_ago} ago",
            "Timestamp ordering violation: event at {event_time} received after event at {later_time}",
        ]),
        LogTemplate("INFO", [
            "NTP status: server={ntp_server}, offset={clock_offset}s, jitter={jitter}ms",
            "chrony: forcing clock synchronization on node {node}",
            "Request trace: timestamps out of order across pods — clock skew suspected",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} experienced intermittent authentication failures on specific pods. JWT tokens "
        "were rejected as expired or issued in the future, depending on which pod handled the "
        "request. NTP status showed node {node} was {clock_offset}s off from cluster time. "
        "Other nodes were unaffected.",

        "etcd leader election instability traced to clock skew on node {node}. The node's system "
        "clock was {clock_offset}s {clock_direction} reference time. {service} pods on this node "
        "were failing time-sensitive operations: token validation, cache TTLs, distributed locks.",

        "Intermittent errors on {service} that depended on which pod handled the request. Pods "
        "on node {node} failed {error_pct}% of token validations while pods on other nodes "
        "succeeded. Root cause was NTP desynchronization: node {node} clock was {clock_offset}s "
        "{clock_direction}. Resyncing chrony resolved the issue.",
    ]),
)

RACE_CONDITION_ARCHETYPE = Archetype(
    cause_class=CauseClass.RACE_CONDITION,
    fingerprint=FingerprintTemplate(
        latency_shift=["p99_spike"],
        error_rate_pattern=["bursty"],
        traffic_correlation=["correlated"],
        topology_pattern=["single_service"],
        onset_shape=["ramp"],
        duration_pattern=["persistent"],
        deploy_correlation=False,
        amplification_ratio_range=(0.9, 1.1),
    ),
    alerts=[
        AlertTemplate("critical", [
            "{service} data inconsistency detected: {duplicate_count} duplicate records in {table}",
            "{service} intermittent failures: {error_pct}% error rate under concurrent load",
        ]),
        AlertTemplate("warning", [
            "{service} lost update detected: {lost_updates} writes overwritten by concurrent transactions",
            "Unique constraint violation rate elevated: {constraint_violations}/min on {table}",
        ]),
        AlertTemplate("warning", [
            "{service} error rate correlates with concurrency: {error_pct}% at {concurrency} concurrent requests",
            "Deadlock detected on {db}: {deadlock_count} deadlocks in last {window}m",
        ]),
        AlertTemplate("info", [
            "{service} concurrency level: {concurrency} simultaneous workers (scaled up from {baseline_concurrency})",
        ]),
    ],
    logs=[
        LogTemplate("ERROR", [
            "Duplicate key violation: INSERT INTO {table} ({column}) VALUES ('{value}') — already exists",
            "Lost update: record {record_id} was modified by another transaction between read and write",
            "Deadlock detected: transaction {txn_id} waiting for lock held by transaction {blocking_txn_id}",
            "Optimistic lock failure: version mismatch on {table}.{record_id} (expected {expected_version}, got {actual_version})",
        ]),
        LogTemplate("WARN", [
            "Concurrent modification of {table}.{record_id}: {concurrent_count} writers in {window}ms window",
            "Race condition suspected: {endpoint} fails only under concurrent execution",
            "Retry #{attempt} for {table} update after optimistic lock failure",
        ]),
        LogTemplate("INFO", [
            "Worker pool: {concurrency} active workers processing {rps} req/s",
            "Transaction isolation level: {isolation_level} on {db}",
            "Lock wait time: avg {lock_wait}ms, max {max_lock_wait}ms over last {window}m",
        ]),
    ],
    summary=SummaryTemplate(templates=[
        "{service} experienced intermittent data inconsistencies under concurrent load. "
        "Duplicate records and lost updates appeared when concurrency exceeded {concurrency} "
        "simultaneous workers. The same operations succeeded under serial execution. The "
        "race was in the read-modify-write sequence on {table} — no locking was in place.",

        "Sporadic errors on {service} that correlated with concurrency level, not traffic volume. "
        "At {baseline_concurrency} workers: no errors. At {concurrency} workers: {error_pct}% "
        "failure rate. Errors were duplicate key violations and lost updates on {table}, "
        "indicating a race condition in the write path.",

        "{service} data integrity issues under load. Error rate was bursty and depended on "
        "request timing, not volume. {duplicate_count} duplicate records were created in {table} "
        "when concurrent requests hit the same create-if-not-exists logic. Adding a database-level "
        "unique constraint confirmed the race and provided the fix.",
    ]),
)

# ---------------------------------------------------------------------------
# Complete archetype registry
# ---------------------------------------------------------------------------

ARCHETYPES: dict[CauseClass, Archetype] = {
    CauseClass.RETRY_STORM: RETRY_STORM_ARCHETYPE,
    CauseClass.DOWNSTREAM_SLOWDOWN: DOWNSTREAM_SLOWDOWN_ARCHETYPE,
    CauseClass.DEPENDENCY_OUTAGE: DEPENDENCY_OUTAGE_ARCHETYPE,
    CauseClass.DNS_RESOLUTION_FAILURE: DNS_RESOLUTION_FAILURE_ARCHETYPE,
    CauseClass.CERTIFICATE_EXPIRY: CERTIFICATE_EXPIRY_ARCHETYPE,
    CauseClass.CONNECTION_POOL_EXHAUSTION: CONNECTION_POOL_EXHAUSTION_ARCHETYPE,
    CauseClass.MEMORY_LEAK: MEMORY_LEAK_ARCHETYPE,
    CauseClass.TRAFFIC_ALLOCATION: TRAFFIC_ALLOCATION_ARCHETYPE,
    CauseClass.CPU_THROTTLING: CPU_THROTTLING_ARCHETYPE,
    CauseClass.DISK_PRESSURE: DISK_PRESSURE_ARCHETYPE,
    CauseClass.CODE_REGRESSION: CODE_REGRESSION_ARCHETYPE,
    CauseClass.CONFIG_REGRESSION: CONFIG_REGRESSION_ARCHETYPE,
    CauseClass.NETWORK_PARTITION: NETWORK_PARTITION_ARCHETYPE,
    CauseClass.LOAD_BALANCER_MISCONFIGURATION: LOAD_BALANCER_MISCONFIGURATION_ARCHETYPE,
    CauseClass.TRAFFIC_SPIKE: TRAFFIC_SPIKE_ARCHETYPE,
    CauseClass.CASCADING_TIMEOUT: CASCADING_TIMEOUT_ARCHETYPE,
    CauseClass.SCHEMA_MIGRATION_FAILURE: SCHEMA_MIGRATION_FAILURE_ARCHETYPE,
    CauseClass.DATA_CORRUPTION: DATA_CORRUPTION_ARCHETYPE,
    CauseClass.CLOCK_SKEW: CLOCK_SKEW_ARCHETYPE,
    CauseClass.RACE_CONDITION: RACE_CONDITION_ARCHETYPE,
}
