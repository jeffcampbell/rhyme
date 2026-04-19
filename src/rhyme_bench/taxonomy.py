"""
Cause-class taxonomy for Rhyme v1.

20 cause classes across 8 families. Confusable pairs are explicitly mapped
with tier assignments. Every class has controlled remediation vocabulary.

Taxonomy version is pinned to the benchmark version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CauseFamily(str, Enum):
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    DEPLOY = "deploy"
    NETWORK = "network"
    TRAFFIC = "traffic"
    AMPLIFICATION = "amplification"
    DATA_SCHEMA = "data_schema"
    TIMING = "timing"


class CauseClass(str, Enum):
    # --- dependency family ---
    RETRY_STORM = "retry_storm"
    DOWNSTREAM_SLOWDOWN = "downstream_slowdown"
    DEPENDENCY_OUTAGE = "dependency_outage"
    DNS_RESOLUTION_FAILURE = "dns_resolution_failure"
    CERTIFICATE_EXPIRY = "certificate_expiry"
    CONNECTION_POOL_EXHAUSTION = "connection_pool_exhaustion"

    # --- resource family ---
    MEMORY_LEAK = "memory_leak"
    TRAFFIC_ALLOCATION = "traffic_allocation"
    CPU_THROTTLING = "cpu_throttling"
    DISK_PRESSURE = "disk_pressure"

    # --- deploy family ---
    CODE_REGRESSION = "code_regression"
    CONFIG_REGRESSION = "config_regression"

    # --- network family ---
    NETWORK_PARTITION = "network_partition"
    LOAD_BALANCER_MISCONFIGURATION = "load_balancer_misconfiguration"

    # --- traffic family ---
    TRAFFIC_SPIKE = "traffic_spike"

    # --- amplification family ---
    CASCADING_TIMEOUT = "cascading_timeout"

    # --- data/schema family ---
    SCHEMA_MIGRATION_FAILURE = "schema_migration_failure"
    DATA_CORRUPTION = "data_corruption"

    # --- timing family ---
    CLOCK_SKEW = "clock_skew"
    RACE_CONDITION = "race_condition"


CAUSE_FAMILY_MAP: dict[CauseClass, CauseFamily] = {
    CauseClass.RETRY_STORM: CauseFamily.DEPENDENCY,
    CauseClass.DOWNSTREAM_SLOWDOWN: CauseFamily.DEPENDENCY,
    CauseClass.DEPENDENCY_OUTAGE: CauseFamily.DEPENDENCY,
    CauseClass.DNS_RESOLUTION_FAILURE: CauseFamily.DEPENDENCY,
    CauseClass.CERTIFICATE_EXPIRY: CauseFamily.DEPENDENCY,
    CauseClass.CONNECTION_POOL_EXHAUSTION: CauseFamily.DEPENDENCY,
    CauseClass.MEMORY_LEAK: CauseFamily.RESOURCE,
    CauseClass.TRAFFIC_ALLOCATION: CauseFamily.RESOURCE,
    CauseClass.CPU_THROTTLING: CauseFamily.RESOURCE,
    CauseClass.DISK_PRESSURE: CauseFamily.RESOURCE,
    CauseClass.CODE_REGRESSION: CauseFamily.DEPLOY,
    CauseClass.CONFIG_REGRESSION: CauseFamily.DEPLOY,
    CauseClass.NETWORK_PARTITION: CauseFamily.NETWORK,
    CauseClass.LOAD_BALANCER_MISCONFIGURATION: CauseFamily.NETWORK,
    CauseClass.TRAFFIC_SPIKE: CauseFamily.TRAFFIC,
    CauseClass.CASCADING_TIMEOUT: CauseFamily.AMPLIFICATION,
    CauseClass.SCHEMA_MIGRATION_FAILURE: CauseFamily.DATA_SCHEMA,
    CauseClass.DATA_CORRUPTION: CauseFamily.DATA_SCHEMA,
    CauseClass.CLOCK_SKEW: CauseFamily.TIMING,
    CauseClass.RACE_CONDITION: CauseFamily.TIMING,
}


class ConfusabilityTier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# Which pairs are confusable, and at what tier.
# Hard = same symptoms, different proximal cause, requires careful reasoning to distinguish.
# Medium = overlapping symptoms but at least one clear discriminating signal.
CONFUSABLE_PAIRS: list[tuple[CauseClass, CauseClass, ConfusabilityTier]] = [
    # --- hard pairs (within-family or very symptom-similar) ---
    (CauseClass.RETRY_STORM, CauseClass.DOWNSTREAM_SLOWDOWN, ConfusabilityTier.HARD),
    (CauseClass.RETRY_STORM, CauseClass.CASCADING_TIMEOUT, ConfusabilityTier.HARD),
    (CauseClass.MEMORY_LEAK, CauseClass.TRAFFIC_ALLOCATION, ConfusabilityTier.HARD),
    (CauseClass.CODE_REGRESSION, CauseClass.CONFIG_REGRESSION, ConfusabilityTier.HARD),
    (CauseClass.DOWNSTREAM_SLOWDOWN, CauseClass.CONNECTION_POOL_EXHAUSTION, ConfusabilityTier.HARD),
    (CauseClass.SCHEMA_MIGRATION_FAILURE, CauseClass.DATA_CORRUPTION, ConfusabilityTier.HARD),
    (CauseClass.CLOCK_SKEW, CauseClass.RACE_CONDITION, ConfusabilityTier.HARD),
    (CauseClass.CPU_THROTTLING, CauseClass.TRAFFIC_ALLOCATION, ConfusabilityTier.HARD),
    (CauseClass.DNS_RESOLUTION_FAILURE, CauseClass.NETWORK_PARTITION, ConfusabilityTier.HARD),
    (CauseClass.DEPENDENCY_OUTAGE, CauseClass.NETWORK_PARTITION, ConfusabilityTier.HARD),
    (CauseClass.DEPENDENCY_OUTAGE, CauseClass.DOWNSTREAM_SLOWDOWN, ConfusabilityTier.HARD),

    # --- medium pairs (cross-family, partial overlap) ---
    (CauseClass.DOWNSTREAM_SLOWDOWN, CauseClass.CASCADING_TIMEOUT, ConfusabilityTier.MEDIUM),
    (CauseClass.MEMORY_LEAK, CauseClass.CODE_REGRESSION, ConfusabilityTier.MEDIUM),
    (CauseClass.TRAFFIC_SPIKE, CauseClass.TRAFFIC_ALLOCATION, ConfusabilityTier.HARD),
    (CauseClass.NETWORK_PARTITION, CauseClass.CERTIFICATE_EXPIRY, ConfusabilityTier.MEDIUM),
    (CauseClass.LOAD_BALANCER_MISCONFIGURATION, CauseClass.NETWORK_PARTITION, ConfusabilityTier.MEDIUM),
    (CauseClass.CONFIG_REGRESSION, CauseClass.SCHEMA_MIGRATION_FAILURE, ConfusabilityTier.MEDIUM),
    (CauseClass.DISK_PRESSURE, CauseClass.MEMORY_LEAK, ConfusabilityTier.MEDIUM),
    (CauseClass.DATA_CORRUPTION, CauseClass.CODE_REGRESSION, ConfusabilityTier.MEDIUM),
    (CauseClass.CONNECTION_POOL_EXHAUSTION, CauseClass.CPU_THROTTLING, ConfusabilityTier.MEDIUM),
]


@dataclass(frozen=True)
class Remediation:
    """Controlled remediation vocabulary for a cause class."""

    canonical: str
    masks_symptom: str
    would_worsen: str | None = None


REMEDIATIONS: dict[CauseClass, Remediation] = {
    CauseClass.RETRY_STORM: Remediation(
        canonical="Apply circuit breaker or exponential backoff on the retrying service",
        masks_symptom="Restart the retrying service pods (clears in-flight retries temporarily)",
        would_worsen="Scale up the downstream service (feeds more capacity to the storm)",
    ),
    CauseClass.DOWNSTREAM_SLOWDOWN: Remediation(
        canonical="Identify and fix the slow downstream dependency (query optimization, scaling, etc.)",
        masks_symptom="Increase timeout values on the calling service",
    ),
    CauseClass.DEPENDENCY_OUTAGE: Remediation(
        canonical="Activate fallback or graceful degradation for the unavailable dependency",
        masks_symptom="Retry requests to the dead dependency with aggressive timeouts (delays failure, doesn't fix it)",
        would_worsen="Scale up your service (the bottleneck is the dependency, not your capacity)",
    ),
    CauseClass.DNS_RESOLUTION_FAILURE: Remediation(
        canonical="Fix DNS configuration or restore the DNS provider/server",
        masks_symptom="Add static host entries or IP-based fallbacks to bypass DNS",
        would_worsen="Increase DNS TTL (delays recovery when DNS is restored)",
    ),
    CauseClass.CERTIFICATE_EXPIRY: Remediation(
        canonical="Renew the expired TLS certificate and redeploy it",
        masks_symptom="Disable TLS verification on clients (restores connectivity but removes security)",
        would_worsen="Restart pods (certificate is still expired after restart)",
    ),
    CauseClass.CONNECTION_POOL_EXHAUSTION: Remediation(
        canonical="Fix connection leak in application code or tune pool size and eviction settings",
        masks_symptom="Restart pods to reset the connection pool",
        would_worsen="Increase max pool size without fixing the leak (delays exhaustion, increases DB load)",
    ),
    CauseClass.MEMORY_LEAK: Remediation(
        canonical="Identify and fix the leaking allocation in the application code",
        masks_symptom="Increase memory limits or restart pods on a schedule",
        would_worsen="Scale horizontally without fixing the leak (multiplies leaked memory)",
    ),
    CauseClass.TRAFFIC_ALLOCATION: Remediation(
        canonical="Scale the service horizontally or vertically to match traffic demand",
        masks_symptom="Restart pods (briefly frees memory but traffic refills it)",
    ),
    CauseClass.CPU_THROTTLING: Remediation(
        canonical="Increase CPU limits or optimize the hot code path consuming excess CPU",
        masks_symptom="Scale horizontally to distribute load across more pods",
        would_worsen="Lower CPU limits to save cost (worsens throttling)",
    ),
    CauseClass.DISK_PRESSURE: Remediation(
        canonical="Expand storage volume or clean up accumulated data (old logs, temp files, WAL segments)",
        masks_symptom="Restart the pod (may free temp files but does not address growth)",
        would_worsen="Disable log rotation (accelerates disk fill)",
    ),
    CauseClass.CODE_REGRESSION: Remediation(
        canonical="Rollback to the previous known-good deployment",
        masks_symptom="Restart pods (may temporarily clear corrupted state)",
        would_worsen="Roll forward with a rushed hotfix that isn't tested",
    ),
    CauseClass.CONFIG_REGRESSION: Remediation(
        canonical="Revert the configuration change to the previous known-good values",
        masks_symptom="Restart pods (reloads config but the bad values are still there)",
        would_worsen="Roll back the code (the code is fine; the config is the problem)",
    ),
    CauseClass.NETWORK_PARTITION: Remediation(
        canonical="Restore network connectivity (fix routing rules, security groups, or physical link)",
        masks_symptom="Fail over to replicas in a different availability zone",
    ),
    CauseClass.LOAD_BALANCER_MISCONFIGURATION: Remediation(
        canonical="Correct the load balancer configuration (routing rules, health check settings, backend pool)",
        masks_symptom="Bypass the load balancer with direct service-to-service routing",
        would_worsen="Scale up backends (the LB isn't routing to them correctly anyway)",
    ),
    CauseClass.TRAFFIC_SPIKE: Remediation(
        canonical="Scale the service horizontally to absorb the legitimate traffic increase",
        masks_symptom="Enable rate limiting (drops legitimate user requests)",
    ),
    CauseClass.CASCADING_TIMEOUT: Remediation(
        canonical="Reduce timeout values on upstream callers to fail fast and shed load",
        masks_symptom="Increase timeout values (allows requests to pile up longer)",
        would_worsen="Add retries on top of existing timeouts (converts timeout cascade into retry storm)",
    ),
    CauseClass.SCHEMA_MIGRATION_FAILURE: Remediation(
        canonical="Roll back the failed schema migration and fix compatibility issues before re-running",
        masks_symptom="Restart the application (it will hit the same schema mismatch)",
        would_worsen="Run the migration forward without fixing it (may corrupt more data)",
    ),
    CauseClass.DATA_CORRUPTION: Remediation(
        canonical="Identify the corruption source, restore from last known-good backup, replay clean writes",
        masks_symptom="Add application-level validation to skip corrupted records",
        would_worsen="Re-run the job that caused the corruption (creates more bad data)",
    ),
    CauseClass.CLOCK_SKEW: Remediation(
        canonical="Sync clocks via NTP/chrony and restart affected services to pick up corrected time",
        masks_symptom="Increase tolerance windows for time-sensitive operations",
        would_worsen="Disable time-based validation (masks the symptom, introduces consistency risks)",
    ),
    CauseClass.RACE_CONDITION: Remediation(
        canonical="Fix the race condition with proper synchronization (locks, atomic operations, idempotency)",
        masks_symptom="Reduce concurrency (fewer workers/threads) to lower collision probability",
        would_worsen="Increase concurrency to improve throughput (raises collision rate)",
    ),
}


@dataclass(frozen=True)
class CauseClassInfo:
    cause_class: CauseClass
    family: CauseFamily
    description: str
    distinguishing_signals: list[str]
    remediation: Remediation


TAXONOMY: dict[CauseClass, CauseClassInfo] = {
    CauseClass.RETRY_STORM: CauseClassInfo(
        cause_class=CauseClass.RETRY_STORM,
        family=CauseFamily.DEPENDENCY,
        description=(
            "A service retries failed requests to a dependency aggressively, creating "
            "exponential amplification. The dependency may be healthy or only slightly degraded; "
            "the storm itself becomes the primary load driver."
        ),
        distinguishing_signals=[
            "Request rate to downstream far exceeds inbound request rate (amplification ratio >3x)",
            "Downstream error rate rises in lockstep with caller retry rate",
            "Removing/throttling the caller immediately relieves downstream",
            "Latency distribution is bimodal: fast failures + timeout-ceiling hits",
        ],
        remediation=REMEDIATIONS[CauseClass.RETRY_STORM],
    ),
    CauseClass.DOWNSTREAM_SLOWDOWN: CauseClassInfo(
        cause_class=CauseClass.DOWNSTREAM_SLOWDOWN,
        family=CauseFamily.DEPENDENCY,
        description=(
            "A downstream dependency becomes genuinely slow (DB query regression, third-party "
            "API degradation, network partition). The calling service sees elevated latency and "
            "errors proportional to its dependency on that service."
        ),
        distinguishing_signals=[
            "Downstream latency rises before caller latency rises (temporal ordering)",
            "Caller request rate to downstream matches inbound rate (no amplification)",
            "Downstream shows internal resource saturation (CPU, query time, connection count)",
            "Latency distribution shifts uniformly, not bimodally",
        ],
        remediation=REMEDIATIONS[CauseClass.DOWNSTREAM_SLOWDOWN],
    ),
    CauseClass.DEPENDENCY_OUTAGE: CauseClassInfo(
        cause_class=CauseClass.DEPENDENCY_OUTAGE,
        family=CauseFamily.DEPENDENCY,
        description=(
            "An external or third-party dependency is completely unavailable — not slow, not "
            "intermittent, but down. The dependency's own status page confirms an outage. Your "
            "service and network are healthy; the problem is entirely on the dependency's side."
        ),
        distinguishing_signals=[
            "Dependency returns errors immediately (connection refused, 503) rather than timing out",
            "Dependency's own status page or health endpoint confirms they are down",
            "Your network connectivity to other services is unaffected",
            "The failure started at the same time the dependency's outage began (check their status page timeline)",
        ],
        remediation=REMEDIATIONS[CauseClass.DEPENDENCY_OUTAGE],
    ),
    CauseClass.DNS_RESOLUTION_FAILURE: CauseClassInfo(
        cause_class=CauseClass.DNS_RESOLUTION_FAILURE,
        family=CauseFamily.DEPENDENCY,
        description=(
            "DNS resolution fails for one or more service endpoints. Services cannot discover "
            "their dependencies. Affects all services relying on the unresolvable names, often "
            "with immediate-onset total failure rather than degradation."
        ),
        distinguishing_signals=[
            "All connections to specific hostnames fail simultaneously",
            "Errors are 'name resolution failed' or 'NXDOMAIN', not connection refused or timeout",
            "Services using IP-based connections or cached DNS are unaffected",
            "Onset is instant across all affected services (not gradual)",
        ],
        remediation=REMEDIATIONS[CauseClass.DNS_RESOLUTION_FAILURE],
    ),
    CauseClass.CERTIFICATE_EXPIRY: CauseClassInfo(
        cause_class=CauseClass.CERTIFICATE_EXPIRY,
        family=CauseFamily.DEPENDENCY,
        description=(
            "A TLS certificate expires, causing all TLS handshakes to the affected endpoint "
            "to fail. The underlying service is healthy but unreachable over HTTPS/mTLS."
        ),
        distinguishing_signals=[
            "TLS handshake failures with 'certificate has expired' or 'x509' errors",
            "The service is reachable via HTTP (if enabled) but not HTTPS",
            "Onset coincides with a specific calendar time (cert expiry), not a deploy or traffic change",
            "Only connections to the specific endpoint with the expired cert are affected",
        ],
        remediation=REMEDIATIONS[CauseClass.CERTIFICATE_EXPIRY],
    ),
    CauseClass.CONNECTION_POOL_EXHAUSTION: CauseClassInfo(
        cause_class=CauseClass.CONNECTION_POOL_EXHAUSTION,
        family=CauseFamily.DEPENDENCY,
        description=(
            "A connection pool (database, HTTP, gRPC) runs out of available connections. New "
            "requests queue waiting for a connection, causing latency spikes and timeouts. Often "
            "caused by a connection leak or sudden increase in concurrent requests."
        ),
        distinguishing_signals=[
            "Connection pool active count at or near max, idle count at zero",
            "Latency spikes are in the 'waiting for connection' phase, not the request itself",
            "The downstream (DB, service) is healthy and responsive for existing connections",
            "Pool wait timeout errors appear in logs before any downstream errors",
        ],
        remediation=REMEDIATIONS[CauseClass.CONNECTION_POOL_EXHAUSTION],
    ),
    CauseClass.MEMORY_LEAK: CauseClassInfo(
        cause_class=CauseClass.MEMORY_LEAK,
        family=CauseFamily.RESOURCE,
        description=(
            "Application code allocates memory that is never freed. Memory usage grows "
            "monotonically regardless of traffic, eventually hitting OOM limits. Restarts "
            "temporarily fix the symptom but the pattern repeats."
        ),
        distinguishing_signals=[
            "Memory usage grows monotonically even during low-traffic periods",
            "Memory does not correlate with request rate or traffic volume",
            "Time-to-OOM is consistent across restarts (deterministic leak rate)",
            "Heap profiling shows specific object types growing unboundedly",
        ],
        remediation=REMEDIATIONS[CauseClass.MEMORY_LEAK],
    ),
    CauseClass.TRAFFIC_ALLOCATION: CauseClassInfo(
        cause_class=CauseClass.TRAFFIC_ALLOCATION,
        family=CauseFamily.RESOURCE,
        description=(
            "Traffic increases beyond the service's provisioned capacity. Memory usage rises "
            "proportionally with request volume. OOM kills occur during peak traffic and resolve "
            "when traffic subsides."
        ),
        distinguishing_signals=[
            "Memory usage correlates strongly with inbound request rate",
            "OOM kills cluster during traffic peaks, not during idle periods",
            "Adding replicas immediately relieves pressure",
            "Memory drops back to baseline when traffic subsides (no monotonic growth)",
        ],
        remediation=REMEDIATIONS[CauseClass.TRAFFIC_ALLOCATION],
    ),
    CauseClass.CPU_THROTTLING: CauseClassInfo(
        cause_class=CauseClass.CPU_THROTTLING,
        family=CauseFamily.RESOURCE,
        description=(
            "A service's CPU usage hits its cgroup limit, causing the kernel to throttle it. "
            "Request processing slows proportionally to the throttle ratio. The service never "
            "crashes but becomes progressively slower."
        ),
        distinguishing_signals=[
            "CPU throttle_count metric is non-zero and rising",
            "Container CPU usage is pegged at the cgroup limit (not below it)",
            "Latency increases are proportional to CPU throttle percentage",
            "No memory pressure, no OOM kills — the service stays running but slow",
        ],
        remediation=REMEDIATIONS[CauseClass.CPU_THROTTLING],
    ),
    CauseClass.DISK_PRESSURE: CauseClassInfo(
        cause_class=CauseClass.DISK_PRESSURE,
        family=CauseFamily.RESOURCE,
        description=(
            "A disk or volume fills up, causing write failures. Services that write logs, "
            "temp files, or data to the volume start failing. May trigger pod eviction "
            "if the node's ephemeral storage is exhausted."
        ),
        distinguishing_signals=[
            "Disk usage at or near 100% on the affected volume",
            "Write errors: 'no space left on device' or 'ENOSPC'",
            "Read operations continue to work; only writes fail",
            "Pod eviction events with reason 'DiskPressure' or 'EphemeralStorageExhausted'",
        ],
        remediation=REMEDIATIONS[CauseClass.DISK_PRESSURE],
    ),
    CauseClass.CODE_REGRESSION: CauseClassInfo(
        cause_class=CauseClass.CODE_REGRESSION,
        family=CauseFamily.DEPLOY,
        description=(
            "A code deployment introduces a bug: new error paths, performance regression, "
            "incorrect business logic. Symptoms begin precisely at deploy time and affect "
            "only the deployed service."
        ),
        distinguishing_signals=[
            "Error rate or latency change begins exactly at deploy timestamp",
            "Only the deployed service is affected; dependencies are healthy",
            "Rollback immediately resolves the issue",
            "New error types or stack traces appear that weren't present before deploy",
        ],
        remediation=REMEDIATIONS[CauseClass.CODE_REGRESSION],
    ),
    CauseClass.CONFIG_REGRESSION: CauseClassInfo(
        cause_class=CauseClass.CONFIG_REGRESSION,
        family=CauseFamily.DEPLOY,
        description=(
            "A configuration change (feature flag, env var, ConfigMap) introduces incorrect "
            "behavior. Symptoms begin at config propagation time. The code itself is unchanged; "
            "rolling back the code has no effect."
        ),
        distinguishing_signals=[
            "Onset coincides with a config change event (ConfigMap update, feature flag toggle)",
            "No code deploy occurred — the container image is unchanged",
            "Rolling back the code does not fix the issue; reverting the config does",
            "Affected behavior corresponds to the specific config key that changed",
        ],
        remediation=REMEDIATIONS[CauseClass.CONFIG_REGRESSION],
    ),
    CauseClass.NETWORK_PARTITION: CauseClassInfo(
        cause_class=CauseClass.NETWORK_PARTITION,
        family=CauseFamily.NETWORK,
        description=(
            "Network connectivity between services or availability zones is disrupted. "
            "Affected connections fail or time out. Services within the same partition "
            "continue to work normally."
        ),
        distinguishing_signals=[
            "Connectivity failures are between specific source/destination pairs, not universal",
            "Services within the same network segment are unaffected",
            "TCP connection establishment fails (SYN timeout), not application-level errors",
            "Traceroute or network diagnostic shows a hop failure or routing blackhole",
        ],
        remediation=REMEDIATIONS[CauseClass.NETWORK_PARTITION],
    ),
    CauseClass.LOAD_BALANCER_MISCONFIGURATION: CauseClassInfo(
        cause_class=CauseClass.LOAD_BALANCER_MISCONFIGURATION,
        family=CauseFamily.NETWORK,
        description=(
            "A load balancer routes traffic incorrectly: wrong backends, failed health checks "
            "removing healthy pods, uneven distribution, or incorrect path rules. Backend "
            "services themselves are healthy."
        ),
        distinguishing_signals=[
            "Backend services report low/normal load but clients see errors",
            "Health check failures on the LB don't correspond to actual service health",
            "Traffic distribution is uneven (some backends overloaded, others idle)",
            "Direct requests to backends succeed; requests through the LB fail",
        ],
        remediation=REMEDIATIONS[CauseClass.LOAD_BALANCER_MISCONFIGURATION],
    ),
    CauseClass.TRAFFIC_SPIKE: CauseClassInfo(
        cause_class=CauseClass.TRAFFIC_SPIKE,
        family=CauseFamily.TRAFFIC,
        description=(
            "Legitimate organic traffic surges beyond provisioned capacity. Unlike bot traffic, "
            "the request pattern is normal (varied endpoints, normal user agents, typical "
            "session behavior). The service is simply overwhelmed."
        ),
        distinguishing_signals=[
            "Request rate increases with normal distribution across endpoints",
            "User agents and session patterns match legitimate traffic",
            "Traffic correlates with a known event (sale, launch, marketing campaign)",
            "Scaling up resolves the issue with no residual errors",
        ],
        remediation=REMEDIATIONS[CauseClass.TRAFFIC_SPIKE],
    ),
    CauseClass.CASCADING_TIMEOUT: CauseClassInfo(
        cause_class=CauseClass.CASCADING_TIMEOUT,
        family=CauseFamily.AMPLIFICATION,
        description=(
            "A slow dependency causes upstream callers to hold connections open until timeout. "
            "These held connections exhaust the caller's capacity, which in turn times out "
            "its own callers. Propagates up the call chain without retry amplification."
        ),
        distinguishing_signals=[
            "Timeout propagation follows the call graph upstream (deepest dependency first)",
            "No amplification — request rates are 1:1 at each hop",
            "Latency at each tier equals the configured timeout value (hard-ceiling hits)",
            "Reducing timeout values on upstream callers breaks the cascade",
        ],
        remediation=REMEDIATIONS[CauseClass.CASCADING_TIMEOUT],
    ),
    CauseClass.SCHEMA_MIGRATION_FAILURE: CauseClassInfo(
        cause_class=CauseClass.SCHEMA_MIGRATION_FAILURE,
        family=CauseFamily.DATA_SCHEMA,
        description=(
            "A database schema migration fails or introduces incompatibilities. The application "
            "can't read/write data correctly because table structure, column types, or constraints "
            "don't match what the code expects."
        ),
        distinguishing_signals=[
            "Errors reference specific columns, tables, or constraints ('column X does not exist')",
            "Onset coincides with a migration execution event, not a code deploy",
            "Only operations touching the migrated table/schema are affected",
            "The database is reachable and healthy; the schema is the problem",
        ],
        remediation=REMEDIATIONS[CauseClass.SCHEMA_MIGRATION_FAILURE],
    ),
    CauseClass.DATA_CORRUPTION: CauseClassInfo(
        cause_class=CauseClass.DATA_CORRUPTION,
        family=CauseFamily.DATA_SCHEMA,
        description=(
            "Data in a datastore becomes invalid: wrong values, broken references, encoding "
            "errors. The application reads corrupted data and produces incorrect results or "
            "crashes on deserialization."
        ),
        distinguishing_signals=[
            "Errors are data-dependent: specific records/keys fail, others succeed",
            "Deserialization errors, checksum mismatches, or constraint violations on read",
            "The pattern is not time-correlated (errors appear when corrupted records are accessed)",
            "Schema is correct; the data values themselves are wrong",
        ],
        remediation=REMEDIATIONS[CauseClass.DATA_CORRUPTION],
    ),
    CauseClass.CLOCK_SKEW: CauseClassInfo(
        cause_class=CauseClass.CLOCK_SKEW,
        family=CauseFamily.TIMING,
        description=(
            "System clocks on one or more nodes drift out of sync. Time-sensitive operations "
            "fail: token validation, cache TTLs, distributed consensus, log ordering. Symptoms "
            "are intermittent and depend on which node handles the request."
        ),
        distinguishing_signals=[
            "Errors mention 'token expired', 'timestamp out of range', or 'clock skew detected'",
            "Symptoms are node-specific: requests to one pod fail, others succeed",
            "NTP/chrony status shows offset >1s on affected nodes",
            "Distributed consensus (etcd, Raft) reports leader election instability",
        ],
        remediation=REMEDIATIONS[CauseClass.CLOCK_SKEW],
    ),
    CauseClass.RACE_CONDITION: CauseClassInfo(
        cause_class=CauseClass.RACE_CONDITION,
        family=CauseFamily.TIMING,
        description=(
            "Concurrent operations interfere with each other due to missing synchronization. "
            "Symptoms are intermittent, depend on timing, and reproduce only under concurrent "
            "load. Data inconsistencies or unexpected errors appear sporadically."
        ),
        distinguishing_signals=[
            "Errors are intermittent and not reproducible under single-threaded execution",
            "Failure rate correlates with concurrency level (more workers = more failures)",
            "Data inconsistencies: duplicate records, lost updates, phantom reads",
            "The same operation succeeds or fails depending on timing",
        ],
        remediation=REMEDIATIONS[CauseClass.RACE_CONDITION],
    ),
}

TAXONOMY_VERSION = "1.0.0"
