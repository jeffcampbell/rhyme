Instructions for prose pool regeneration agents:

Each agent should write a JSON file to /Users/jeff/development/incident-corrbench/data/prose_pools/{cause_class}.json

The JSON must match this schema:
{
  "cause_class": "{cause_class}",
  "summaries": [20 strings],
  "alerts": [
    {"severity": "critical", "messages": [10 strings]},
    {"severity": "warning", "messages": [10 strings]},
    {"severity": "info", "messages": [5 strings]}
  ],
  "logs": [
    {"level": "ERROR", "messages": [15 strings]},
    {"level": "WARN", "messages": [12 strings]},
    {"level": "INFO", "messages": [8 strings]}
  ]
}

Style requirements:
- Summaries must sound like REAL cloud provider incident reports, not textbook descriptions
- Use timestamped update sequences like AWS: "1:06 PM PST We are investigating..."
- Use RCA/PIR format like Azure: "Between 09:21 and 17:32 UTC on 03 Sep 2020..."
- Use resolution format like GCP: "The issue with X has been resolved..."
- Reference real infrastructure: AWS regions (us-east-1, eu-west-1), Azure regions (West Europe, East US), GCP zones
- Use real service names and error codes
- Use template variables: {service}, {downstream}, {error_pct}, {latency}, etc.
- CRITICAL: Do NOT name the cause class directly in every summary
- Mix provider styles across summaries (some AWS-style, some Azure-style, some GCP-style)
