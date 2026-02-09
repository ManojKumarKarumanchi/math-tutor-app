"""
Trace Logger: Tracks agent execution for debugging and transparency.
"""


class TraceLogger:
    """Trace logger for agent execution."""

    def __init__(self):
        self.logs = []

    def log(self, agent: str, message: str, details: dict = None):
        """Add a log entry. details: optional dict for extra info."""
        entry = {"agent": agent, "message": message}
        if details:
            entry["details"] = details
        self.logs.append(entry)

    def get_logs(self) -> list:
        """Return all log entries."""
        return self.logs

    def clear(self):
        """Clear all logs."""
        self.logs = []

    def get_summary(self) -> str:
        """Return a formatted summary of all logs."""
        summary = []
        for log in self.logs:
            summary.append(f"**{log['agent']}**: {log['message']}")
            if "details" in log:
                summary.append(f"  Details: {log['details']}")
        return "\n".join(summary)
