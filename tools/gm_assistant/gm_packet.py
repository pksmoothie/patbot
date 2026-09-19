"""Generate a concise weekly GM packet from the latest validated snapshot.

No network access and no transaction execution. The packet reuses the same local
lineup, waiver, streaming, and trade engines used by the dashboard.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from gm_schema import BASE
from gm_snapshot import load_snapshot
from gm_lineup import analyze as analyze_lineup, save_analysis
from gm_waivers import analyze_waivers, save_waiver_analysis
from gm_streaming import analyze_streaming, route_waivers, save_streaming_analysis
from gm_trades import analyze_trades, save_trade_analysis


OUTPUT = BASE / "data" / "GM_PACKET.md"


def text(value, default="Unavailable"):
    return default if value is None or value == "" else str(value)


def number(value):
    return f"{value:.2f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "Unavailable"


def probability(value):
    return f"{value:.1%}" if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1 else "Unavailable"


def esc(value):
    return text(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    if not rows:
        return ["_None._", ""]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(esc(v) for v in row) + " |")
    out.append("")
    return out


def build_packet(snapshot, lineup, waivers, streaming, trades):
    lines = [
        "# GM Packet",
        "",
        f"Snapshot: `{snapshot['snapshot_id']}`  ",
        f"FantasyPros refreshed: {snapshot['refreshed_at']}  ",
        f"Packet generated: {lineup['analysis_timestamp']}  ",
        f"League key: `{snapshot['league_key']}`",
        "",
        "## This Week",
        "",
        f"Strategy: **{lineup['strategy']}**  ",
        f"GM projected score: **{number(lineup['recommended_projection'])}**  ",
        f"FantasyPros current-lineup win probability: **{probability(lineup['win_probability'])}**",
        "",
    ]

    if lineup["lineup_changes"]:
        lines.append("### Lineup changes")
        lines.append("")
        for change in lineup["lineup_changes"]:
            lines.append(f"- **{change['action']}** ({change['confidence']} confidence): {change['explanation']}")
        lines.append("")
    else:
        lines += ["### Lineup changes", "", "No lineup change recommended.", ""]

    lines += ["### Recommended lineup", ""]
    lines += table(
        ["Slot", "Player", "Projected points"],
        [(r["slot"], r["name"], number(r.get("points"))) for r in lineup["recommended_lineup"]],
    )

    lines += ["## Waivers", "", f"Overall call: **{waivers['overall']}**", ""]
    claim_rows = []
    for row in waivers.get("claim_plan", []):
        faab = row.get("faab", {})
        claim_rows.append((
            row.get("priority"), row.get("add"), row.get("drop") or "NO CLAIM",
            row.get("position"), faab.get("recommended", 0), row.get("confidence"),
        ))
    lines += table(["Priority", "Add", "Drop", "Pos", "FAAB", "Confidence"], claim_rows)

    lines += ["### Drop hierarchy", ""]
    drop_rows = [
        (i, p["name"], p["classification"], p.get("ros_rank"), p.get("drop_reason"))
        for i, p in enumerate(waivers.get("drop_hierarchy", []), 1)
        if p.get("classification") != "Core hold"
    ]
    lines += table(["Order", "Player", "Class", "ROS rank", "Reason"], drop_rows)

    week = streaming.get("current_week", {}).get("week")
    lines += ["## Streaming", "", f"Playing week: **{text(week)}**", ""]
    stream_rows = []
    for call in streaming.get("calls", []):
        stream_rows.append((
            call.get("position"), call.get("current"), call.get("recommendation"),
            call.get("add") or "—", call.get("drop") or "—", number(call.get("this_week_edge")),
            call.get("confidence"),
        ))
    lines += table(["Pos", "Current", "Call", "Add", "Drop", "Week edge", "Confidence"], stream_rows)

    lines += ["## Trades", ""]
    if not trades.get("enabled"):
        lines += ["Trade recommendations are currently disabled because required current inputs are incomplete or stale.", ""]
    elif trades.get("priority_targets"):
        trade_rows = []
        for target in trades["priority_targets"]:
            proposal = target["proposal"]
            trade_rows.append((
                target.get("priority"), target.get("target"), target.get("owner"),
                ", ".join(proposal.get("give", [])), ", ".join(proposal.get("receive", [])),
                proposal.get("tier"), proposal.get("confidence"),
            ))
        lines += table(["Priority", "Target", "Owner", "Give", "Receive", "Tier", "Confidence"], trade_rows)
    else:
        lines += ["No verified send-ready trade target clears the current gates.", ""]

    research = trades.get("research_candidates", [])
    if research:
        lines += ["### Trade research", ""]
        lines += table(
            ["Partner", "Give", "Receive", "Reason"],
            [
                (r.get("owner"), ", ".join(r.get("give", [])), ", ".join(r.get("receive", [])), r.get("reason"))
                for r in research
            ],
        )

    warnings = list(dict.fromkeys(
        lineup.get("warnings", []) + waivers.get("warnings", []) + streaming.get("warnings", []) + trades.get("warnings", [])
    ))
    lines += ["## Important limitations", ""]
    for warning in warnings:
        lines.append(f"- {warning}")
    lines += [
        "",
        "---",
        "This packet is decision support only. Yahoo lineup, waiver, FAAB, and trade actions remain manual.",
        "",
    ]
    return "\n".join(lines)


def main():
    snapshot = load_snapshot()
    lineup = analyze_lineup(snapshot)
    waivers = analyze_waivers(snapshot)
    streaming = analyze_streaming(snapshot, waiver_analysis=waivers)
    routed_waivers = route_waivers(waivers, streaming)
    trades = analyze_trades(snapshot, waiver_analysis=waivers, streaming_analysis=streaming)

    save_analysis(lineup, BASE / "data")
    save_waiver_analysis(routed_waivers, BASE / "data")
    save_streaming_analysis(streaming, BASE / "data")
    save_trade_analysis(trades, BASE / "data")

    output = build_packet(snapshot, lineup, routed_waivers, streaming, trades)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.parent / (".packet-" + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(output, encoding="utf-8")
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Generated GM packet: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
