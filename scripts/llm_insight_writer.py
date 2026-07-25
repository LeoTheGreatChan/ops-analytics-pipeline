"""
LLM insight writer — Phase 2

Replaces the Phase 1 template function (write_insight() in build_insights.py)
with a genuine Claude API call. This is deliberately a separate module so the
distinction stays visible in the codebase: statistics and the model produce
the numbers, this module's only job is turning verified numbers into a
sentence. It does not decide what's true, it writes up what the stats layer
already found.

Requires ANTHROPIC_API_KEY set as an environment variable (set this as a
secret in Render's dashboard, never commit it to the repo).

Model choice: claude-haiku-4-5-20251001. This task is a short, templated
completion (one sentence from a handful of numbers), not open-ended
reasoning, so the fastest/cheapest current model is the right tool, not the
strongest one. Revisit only if the sentence quality genuinely needs it.
"""

import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

PROMPT_TEMPLATE = """You are writing one operations-dashboard insight card from real statistics. Do not invent any number not given below.

Dimension: {dimension}
Segment: {segment}
Breach rate in this segment: {breach_rate}%
Overall baseline breach rate: {baseline}%
Relative lift vs baseline: {lift:+.0f}%
Suggested buffer (from real 90th-percentile-vs-median gap): {buffer}%
Sample size: {n} deliveries

Write exactly two sentences in plain English:
1. State the finding, bold the key percentage figures using **markdown**.
2. Recommend one concrete action. First decide what kind of factor this
   dimension represents: an EXTERNAL CONDITION (weather, traffic, vehicle
   type, geographic area, time, or a combination of these) that operations
   cannot control but can plan around, versus an INTERNAL PERFORMANCE DRIVER
   (something about the people, agents, or process carrying out the work)
   that can be directly managed or improved. For external conditions, a
   scheduling buffer is the appropriate action, use the bolded buffer
   percentage provided above. For internal performance drivers, recommend
   an operational response such as training, coaching, or reassignment
   instead, a scheduling buffer does not address the actual cause. If the
   lift is negative (this segment already outperforms baseline), state that
   no action is needed rather than recommending either.

Output only the two sentences, nothing else."""


def write_insight_llm(row, overall_breach_rate):
    """
    row: dict with keys dimension, segment_label, breach_rate (0-1), n,
         breach_lift_pct, suggested_buffer_pct
    overall_breach_rate: float, 0-1
    Returns: finding text (str) with **bold** markers, same format the
    frontend already parses.
    """
    prompt = PROMPT_TEMPLATE.format(
        dimension=row["dimension"],
        segment=row["segment_label"],
        breach_rate=round(row["breach_rate"] * 100, 1),
        baseline=round(overall_breach_rate * 100, 1),
        lift=row["breach_lift_pct"],
        buffer=row["suggested_buffer_pct"],
        n=int(row["n"]),
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def write_insight_batch(rows, overall_breach_rate):
    """Runs write_insight_llm for a list of finding rows. One API call per
    row, kept simple and debuggable; batching into a single multi-insight
    prompt is a possible later optimisation but adds parsing complexity
    that isn't worth it at this volume (typically 6-12 insights per run)."""
    return [
        {**row, "finding_text": write_insight_llm(row, overall_breach_rate)}
        for row in rows
    ]
