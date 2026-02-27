"""
ranking.py
----------
Builds and sorts a ranked leaderboard from a list of already-processed
candidate dictionaries (produced by the /api/leaderboard route in app.py).

Each candidate dict must contain at least:
    {
        "name": str,
        "score": float,          # skill_strength_score from extraction pipeline
        "metrics": dict,         # raw analysis_metrics
        "skills": dict,          # extracted_skills by category
        "recruiter_note": str    # LLM-generated paragraph
    }
"""

from typing import List, Dict


def get_domain_leaderboard(candidates: List[Dict]) -> List[Dict]:
    """
    Sort candidates by their skill_strength_score (descending) and attach
    a 1-based rank.  Ties are broken alphabetically by name.

    Args:
        candidates: list of processed candidate dicts (see module docstring).

    Returns:
        The same list, sorted and annotated with a 'rank' key.
    """
    ranked = sorted(
        candidates,
        key=lambda c: (-c.get("score", 0.0), c.get("name", "").lower()),
    )

    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank

    return ranked


def build_leaderboard_summary(ranked_candidates: List[Dict]) -> Dict:
    """
    Produce a compact summary dict useful for a dashboard header or email.

    Returns:
        {
            "total_candidates": int,
            "top_candidate": str,
            "average_score": float,
            "score_distribution": {"strong": int, "medium": int, "weak": int}
        }
    """
    if not ranked_candidates:
        return {
            "total_candidates": 0,
            "top_candidate": None,
            "average_score": 0.0,
            "score_distribution": {"strong": 0, "medium": 0, "weak": 0},
        }

    scores = [c.get("score", 0.0) for c in ranked_candidates]
    avg = round(sum(scores) / len(scores), 4)

    dist = {"strong": 0, "medium": 0, "weak": 0}
    for c in ranked_candidates:
        m = c.get("metrics", {})
        dist["strong"] += m.get("total_strong_skills", 0)
        dist["medium"] += m.get("total_medium_skills", 0)
        dist["weak"]   += m.get("total_weak_skills",   0)

    return {
        "total_candidates": len(ranked_candidates),
        "top_candidate":    ranked_candidates[0].get("name"),
        "average_score":    avg,
        "score_distribution": dist,
    }