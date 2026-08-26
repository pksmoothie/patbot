from __future__ import annotations

from collections import Counter


_INSTALLED = False


def install_live_roster_context_patch() -> None:
    """Feed live PatBot player IDs into identity-aware draft strategy layers.

    The Streamlit room records roster IDs before calling recommend(). Tests and
    other callers may only provide position counts, so validate the cached IDs
    against those positions before using them. A stale/mismatched cache is
    ignored rather than allowing one prior roster to affect another call.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .draft import DraftEngine
    from .draft_state import last_roster_ids

    original_recommend = DraftEngine.recommend

    def recommend_with_roster_identity(
        self,
        current_pick: int,
        drafted_ids,
        roster_positions,
        top_n: int = 12,
    ):
        cached_ids = [str(x) for x in last_roster_ids()]
        supplied_positions = Counter(str(p).upper() for p in roster_positions)

        id_to_pos = {}
        if "player_id" in self.players.columns and "pos" in self.players.columns:
            id_to_pos = dict(
                zip(
                    self.players["player_id"].astype(str),
                    self.players["pos"].astype(str).str.upper(),
                )
            )
        cached_positions = Counter(
            id_to_pos[pid]
            for pid in cached_ids
            if pid in id_to_pos
        )

        if cached_ids and len(cached_positions.elements()) == len(cached_ids) and cached_positions == supplied_positions:
            self._patbot_roster_ids = cached_ids
        else:
            self._patbot_roster_ids = []

        return original_recommend(
            self,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            roster_positions=roster_positions,
            top_n=top_n,
        )

    DraftEngine.recommend = recommend_with_roster_identity
    _INSTALLED = True
