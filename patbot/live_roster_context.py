from __future__ import annotations


_INSTALLED = False


def install_live_roster_context_patch() -> None:
    """Feed live PatBot player IDs into identity-aware draft strategy layers."""
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
        self._patbot_roster_ids = last_roster_ids()
        return original_recommend(
            self,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            roster_positions=roster_positions,
            top_n=top_n,
        )

    DraftEngine.recommend = recommend_with_roster_identity
    _INSTALLED = True
