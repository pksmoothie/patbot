__version__ = "0.4.9"

from .opponent_history import install_owner_history_patch
from .roster_strategy import install_roster_strategy_patch
from .live_roster_context import install_live_roster_context_patch

install_owner_history_patch()
install_roster_strategy_patch()
install_live_roster_context_patch()
