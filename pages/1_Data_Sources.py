from __future__ import annotations

from pathlib import Path

import streamlit as st

from patbot.config import load_config
from patbot.refresh_pipeline import run_full_refresh


LIVE_CSV = Path("data/players_2026_live.csv")
LIVE_META = Path("data/players_2026_live.meta.json")

st.set_page_config(page_title="PatBot Data Sources", layout="wide")
st.title("PatBot — Data Sources")
st.caption("Update local private inputs, then rebuild the production player snapshot before draft night.")

cfg = load_config()
source_cfg = cfg.get("v03_consensus", {}) or {}
athletic_path = Path(source_cfg.get("athletic_path", "private_sources/athletic.xlsx"))

st.subheader("The Athletic custom rankings")
st.write(
    "The Athletic workbook is a local private input. It is saved under private_sources, which is ignored by Git, "
    "and is merged into PatBot only when the full production refresh runs."
)

if athletic_path.exists():
    stat = athletic_path.stat()
    st.success(f"Current workbook found: {athletic_path.name} • {stat.st_size / 1024:.1f} KB")
else:
    st.warning(f"No Athletic workbook is currently present at {athletic_path}.")

athletic_upload = st.file_uploader(
    "Upload updated Athletic custom rankings (.xlsx)",
    type=["xlsx"],
    help="Upload the newest workbook whenever The Athletic updates its custom rankings/projections.",
)

if athletic_upload is not None:
    athletic_path.parent.mkdir(parents=True, exist_ok=True)
    athletic_path.write_bytes(athletic_upload.getvalue())
    st.success(f"Saved locally as {athletic_path}.")
    st.info("The new workbook is saved. Run the full production refresh below to merge it into PatBot.")

st.divider()
st.subheader("Apply updated sources")
st.caption(
    "This is the same full production refresh used by the Draft Day page. It rebuilds the live player snapshot, "
    "including the latest Athletic workbook and the other configured production sources."
)

if st.button("Run full production refresh", type="primary", use_container_width=True):
    with st.spinner("Running the full production refresh chain..."):
        try:
            _, _, meta = run_full_refresh(cfg, LIVE_CSV, LIVE_META)
            st.cache_data.clear()
            st.success(f"Full refresh complete: {meta.get('draftable_rows', '?')} players.")
            st.info("Return to the Draft Day page; the refreshed snapshot will be loaded there.")
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

st.divider()
st.caption(
    "The uploaded workbook remains local to this computer and is not committed to the public PatBot repository."
)
