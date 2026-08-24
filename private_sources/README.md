# Private local ranking sources

PatBot can read a locally saved Athletic fantasy-football projection workbook at:

`private_sources/athletic.xlsx`

The workbook itself is intentionally ignored by Git and must never be committed to the public repository. The Streamlit sidebar can save/replace this file for you when you upload a newer `.xlsx`.

PatBot reads the workbook's offensive overall ranking, custom fantasy points, and VORP. Defense rows are intentionally ignored because PatBot keeps its own league-specific defense treatment.
