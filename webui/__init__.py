"""Read-only web UI for browsing the Gold Evidence Reconstruction results.

Deliberately independent of the `veritas` package: it talks to the same
PostgreSQL database and the same ezMM item registry, but only ever *reads* from
them. That keeps the container small (no LLM SDKs, no scraping stack) and makes
it structurally impossible for the viewer to modify pipeline data.
"""

__version__ = "1.0.0"
