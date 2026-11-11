"""Inference engines for the app.

Backends live in their own subpackages; ``engine.torch`` is the PyTorch one.
Importing this package stays cheap: backends are only loaded on demand.
"""
