"""Test package for LAiSER.

Marks ``tests/`` as a regular package so ``from tests.test_helpers import ...``
resolves to this directory. Without it, ``tests`` is only a namespace package,
and any dependency that ships its own top-level ``tests`` package into
site-packages (ultralytics, pyjks and yarg all do) shadows it and breaks
collection.
"""
