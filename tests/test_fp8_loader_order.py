"""Exercise actual loader bookkeeping without requiring CUDA/Transformers.

Tensor stand-ins check dimensions only; numerical equivalence is separately
covered by test_fish_omni_backbone.py.
"""

import ast
import re
import types
import unittest
from pathlib import Path
from typing import Iterable

PREFIX = "model.layers.0.mlp.experts"


class Tensor:
    dtype = "fp8"

    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, key):
        if isinstance(key, int):
            return Tensor(self.shape[1:])
        _, sl = key
        return Tensor(((sl.stop or self.shape[1]) - (sl.start or 0), self.shape[2]))

    def copy_(self, other):
        assert self.shape == other.shape


class Model:
    def named_parameters(self):
        return [
            (PREFIX + ".w13_weight", Tensor((2, 4, 2))),
            (PREFIX + ".w2_weight", Tensor((2, 2, 2))),
        ]

    def named_buffers(self):
        return []

    def get_submodule(self, name):
        return types.SimpleNamespace(fp8=True, num_experts=2)


class LoaderOrderTest(unittest.TestCase):
    def load(self, order):
        path = (
            Path(__file__).resolve().parents[1]
            / "fish_speech/models/text2semantic/fish_omni/backbone.py"
        )
        tree = ast.parse(path.read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id.endswith("PATTERN") for t in n.targets
            )
        ]
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "load_backbone_weights"
        )
        fn.decorator_list = []
        nodes.append(fn)
        ns = dict(
            re=re,
            FishOmniBackbone=Model,
            Iterable=Iterable,
            Tensor=Tensor,
            FP8_DTYPE="fp8",
            backbone_name=lambda x: x,
            logger=types.SimpleNamespace(info=lambda x: None),
            torch=types.SimpleNamespace(
                float16="fp16",
                stack=lambda ts, dim: Tensor((len(ts), *ts[0].shape)),
                cat=lambda ts, dim: Tensor((2, 4, 2)),
            ),
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        ns["load_backbone_weights"](
            Model(),
            [(PREFIX + f".{e}.{p}_proj.weight", Tensor((2, 2))) for e, p in order],
        )

    def test_interleaved_down_survives_gate_up_flush(self):
        self.load(
            [(0, "down"), (0, "gate"), (1, "gate"), (0, "up"), (1, "up"), (1, "down")]
        )

    def test_shuffled_order(self):
        import random

        for seed in range(20):
            order = [(e, p) for e in range(2) for p in ("gate", "up", "down")]
            random.Random(seed).shuffle(order)
            self.load(order)

    def test_duplicate_after_completed_batch_is_rejected(self):
        order = [(e, p) for e in range(2) for p in ("gate", "up", "down")]
        with self.assertRaisesRegex(ValueError, "filled twice"):
            self.load(order + [(0, "gate")])

    def test_missing_projection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not loaded"):
            self.load([(e, p) for e in range(2) for p in ("gate", "up")])
