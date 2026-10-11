#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 HYBRID PROGRAM-SYNTHESIS ENGINE  ·  v2
 MDP × MCTS × Genetic Algorithms × Compiler Design

 WHAT IS NEW IN v2  (vs v1)
   1. Optimizing canonicalizer (constant folding + algebraic simplification):
      every cache (eval, traces, GA dedup, MCTS elites) is keyed by canonical
      form, so semantically equivalent programs share statistics. Classic
      compiler local optimization; the generalization is equality saturation
      (Tate et al. 2009; Willsey et al. 2021, "egg").
   2. Execution traces: each concrete program gets a behavioral fingerprint
      (output vector over train inputs). Enables behavioral deduplication and
      guided generation (Ellis et al. 2019, execution-guided synthesis).
   3. Semantic mutation: backpropagate required outputs through arithmetic
      parents, locate the worst-fitting subtree, regenerate candidates that
      minimize trace distance (Krawiec & Pawlak 2013, semantic GP).
   4. Lexicase parent selection using per-test pass vectors (Spector 2012).
   5. MAP-Elites quality-diversity archive over (size, operator-variety)
      niches; feeds GA seeds and behavior cloning (Mouret & Clune 2015).
   6. MCTS tree reuse across iterations + Dirichlet root noise (Silver 2018).
   7. Gumbel-style root search: candidates sampled via Gumbel-max, simulations
      allocated by sequential halving (Danihelka et al., ICLR 2022).
   8. Visit-confidence-weighted policy/value distillation.
   9. Train/holdout split: reward on train inputs, generalization on an
      extrapolation grid (anti-test-memorization).
  10. Policy-only beam decoder as an inference-mode probe of the learned head.
  11. Warm-start transfer across tasks; per-task seeded RNG (reproducible);
      time budget; early exit; fold-minted constants re-enter the grammar.

 Research grounding (v1) is retained: MDPs (Sutton & Barto 2018), UCT (Kocsis
 & Szepesvari 2006), PUCT/AlphaZero (Silver et al. 2018), Expert Iteration
 (Anthony et al. 2017), synthesis-as-search (Bunel 2018; Alur 2013, SyGuS),
 strongly-typed GP (Montana 1995), grammar-guided GP (Whigham 1995), syntax-
 directed translation (Aho, Lam, Sethi & Ullman).
================================================================================
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


# =============================================================================
# 1. GRAMMAR & AST  (compiler front-end: typed CFG + derivation trees)
# =============================================================================

class T(Enum):
    EXPR = "expr"
    BOOL = "bool"


class Production:
    __slots__ = ("name", "lhs", "child_types", "template", "weight", "terminal")

    def __init__(self, name: str, lhs: T, child_types, template: str, weight: float = 1.0):
        self.name = name
        self.lhs = lhs
        self.child_types = tuple(child_types)
        self.template = template
        self.weight = float(weight)
        self.terminal = len(self.child_types) == 0

    def __repr__(self):
        return f"<{self.name}>"


class Grammar:
    """Typed CFG. `hp_map` holds normalized heuristic priors per non-terminal;
    `get_const` lets the optimizer mint new literal productions at runtime,
    which immediately become legal MDP actions (bottom-up synthesis touch)."""
    def __init__(self):
        self.rules: Dict[T, List[Production]] = {t: [] for t in T}
        self.by_name: Dict[str, Production] = {}
        self.hp_map: Dict[T, Dict[str, float]] = {}

    def add(self, p: Production) -> None:
        self.rules[p.lhs].append(p)
        self.by_name[p.name] = p

    def finalize(self) -> None:
        for t in T:
            tot = sum(p.weight for p in self.rules[t]) or 1.0
            self.hp_map[t] = {p.name: p.weight / tot for p in self.rules[t]}

    def get_const(self, v: int) -> Production:
        name = f"const_{v}"
        p = self.by_name.get(name)
        if p is None:
            p = Production(name, T.EXPR, (), str(v), 0.6)
            self.rules[T.EXPR].append(p)
            self.by_name[name] = p
            self.finalize()
        return p

    def productions(self, t: T) -> List[Production]:
        return self.rules[t]

    def emit(self, node: "Node") -> str:
        """Code-generation phase: AST -> Python source (an unparser)."""
        if node.prod.terminal:
            return node.prod.template
        parts = [self.emit(c) for c in node.children]
        return node.prod.template.format(*parts)


class Node:
    """Immutable-style AST node; updates are persistent path copies."""
    __slots__ = ("prod", "children", "hole_type", "_key")

    def __init__(self, prod: Optional[Production] = None, children=(), hole_type: Optional[T] = None):
        self.prod = prod
        self.children: List[Node] = list(children)
        self.hole_type = hole_type
        self._key: Optional[str] = None

    @classmethod
    def hole(cls, t: T) -> "Node":
        return cls(hole_type=t)

    @property
    def is_hole(self) -> bool:
        return self.prod is None

    def copy(self) -> "Node":
        return Node(self.prod, [c.copy() for c in self.children], self.hole_type)

    def serialize(self) -> str:
        if self._key is None:
            if self.is_hole:
                self._key = f"(?{self.hole_type.value})"
            elif not self.children:
                self._key = f"({self.prod.name})"
            else:
                self._key = "(" + self.prod.name + " " + \
                            " ".join(c.serialize() for c in self.children) + ")"
        return self._key

    def size(self) -> int:
        if self.is_hole:
            return 0
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if self.is_hole or not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def leftmost_hole_path(self):
        if self.is_hole:
            return ()
        for i, c in enumerate(self.children):
            sub = c.leftmost_hole_path()
            if sub is not None:
                return (i,) + sub
        return None

    def has_holes(self) -> bool:
        return self.leftmost_hole_path() is not None

    def subtree_at(self, path):
        n = self
        for i in path:
            n = n.children[i]
        return n

    def replace_at(self, path, repl: "Node") -> "Node":
        if not path:
            return repl
        children = list(self.children)
        children[path[0]] = self.children[path[0]].replace_at(path[1:], repl)
        return Node(self.prod, children, self.hole_type)

    def typed_paths(self, out=None, path=()):
        out = [] if out is None else out
        if self.is_hole:
            return out
        out.append((path, self.prod.lhs))
        for i, c in enumerate(self.children):
            c.typed_paths(out, path + (i,))
        return out

    def preorder(self):
        if self.is_hole:
            return
        yield self.prod
        for c in self.children:
            yield from c.preorder()


CONST_MAX = 5


def build_grammar() -> Grammar:
    g = Grammar()
    E, B = T.EXPR, T.BOOL
    g.add(Production("add",  E, (E, E), "({0} + {1})", 6.0))
    g.add(Production("sub",  E, (E, E), "({0} - {1})", 6.0))
    g.add(Production("mul",  E, (E, E), "({0} * {1})", 4.0))
    g.add(Production("fdiv", E, (E, E), "({0} // {1})", 1.2))
    g.add(Production("mod",  E, (E, E), "({0} % {1})", 1.2))
    g.add(Production("neg",  E, (E,), "(-{0})", 0.8))
    g.add(Production("ite",  E, (B, E, E), "({1} if {0} else {2})", 1.5))
    g.add(Production("var_x", E, (), "x", 5.0))
    g.add(Production("var_y", E, (), "y", 5.0))
    for c in range(CONST_MAX + 1):
        g.add(Production(f"const_{c}", E, (), str(c), 2.5))
    for name, op in (("lt", "<"), ("le", "<="), ("gt", ">"),
                     ("ge", ">="), ("eq", "=="), ("ne", "!=")):
        g.add(Production(name, B, (E, E), f"({{0}} {op} {{1}})", 2.0))
    g.add(Production("and_", B, (B, B), "({0} and {1})", 1.0))
    g.add(Production("or_",  B, (B, B), "({0} or {1})", 1.0))
    g.add(Production("not_", B, (B,), "(not {0})", 0.6))
    g.add(Production("true",  B, (), "True", 0.8))
    g.add(Production("false", B, (), "False", 0.8))
    g.finalize()
    return g


# =============================================================================
# 2. OPTIMIZING CANONICALIZER  (NEW in v2 — compiler middle-end)
# =============================================================================

class Optimizer:
    """Bottom-up local optimization: constant folding + algebraic identity
    elimination, applied to a fixpoint. Because every rule strictly shrinks
    the tree, termination is guaranteed. Canonical signatures give all
    downstream components *semantic* deduplication: `x+0`, `0+x` and `x`
    become one cache entry. In a production system this layer generalizes to
    e-graphs / equality saturation (Tate et al. 2009; Willsey et al. 2021).
    """
    FOLD_LIMIT = 64

    def __init__(self, g: Grammar):
        self.g = g
        self._memo: Dict[str, Node] = {}

    # ---- helpers -------------------------------------------------------------
    def _is_const(self, n: Node) -> bool:
        return (not n.is_hole) and n.prod.name.startswith("const_")

    def _cval(self, n: Node) -> int:
        return int(n.prod.name.split("_", 1)[1])

    def _is_bool_lit(self, n: Node) -> bool:
        return (not n.is_hole) and n.prod.template in ("True", "False")

    def _same(self, a: Node, b: Node) -> bool:
        return a.serialize() == b.serialize()

    def _const(self, v: int) -> Optional[Node]:
        if -self.FOLD_LIMIT <= v <= self.FOLD_LIMIT:
            return Node(self.g.get_const(v))
        return None

    def _lit(self, which: str) -> Node:
        return Node(self.g.by_name[which])

    # ---- one round of rewrite rules at the root ------------------------------
    def _rule(self, n: Node) -> Node:
        p, k = n.prod, n.children
        name = p.name
        if name == "add":
            if self._is_const(k[0]) and self._is_const(k[1]):
                c = self._const(self._cval(k[0]) + self._cval(k[1]))
                if c: return c
            if self._is_const(k[1]) and self._cval(k[1]) == 0: return k[0]
            if self._is_const(k[0]) and self._cval(k[0]) == 0: return k[1]
        elif name == "sub":
            if self._same(k[0], k[1]): return self._const(0)
            if self._is_const(k[0]) and self._is_const(k[1]):
                c = self._const(self._cval(k[0]) - self._cval(k[1]))
                if c: return c
            if self._is_const(k[1]) and self._cval(k[1]) == 0: return k[0]
        elif name == "mul":
            if self._is_const(k[0]) and self._is_const(k[1]):
                c = self._const(self._cval(k[0]) * self._cval(k[1]))
                if c: return c
            for a, b in ((k[0], k[1]), (k[1], k[0])):
                if self._is_const(b):
                    if self._cval(b) == 0: return self._const(0)
                    if self._cval(b) == 1: return a
        elif name == "fdiv":
            if self._is_const(k[1]) and self._cval(k[1]) != 0:
                if self._is_const(k[0]):
                    c = self._const(self._cval(k[0]) // self._cval(k[1]))
                    if c: return c
                if self._cval(k[1]) == 1: return k[0]
        elif name == "mod":
            if self._is_const(k[1]) and self._cval(k[1]) != 0:
                if self._is_const(k[0]):
                    c = self._const(self._cval(k[0]) % self._cval(k[1]))
                    if c: return c
                if self._cval(k[1]) == 1: return self._const(0)
        elif name == "neg":
            if not k[0].is_hole and k[0].prod.name == "neg":
                return k[0].children[0]
            if self._is_const(k[0]):
                c = self._const(-self._cval(k[0]))
                if c: return c
        elif name == "ite":
            if self._is_bool_lit(k[0]):
                return k[1] if k[0].prod.template == "True" else k[2]
            if self._same(k[1], k[2]): return k[1]
        elif name in ("lt", "le", "gt", "ge", "eq", "ne"):
            if self._is_const(k[0]) and self._is_const(k[1]):
                a, b = self._cval(k[0]), self._cval(k[1])
                hit = {"lt": a < b, "le": a <= b, "gt": a > b,
                       "ge": a >= b, "eq": a == b, "ne": a != b}[name]
                return self._lit("true" if hit else "false")
            if self._same(k[0], k[1]):
                return self._lit("true" if name in ("eq", "le", "ge") else "false")
        elif name == "and_":
            if self._is_bool_lit(k[0]):
                return k[1] if k[0].prod.template == "True" else self._lit("false")
            if self._is_bool_lit(k[1]):
                return k[0] if k[1].prod.template == "True" else self._lit("false")
            if self._same(k[0], k[1]): return k[0]
        elif name == "or_":
            if self._is_bool_lit(k[0]):
                return self._lit("true") if k[0].prod.template == "True" else k[1]
            if self._is_bool_lit(k[1]):
                return self._lit("true") if k[1].prod.template == "True" else k[0]
            if self._same(k[0], k[1]): return k[0]
        elif name == "not_":
            if self._is_bool_lit(k[0]):
                return self._lit("false" if k[0].prod.template == "True" else "true")
            if not k[0].is_hole and k[0].prod.name == "not_":
                return k[0].children[0]
        return n

    # ---- public API ------------------------------------------------------------
    def canonical(self, tree: Node) -> Node:
        if tree.is_hole:
            return tree
        key = tree.serialize()
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        node = Node(tree.prod, [self.canonical(c) for c in tree.children])
        while True:
            new = self._rule(node)
            if new is node:
                break
            node = new
        self._memo[key] = node
        return node

    def sig(self, tree: Node) -> str:
        return self.canonical(tree).serialize()


# =============================================================================
# 3. TASKS & REWARD  (shared oracle; train/holdout split — NEW)
# =============================================================================

@dataclass
class SynthesisTask:
    name: str
    description: str
    examples: List[Tuple[Tuple[int, int], int]]          # train inputs -> reward
    examples_holdout: List[Tuple[Tuple[int, int], int]] = field(default_factory=list)


@dataclass
class Reward:
    value: float
    pass_frac: float
    error: Optional[str] = None
    pass_vec: Tuple[bool, ...] = ()       # per-test vector (lexicase uses this)


def _make_tasks() -> Dict[str, SynthesisTask]:
    train_grid = (-2, -1, 0, 1, 2, 3)
    hold_grid = (-5, -4, 4, 5)             # strictly extrapolation inputs
    ex = lambda f, grid: [((x, y), f(x, y)) for x in grid for y in grid]

    def mk(name, desc, f):
        return SynthesisTask(name, desc, ex(f, train_grid), ex(f, hold_grid))

    return {
        "max2": mk("max2", "synthesize max(x, y): branching (ite + comparison)",
                   lambda x, y: x if x > y else y),
        "square_sum": mk("square_sum", "synthesize x*x + y: multiplication",
                         lambda x, y: x * x + y),
        "absdiff": mk("absdiff", "synthesize |x - y|: branching + two branches",
                      lambda x, y: abs(x - y)),
        "parity": mk("parity", "synthesize (x + y) mod 2: modulo operator",
                     lambda x, y: (x + y) % 2),
    }


TASKS = _make_tasks()


# =============================================================================
# 4. COMPILER BACK-END  (semantic analysis -> codegen -> bytecode -> reward)
# =============================================================================

class SemanticError(Exception):
    pass


class TypeChecker:
    """Semantic-analysis phase: re-derives types bottom-up, checks scoping."""
    def __init__(self):
        self.scope = {"x": T.EXPR, "y": T.EXPR}

    def check(self, tree: Node):
        errors: List[str] = []
        ty = self._infer(tree, errors)
        return (not errors), ty, errors

    def _infer(self, n: Node, errors: List[str]) -> Optional[T]:
        if n.is_hole:
            errors.append("incomplete program: unexpanded hole")
            return None
        p = n.prod
        if p.terminal:
            if p.name.startswith("const_"):
                return T.EXPR
            if p.template in self.scope:
                return self.scope[p.template]
            if p.template in ("True", "False"):
                return T.BOOL
            errors.append(f"unbound symbol '{p.template}'")
            return None
        ts = [self._infer(c, errors) for c in n.children]
        for want, got in zip(p.child_types, ts):
            if got is not want:
                errors.append(f"'{p.name}': operand type mismatch")
        return p.lhs


class Evaluator:
    """AST -> semantic check -> source -> bytecode -> sandboxed exec -> tests.
    v2: functions, traces and rewards are all cached by *canonical* signature,
    and per-test pass vectors are exposed for lexicase selection."""
    def __init__(self, grammar: Grammar, task: SynthesisTask, cfg: "Config", opt: Optimizer):
        self.g = grammar
        self.task = task
        self.cfg = cfg
        self.opt = opt
        self.cache: Dict[str, Reward] = {}
        self._fn: Dict[str, Callable] = {}
        self._trace: Dict[str, tuple] = {}
        self.stats = defaultdict(int)
        self._checker = TypeChecker()

    def source(self, tree: Node) -> str:
        return f"def f(x, y):\n    return {self.g.emit(tree)}"

    def compile_tree(self, tree: Node):
        ok, _ty, errors = self._checker.check(tree)
        if not ok:
            raise SemanticError("; ".join(errors[:3]))
        code = compile(self.source(tree), "<synthesized>", "exec")
        ns: Dict[str, object] = {"__builtins__": {}}          # sandbox
        exec(code, ns)
        return ns["f"]

    def _get_fn(self, sig: str, tree: Node) -> Callable:
        fn = self._fn.get(sig)
        if fn is None:
            fn = self.compile_tree(self.opt.canonical(tree))   # compile simplified form
            self._fn[sig] = fn
            self.stats["compiles"] += 1
        return fn

    # ---- behavioral fingerprint (NEW) ----------------------------------------
    def trace(self, tree: Node) -> Optional[tuple]:
        """Output vector over train inputs; None where execution faults.
        Two programs with equal traces are behaviorally identical on the tests."""
        if tree.has_holes():
            return None
        sig = self.opt.sig(tree)
        t = self._trace.get(sig)
        if t is None:
            fn = self._get_fn(sig, tree)
            out = []
            for args, _expected in self.task.examples:
                try:
                    out.append(fn(*args))
                except Exception:
                    out.append(None)
            t = tuple(out)
            self._trace[sig] = t
            self.stats["unique_traces"] += 1
        return t

    # ---- reward ---------------------------------------------------------------
    def _score(self, fn, examples, size: int) -> Reward:
        passed, vec, err = 0, [], None
        for args, expected in examples:
            try:
                got = fn(*args)
            except Exception as e:
                got, err = None, (type(e).__name__ if err is None else err)
            ok = got == expected
            vec.append(ok)
            passed += ok
        val = passed / len(examples) - self.cfg.size_penalty * size
        if passed == len(examples):
            val += self.cfg.perfect_bonus
        return Reward(max(val, self.cfg.min_reward), passed / len(examples), err, tuple(vec))

    def evaluate(self, tree: Node) -> Reward:
        canon = self.opt.canonical(tree)
        sig = canon.serialize()
        hit = self.cache.get(sig)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit
        self.stats["evals"] += 1
        try:
            fn = self._get_fn(sig, tree)
        except Exception as e:
            r = Reward(self.cfg.min_reward, 0.0, f"compile: {type(e).__name__}: {e}")
            self.cache[sig] = r
            return r
        r = self._score(fn, self.task.examples, canon.size())
        self.cache[sig] = r
        return r

    def evaluate_holdout(self, tree: Node) -> Reward:
        fn = self._get_fn(self.opt.sig(tree), tree)
        return self._score(fn, self.task.examples_holdout or self.task.examples,
                           self.opt.canonical(tree).size())


# =============================================================================
# 5. THE MDP  (program derivation as sequential decision making)
# =============================================================================

class State:
    __slots__ = ("tree", "size_budget", "_key", "_hole_path")

    def __init__(self, tree: Node, size_budget: int):
        self.tree = tree
        self.size_budget = size_budget
        self._key: Optional[str] = None
        self._hole_path = None

    @property
    def key(self) -> str:
        if self._key is None:
            self._key = f"{self.tree.serialize()}|s{self.size_budget}"
        return self._key

    @property
    def hole_path(self):
        if self._hole_path is None:
            self._hole_path = self.tree.leftmost_hole_path()
        return self._hole_path

    def is_terminal(self) -> bool:
        return self.hole_path is None


class CodeGenEnv:
    """Program-derivation MDP: states = partial ASTs, actions = grammar
    productions legal at the leftmost hole (grammar-constrained decoding)."""
    def __init__(self, grammar: Grammar, evaluator: Evaluator, cfg: "Config"):
        self.g = grammar
        self.ev = evaluator
        self.cfg = cfg

    def reset(self) -> State:
        return State(Node.hole(T.EXPR), self.cfg.max_size)

    def actions(self, s: State) -> List[Production]:
        if s.is_terminal() or s.size_budget < 1:
            return []
        hp = s.hole_path
        hole = s.tree.subtree_at(hp)
        depth = len(hp) + 1
        return [p for p in self.g.productions(hole.hole_type)
                if p.terminal or depth + 1 <= self.cfg.max_depth]

    def step(self, s: State, prod: Production):
        node = Node(prod, [Node.hole(ct) for ct in prod.child_types])
        tree = s.tree.replace_at(s.hole_path, node)
        ns = State(tree, s.size_budget - 1)
        return ns, self.cfg.step_penalty, ns.is_terminal()

    def terminal_value(self, s: State) -> Reward:
        return self.ev.evaluate(s.tree)


# =============================================================================
# 6. LEARNED POLICY / VALUE HEADS  (confidence-weighted training — NEW)
# =============================================================================

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


class PolicyNet:
    def __init__(self, lr: float = 0.08, l2: float = 1e-4):
        self.lr, self.l2 = lr, l2
        self.w: Dict[str, float] = {}

    def _logit(self, parent, ht, a):
        w = self.w
        return (w.get(f"a:{a.name}", 0.0)
                + w.get(f"pa:{parent}>{a.name}", 0.0)
                + w.get(f"ta:{ht.value}>{a.name}", 0.0))

    def probs(self, parent, ht, actions):
        return softmax([self._logit(parent, ht, a) for a in actions])

    def train_one(self, parent, ht, actions, target, weight: float = 1.0) -> float:
        p = self.probs(parent, ht, actions)
        ce = -weight * sum(q * math.log(max(pi, 1e-12)) for pi, q in zip(p, target))
        for a, pi, q in zip(actions, p, target):
            d = (pi - q) * weight
            if -1e-9 < d < 1e-9:
                continue
            for k in (f"a:{a.name}", f"pa:{parent}>{a.name}", f"ta:{ht.value}>{a.name}"):
                wv = self.w.get(k, 0.0)
                self.w[k] = wv - self.lr * d - self.l2 * wv
        return ce

    def train_batch(self, batch) -> float:
        """Items: (parent, hole_type, actions, target, weight). Weights are
        normalized to mean 1 so big-visit nodes dominate without exploding LR."""
        if not batch:
            return 0.0
        tw = sum(b[-1] for b in batch) or 1.0
        ces = [self.train_one(p, h, a, d, w * len(batch) / tw)
               for (p, h, a, d, w) in batch]
        return sum(ces) / len(ces)


class ValueNet:
    def __init__(self, lr: float = 0.06, l2: float = 1e-5):
        self.lr, self.l2 = lr, l2
        self.w: Dict[str, float] = {}

    def predict(self, feats) -> float:
        return sum(self.w.get(k, 0.0) * v for k, v in feats.items())

    def train_one(self, feats, target, weight: float = 1.0) -> float:
        err = target - self.predict(feats)
        for k, v in feats.items():
            wv = self.w.get(k, 0.0)
            self.w[k] = wv + self.lr * weight * err * v - self.l2 * wv
        return abs(err)

    def train_batch(self, batch) -> float:
        if not batch:
            return 0.0
        tw = sum(b[-1] for b in batch) or 1.0
        errs = [self.train_one(f, t, w * len(batch) / tw) for (f, t, w) in batch]
        return sum(errs) / len(errs)


# =============================================================================
# 7. MONTE CARLO TREE SEARCH  (tree reuse + Dirichlet + Gumbel halving — NEW)
# =============================================================================

class MCTSNode:
    __slots__ = ("state", "parent", "action", "r_in", "children",
                 "untried", "N", "W", "prior", "terminal", "value")

    def __init__(self, state: State, parent=None, action: Optional[Production] = None, r_in: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.r_in = r_in
        self.children: List["MCTSNode"] = []
        self.untried: List[Production] = []
        self.N = 0
        self.W = 0.0
        self.prior: Dict[str, float] = {}
        self.terminal = False
        self.value = 0.0


@dataclass
class SearchResult:
    best_tree: Optional[Node]
    best_reward: Optional[Reward]
    best_value: float
    terminals: int
    samples: list
    top: list


class MCTSAgent:
    """PUCT search over the derivation MDP, v2 upgrades:
      * persistent root across iterations (statistics compound; priors are
        refreshed from the current policy — AlphaZero tree reuse);
      * Dirichlet noise mixed into root priors each search;
      * optional Gumbel root phase: candidates sampled by Gumbel-max over
        prior logits, simulation budget allocated by sequential halving
        (Danihelka et al. 2022, root-level);
      * early exit once a perfect program is found;
      * visit-mass confidence attached to distillation samples.
    """
    def __init__(self, env: CodeGenEnv, policy: PolicyNet, value: ValueNet,
                 cfg: "Config", rng: random.Random):
        self.env = env
        self.policy = policy
        self.value = value
        self.cfg = cfg
        self.rng = rng
        self._root: Optional[MCTSNode] = None
        self._root_key: Optional[str] = None
        self._cur_mix, self._v_weight = 0.5, 0.0
        self._gumbel_used = 0

    # ---- context / features ---------------------------------------------------
    def context(self, s: State):
        hp = s.hole_path
        if hp is None:
            return "DONE", s.tree.prod.lhs
        hole = s.tree.subtree_at(hp)
        parent = s.tree.subtree_at(hp[:-1]).prod.name if hp else "ROOT"
        return parent, hole.hole_type

    def value_feats(self, s: State) -> Dict[str, float]:
        parent, ht = self.context(s)
        d = len(s.hole_path) if s.hole_path is not None else 0
        return {"b": 1.0, f"t:{ht.value}": 1.0, f"p:{parent}": 1.0,
                f"dp:{min(d, 6)}": 1.0, f"sz:{min(s.size_budget // 8, 8)}": 1.0}

    # ---- priors ----------------------------------------------------------------
    def _set_prior(self, node: MCTSNode, is_root: bool = False) -> None:
        s = node.state
        acts = self.env.actions(s)
        if not acts:
            node.terminal, node.value = True, self.cfg.failure_value
            node.untried, node.prior = [], {}
            return
        parent, ht = self.context(s)
        p_learn = self.policy.probs(parent, ht, acts)
        hp = self.g.hp_map[ht]
        tot = sum(hp.get(a.name, 0.0) for a in acts) or 1.0
        mix = self._cur_mix
        node.prior = {a.name: mix * pl + (1 - mix) * hp.get(a.name, 0.0) / tot
                      for a, pl in zip(acts, p_learn)}
        if is_root and self.cfg.dirichlet_eps > 0:      # AlphaZero root noise
            names = list(node.prior)
            noise = [self.rng.gammavariate(self.cfg.dirichlet_alpha, 1.0) + 1e-9
                     for _ in names]
            nsum = sum(noise)
            eps = self.cfg.dirichlet_eps
            for nm, nv in zip(names, noise):
                node.prior[nm] = (1 - eps) * node.prior[nm] + eps * nv / nsum
        child_names = {c.action.name for c in node.children}
        node.untried = [a for a in acts if a.name not in child_names]

    def _init_node(self, node: MCTSNode, is_root: bool = False) -> None:
        s = node.state
        if s.is_terminal():
            rew = self.env.terminal_value(s)
            node.terminal, node.value = True, rew.value
            self._note_terminal(s.tree, rew)
            return
        self._set_prior(node, is_root)

    def _refresh(self, node: MCTSNode, is_root: bool = False) -> None:
        """Recompute priors/untried on a reused tree without wiping statistics."""
        if node.terminal:
            return
        self._set_prior(node, is_root)
        for c in node.children:
            self._refresh(c)

    def _action_prior(self, state: State, mix: float, acts=None):
        if acts is None:
            acts = self.env.actions(state)
        if not acts:
            return acts, None, None, None
        parent, ht = self.context(state)
        p_learn = self.policy.probs(parent, ht, acts)
        hp = self.g.hp_map[ht]
        tot = sum(hp.get(a.name, 0.0) for a in acts) or 1.0
        blended = [mix * pl + (1 - mix) * hp.get(a.name, 0.0) / tot
                   for a, pl in zip(acts, p_learn)]
        return acts, blended, parent, ht

    def _note_terminal(self, tree: Node, rew: Reward) -> None:
        self._terminals += 1
        if rew.value > self._best_value:
            self._best_value, self._best_tree, self._best_reward = rew.value, tree, rew
        sig = self.ev_sig(tree)
        for v, t in self._top:
            if self.ev_sig(t) == sig:
                return
        self._top.append((rew.value, tree))
        self._top.sort(key=lambda x: -x[0])
        del self._top[5:]

    def ev_sig(self, tree: Node) -> str:
        return self.env.ev.opt.sig(tree)

    # ---- tree policy -------------------------------------------------------------
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        sqrt_n = math.sqrt(node.N)
        best, best_score = None, -1e18
        for c in node.children:
            q = (c.W / c.N) if c.N else 0.0
            u = self.cfg.c_puct * c.prior.get(c.action.name, 0.0) * sqrt_n / (1.0 + c.N)
            s = q + u
            if s > best_score:
                best, best_score = c, s
        return best

    def _sample_untried(self, node: MCTSNode) -> Production:
        ws = [node.prior.get(a.name, 1e-6) for a in node.untried]
        return self.rng.choices(node.untried, weights=ws, k=1)[0]

    # ---- rollouts / evaluation ------------------------------------------------------
    def _rollout(self, state: State) -> float:
        G, s = 0.0, state
        for _ in range(state.size_budget + 1):
            if s.is_terminal():
                rew = self.env.terminal_value(s)
                self._note_terminal(s.tree, rew)
                return G + rew.value
            acts, probs, _p, _h = self._action_prior(s, self._cur_mix)
            if not acts:
                return G + self.cfg.failure_value
            a = self.rng.choices(acts, weights=probs, k=1)[0]
            s, r, _done = self.env.step(s, a)
            G += r
        return G + self.cfg.failure_value

    def _leaf_value(self, state: State) -> float:
        return max(-0.5, min(1.25, self.value.predict(self.value_feats(state))))

    # ---- Gumbel root phase (Danihelka et al. 2022, root level) ----------------------
    def _gumbel(self) -> float:
        u = max(self.rng.random(), 1e-12)
        return -math.log(max(-math.log(u), 1e-12))

    def _child_for(self, node: MCTSNode, a: Production) -> Optional[MCTSNode]:
        for c in node.children:
            if c.action.name == a.name:
                return c
        if a in node.untried:
            ns, r, _d = self.env.step(node.state, a)
            child = MCTSNode(ns, node, a, r)
            node.untried.remove(a)
            node.children.append(child)
            self._init_node(child)
            return child
        return None

    def _probe(self, node: MCTSNode, a: Production, n: int) -> float:
        child = self._child_for(node, a)
        if child is None:
            return 0.0
        for _ in range(n):
            if child.terminal:
                z = child.value
            elif self.rng.random() < self._v_weight:
                z = self._leaf_value(child.state)
            else:
                z = self._rollout(child.state)
            g = z
            for nd in (child, node):       # backup path root -> child
                nd.N += 1
                nd.W += g
                g += nd.r_in
            if self._best_value >= self.cfg.target_fitness:
                break
        return (child.W / child.N) if child.N else 0.0

    def _gumbel_phase(self, root: MCTSNode, budget: int) -> int:
        cands = [c.action for c in root.children] + list(root.untried)
        if len(cands) < 2 or budget < 4:
            return 0
        scored = sorted(cands,
                        key=lambda c: -(math.log(max(root.prior.get(c.name, 1e-6), 1e-9))
                                        + self._gumbel()))
        cands = scored[: min(len(scored), max(4, budget // 6))]
        used, scores = 0, {}
        while used < budget and len(cands) > 1:
            per = max(1, (budget - used) // (2 * len(cands)))
            for c in cands:
                scores[c.name] = self._probe(root, c, per)
            used += per * len(cands)
            cands = sorted(cands, key=lambda c: -scores.get(c.name, -1e18))[: math.ceil(len(cands) / 2)]
            if self._best_value >= self.cfg.target_fitness:
                break
        if cands and used < budget:
            self._probe(root, cands[0], budget - used)
            used = budget
        return used

    # ---- main entry --------------------------------------------------------------------
    def search(self, root_state: State, n_sims: int, mix: float, v_weight: float,
               reuse: bool = True) -> SearchResult:
        self._cur_mix, self._v_weight, self._gumbel_used = mix, v_weight, 0
        self._best_value, self._best_tree, self._best_reward = -1e18, None, None
        self._terminals, self._top = 0, []
        key = root_state.key
        reused = reuse and self._root is not None and self._root_key == key
        if reused:
            root = self._root
            self._refresh(root, is_root=True)
        else:
            root = MCTSNode(root_state)
            self._root, self._root_key = root, key
            self._init_node(root, is_root=True)
        if root.terminal:
            return SearchResult(root_state.tree, self._best_reward, self._best_value,
                                self._terminals, [], list(self._top))
        used = 0
        if self.cfg.use_gumbel and n_sims >= 90:
            used = self._gumbel_phase(root, min(self.cfg.gumbel_budget, n_sims // 3))
            n_sims -= used
        for _ in range(n_sims):
            if self._best_value >= self.cfg.target_fitness:   # early exit
                break
            node, path = root, [root]
            while True:
                if node.terminal:
                    z = node.value
                    break
                if node.untried:                               # expansion
                    a = self._sample_untried(node)
                    ns, r, _d = self.env.step(node.state, a)
                    child = MCTSNode(ns, node, a, r)
                    node.untried.remove(a)
                    node.children.append(child)
                    self._init_node(child)
                    path.append(child)
                    if child.terminal:
                        z = child.value
                    elif self.rng.random() < self._v_weight:
                        z = self._leaf_value(child.state)
                    else:
                        z = self._rollout(child.state)
                    break
                node = self._select_child(node)                 # selection
                path.append(node)
            g = z                                               # backup
            for n in reversed(path):
                n.N += 1
                n.W += g
                g += n.r_in
        # ---- distillation samples with confidence weights ----
        samples = []
        stack = [root]
        while stack:
            n = stack.pop()
            if n.children and n.N >= self.cfg.min_node_visits:
                tot = sum(c.N for c in n.children)
                if tot > 0:
                    parent, ht = self.context(n.state)
                    samples.append((parent, ht, [c.action for c in n.children],
                                    [c.N / tot for c in n.children],
                                    n.W / n.N, self.value_feats(n.state),
                                    float(tot)))
                stack.extend(n.children)
        samples.sort(key=lambda s: -s[-1])
        samples = samples[: self.cfg.max_train_nodes]
        return SearchResult(self._best_tree, self._best_reward, self._best_value,
                            self._terminals, samples, list(self._top))


# =============================================================================
# 8. BEAM DECODER  (NEW: policy-only inference probe)
# =============================================================================

def beam_decode(env: CodeGenEnv, agent: MCTSAgent, policy: PolicyNet,
                mix: float, width: int, max_steps: int):
    """Deterministic beam search under the (blended) policy — no simulations.
    A rising beam score on solved tasks is direct evidence the distillation
    transferred search competence into the policy itself."""
    beams = [(env.reset(), 0.0)]
    done: List = []
    for _ in range(max_steps):
        if not beams:
            break
        cand, seen = [], set()
        for s, lp in beams:
            if s.is_terminal():
                done.append((s, lp))
                continue
            acts, probs, _p, _h = agent._action_prior(s, mix)
            if not acts:
                continue
            order = sorted(range(len(acts)), key=lambda i: -probs[i])[:width]
            for i in order:
                ns, _r, _d = env.step(s, acts[i])
                k = ns.key
                if k in seen:
                    continue
                seen.add(k)
                cand.append((ns, lp + math.log(max(probs[i], 1e-9))))
        cand.sort(key=lambda x: -x[1])
        beams = cand[:width]
    for s, lp in beams:
        if s.is_terminal():
            done.append((s, lp))
    if not done:
        return None
    s, lp = max(done, key=lambda x: x[1])
    return s.tree, lp


# =============================================================================
# 9. GENETIC ALGORITHM  (lexicase + semantic mutation + behavioral dedup — NEW)
# =============================================================================

class GeneticAlgorithm:
    """Strongly-typed, grammar-guided GP, v2 upgrades:
      * lexicase selection over per-test pass vectors (Spector 2012);
      * semantic mutation via output backpropagation + guided generation
        (Krawiec & Pawlak 2013);
      * behavioral dedup: candidates whose execution trace already exists are
        rejected, keeping the population semantically diverse;
      * canonical-signature dedup shared with the rest of the pipeline.
    """
    def __init__(self, grammar: Grammar, evaluator: Evaluator, cfg: "Config", rng: random.Random):
        self.g = grammar
        self.ev = evaluator
        self.cfg = cfg
        self.rng = rng

    # ---- generators -----------------------------------------------------------
    def _wc(self, items: List[Production]) -> Production:
        return self.rng.choices(items, weights=[p.weight for p in items], k=1)[0]

    def random_tree(self, t: T, depth_left: int, full: bool = False) -> Node:
        prods = self.g.productions(t)
        terms = [p for p in prods if p.terminal]
        if depth_left <= 1:
            return Node(self._wc(terms))
        if not full and self.rng.random() < 0.3:
            return Node(self._wc(terms))
        p = self._wc(prods)
        return Node(p, [self.random_tree(ct, depth_left - 1, full) for ct in p.child_types])

    def _valid(self, tree: Node) -> bool:
        return tree.depth() <= self.cfg.max_depth and tree.size() <= self.cfg.max_size

    # ---- semantic machinery (NEW) ------------------------------------------------
    @staticmethod
    def _pointwise(fn, a, b):
        if a is None or b is None:
            return None
        out = []
        for x, y in zip(a, b):
            if x is None or y is None:
                return None
            v = fn(x, y)
            if v is None:
                return None
            out.append(v)
        return tuple(out)

    @staticmethod
    def _tdist(tr, ideal) -> Optional[float]:
        if tr is None or ideal is None:
            return None
        errs = [2.0 if t is None else abs(t - i) for t, i in zip(tr, ideal)]
        return sum(errs) / len(errs)

    def _semantic_mutate(self, tree: Node) -> Optional[Node]:
        """Walk down arithmetic chains, backpropagating the target output
        vector to each operand ('what should this subtree compute?'), then
        regenerate the worst-fitting subtree with candidates ranked by trace
        distance to the backpropagated ideal."""
        target = tuple(exp for _a, exp in self.ev.task.examples)
        node, path, best = tree, (), None
        for _ in range(3):
            if node.is_hole or node.prod.terminal:
                break
            op = node.prod.name
            if op not in ("add", "sub", "mul"):
                break
            c0, c1 = node.children
            t0, t1 = self.ev.trace(c0), self.ev.trace(c1)
            if op == "add":
                i0 = self._pointwise(lambda a, b: a - b, target, t1)
                i1 = self._pointwise(lambda a, b: a - b, target, t0)
            elif op == "sub":
                i0 = self._pointwise(lambda a, b: a + b, target, t1)
                i1 = self._pointwise(lambda a, b: a - b, t0, target)
            else:  # mul — exact only when division is integral
                def inv(a, b):
                    return None if (b == 0 or a % b) else a // b
                i0 = self._pointwise(inv, target, t1)
                i1 = self._pointwise(inv, target, t0)
            e0, e1 = self._tdist(t0, i0), self._tdist(t1, i1)
            for e, sp, ideal in ((e0, path + (0,), i0), (e1, path + (1,), i1)):
                if e is not None and (best is None or e > best[0]):
                    best = (e, sp, ideal)
            if e0 is None and e1 is None:
                break
            if e1 is None or (e0 is not None and e0 >= e1):
                if e0 is None:
                    break
                node, path, target = c0, path + (0,), i0
            else:
                node, path, target = c1, path + (1,), i1
            if target is None:
                break
        if best is None or best[2] is None:
            return None
        _e, sub_path, ideal = best
        depth_allowed = max(1, self.cfg.max_depth - len(sub_path))
        cands = []
        for _ in range(self.cfg.guided_k):
            c = self.random_tree(T.EXPR, min(depth_allowed, self.rng.choice((1, 2, 3))))
            if self.ev.trace(c) is not None:
                cands.append(c)
        if not cands:
            return None
        best_sub = min(cands, key=lambda c: self._tdist(self.ev.trace(c), ideal) or 1e9)
        return tree.replace_at(sub_path, best_sub)

    # ---- variation operators -------------------------------------------------------
    def crossover(self, a: Node, b: Node) -> Optional[Node]:
        """Strongly-typed subtree swap (Montana 1995)."""
        path, ty = self.rng.choice(a.typed_paths())
        cands = [p for p, t2 in b.typed_paths() if t2 == ty]
        if not cands:
            return None
        child = a.replace_at(path, b.subtree_at(self.rng.choice(cands)).copy())
        return child if self._valid(child) else None

    def _mutate_regen(self, tree, paths):
        path, ty = self.rng.choice(paths)
        allowed = max(1, self.cfg.max_depth - len(path))
        sub = self.random_tree(ty, min(allowed, self.rng.choice((1, 2, 3))), full=False)
        return tree.replace_at(path, sub)

    def _mutate_const(self, tree, const_paths):
        path = self.rng.choice(const_paths)
        cur = int(tree.subtree_at(path).prod.name.split("_", 1)[1])
        nxt = max(0, min(CONST_MAX, cur + self.rng.choice((-2, -1, 1, 2))))
        return tree.replace_at(path, Node(self.g.by_name[f"const_{nxt}"]))

    def _mutate_var(self, tree, var_paths):
        path = self.rng.choice(var_paths)
        node = tree.subtree_at(path)
        other = "var_y" if node.prod.name == "var_x" else "var_x"
        return tree.replace_at(path, Node(self.g.by_name[other]))

    def _mutate_opswap(self, tree, paths):
        path, ty = self.rng.choice(paths)
        node = tree.subtree_at(path)
        if node.prod.terminal:
            return None
        alts = [p for p in self.g.productions(ty)
                if not p.terminal and p.child_types == node.prod.child_types
                and p.name != node.prod.name]
        if not alts:
            return None
        return tree.replace_at(path, Node(self.rng.choice(alts),
                                          [c.copy() for c in node.children]))

    def mutate(self, tree: Node) -> Node:
        if self.rng.random() < self.cfg.sem_mut_prob:            # NEW
            m = self._semantic_mutate(tree)
            if m is not None and self._valid(m):
                return m
        paths = tree.typed_paths()
        const_paths = [p for p, _ in paths
                       if tree.subtree_at(p).prod.name.startswith("const_")]
        var_paths = [p for p, _ in paths
                     if tree.subtree_at(p).prod.name in ("var_x", "var_y")]
        roll, new = self.rng.random(), None
        if roll < 0.45:
            new = self._mutate_regen(tree, paths)
        elif roll < 0.72 and const_paths:
            new = self._mutate_const(tree, const_paths)
        elif roll < 0.85 and var_paths:
            new = self._mutate_var(tree, var_paths)
        else:
            new = self._mutate_opswap(tree, paths)
            if new is None:
                new = self._mutate_regen(tree, paths)
        return new if (new is not None and self._valid(new)) else tree.copy()

    # ---- selection (NEW: lexicase) ----------------------------------------------------
    def _select(self, pop):
        if self.cfg.use_lexicase and pop and pop[0][2]:
            pool = list(pop)
            idxs = list(range(len(pop[0][2])))
            self.rng.shuffle(idxs)
            for i in idxs:
                if len(pool) <= 1:
                    break
                best = max(p[2][i] for p in pool)
                pool = [p for p in pool if p[2][i] == best]
            return self.rng.choice(pool)
        winner = self.rng.choice(pop)
        for _ in range(self.cfg.tournament_k - 1):
            c = self.rng.choice(pop)
            if c[1] > winner[1]:
                winner = c
        return winner

    # ---- evolution loop -------------------------------------------------------------------
    def run(self, seeds, pop_size: int, generations: int):
        rng = self.rng
        pop, seen_sig, seen_tr = [], set(), set()
        dups = 0

        def add(t: Optional[Node]) -> bool:
            if t is None or not self._valid(t):
                return False
            sig = self.ev.opt.sig(t)
            if sig in seen_sig:
                return False
            tr = self.ev.trace(t)
            if tr in seen_tr:                 # behavioral duplicate (NEW)
                nonlocal_dups[0] += 1
                return False
            seen_sig.add(sig)
            seen_tr.add(tr)
            r = self.ev.evaluate(t)
            pop.append((t, r.value, r.pass_vec))
            return True

        nonlocal_dups = [0]
        for t in seeds:
            add(t)
        guard = 0
        while len(pop) < pop_size and guard < pop_size * 20:
            guard += 1
            t = self.random_tree(T.EXPR, rng.randint(2, self.cfg.max_depth),
                                 full=rng.random() < 0.5)
            add(t)
        dups = nonlocal_dups[0]
        if not pop:
            return None, -1e9, [], {"gens": 0, "best": -1e9, "pop": 0, "dups": 0}

        best_tree, best_fit = max(pop, key=lambda x: x[1])
        gens = 0
        for _gen in range(generations):
            gens += 1
            pop.sort(key=lambda x: -x[1])
            if pop[0][1] >= self.cfg.target_fitness:
                break
            newpop = list(pop[: self.cfg.ga_elite])             # elitism
            attempts = 0
            while len(newpop) < pop_size and attempts < pop_size * 8:
                attempts += 1
                p1, p2 = self._select(pop), self._select(pop)
                if rng.random() < self.cfg.cx_prob:
                    child = self.crossover(p1[0], p2[0])
                    if child is None:
                        continue
                else:
                    child = p1[0].copy()
                if rng.random() < self.cfg.mut_prob:
                    child = self.mutate(child)
                add(child)
            pop = newpop
            cur = max(pop, key=lambda x: x[1])
            if cur[1] > best_fit:
                best_tree, best_fit = cur
        pop.sort(key=lambda x: -x[1])
        top = [(f, t) for t, f, _v in pop[:6]]
        return best_tree, best_fit, top, {"gens": gens, "best": best_fit,
                                          "pop": len(pop), "dups": dups}


# =============================================================================
# 10. MAP-ELITES ARCHIVE  (NEW: quality-diversity, Mouret & Clune 2015)
# =============================================================================

class MAPArchive:
    """Best program per (size bucket, operator-variety) niche. Keeps small
    clever programs from being buried by big accurate ones; seeds the GA and
    behavior cloning every iteration."""
    def __init__(self):
        self.cells: Dict[tuple, Tuple[float, Node]] = {}

    def descriptor(self, tree: Node) -> tuple:
        ops = sum(1 for p in tree.preorder() if not p.terminal)
        return (tree.size() // 4, min(ops, 8))

    def update(self, tree: Node, fit: float) -> None:
        d = self.descriptor(tree)
        cur = self.cells.get(d)
        if cur is None or fit > cur[0]:
            self.cells[d] = (fit, tree)

    def seeds(self, k: int) -> List[Node]:
        ranked = sorted(self.cells.values(), key=lambda x: -x[0])
        return [t for _f, t in ranked[:k]]


# =============================================================================
# 11. HYBRID ORCHESTRATOR
# =============================================================================

@dataclass
class Config:
    # search space
    max_depth: int = 6
    max_size: int = 64
    # MDP reward shaping
    step_penalty: float = -0.01
    size_penalty: float = 0.004
    perfect_bonus: float = 0.10
    min_reward: float = -0.05
    failure_value: float = -0.30
    target_fitness: float = 1.00
    # MCTS
    mcts_sims: int = 150
    episodes_per_iter: int = 3
    c_puct: float = 1.6
    initial_mix: float = 0.35
    mix_ramp: float = 0.12
    value_ramp: float = 0.12
    min_node_visits: int = 3
    max_train_nodes: int = 80
    dirichlet_eps: float = 0.25          # NEW
    dirichlet_alpha: float = 0.40        # NEW
    use_gumbel: bool = True              # NEW
    gumbel_budget: int = 48              # NEW
    # learning heads
    lr_policy: float = 0.08
    lr_value: float = 0.06
    l2: float = 1e-4
    # genetic algorithm
    ga_pop: int = 100
    ga_gens: int = 24
    ga_elite: int = 6
    ga_seeds: int = 8
    tournament_k: int = 4
    cx_prob: float = 0.85
    mut_prob: float = 0.30
    bc_k: int = 3
    use_lexicase: bool = True            # NEW
    sem_mut_prob: float = 0.30           # NEW
    guided_k: int = 8                    # NEW
    # beam decoder / outer loop
    beam_width: int = 6                  # NEW
    time_budget: float = 240.0           # NEW (seconds)
    iterations: int = 6
    seed: int = 7


def _spark(vals):
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return "▁" * len(vals)
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


class HybridSynthesizer:
    def __init__(self, task: SynthesisTask, cfg: Config,
                 policy: Optional[PolicyNet] = None, value: Optional[ValueNet] = None):
        self.cfg = cfg
        self.task = task
        self.rng = random.Random(f"{cfg.seed}|{task.name}")   # reproducible per task
        self.grammar = build_grammar()
        self.opt = Optimizer(self.grammar)
        self.ev = Evaluator(self.grammar, task, cfg, self.opt)
        self.env = CodeGenEnv(self.grammar, self.ev, cfg)
        self.policy = policy if policy is not None else PolicyNet(cfg.lr_policy, cfg.l2)
        self.value = value if value is not None else ValueNet(cfg.lr_value)
        self.mcts = MCTSAgent(self.env, self.policy, self.value, cfg, self.rng)
        self.ga = GeneticAlgorithm(self.grammar, self.ev, cfg, self.rng)
        self.archive = MAPArchive()
        self.best_tree: Optional[Node] = None
        self.best_reward: Optional[Reward] = None
        self.curve: List[float] = []
        self.iters_used = 0
        self.t_start = 0.0

    def _update_best(self, tree: Node, reward: Reward) -> None:
        if self.best_reward is None or reward.value > self.best_reward.value:
            self.best_reward, self.best_tree = reward, tree

    def _register(self, tree: Node) -> Reward:
        r = self.ev.evaluate(tree)
        self._update_best(tree, r)
        self.archive.update(tree, r.value)
        return r

    def _distill_champions(self, trees, k: int) -> float:
        """Behavior-clone GA/archive champions into the policy."""
        ces = []
        for tree in trees[:k]:
            if tree is None:
                continue
            rew = self.ev.evaluate(tree)
            prods = list(tree.preorder())
            s = self.env.reset()
            for i, prod in enumerate(prods):
                if s.is_terminal():
                    break
                acts = self.env.actions(s)
                if prod.name not in {a.name for a in acts}:
                    break
                dist = [1.0 if a.name == prod.name else 0.0 for a in acts]
                parent, ht = self.mcts.context(s)
                ces.append(self.policy.train_one(parent, ht, acts, dist, 1.0))
                G = rew.value + self.cfg.step_penalty * (len(prods) - i)
                self.value.train_one(self.mcts.value_feats(s), G, 1.0)
                s, _, _ = self.env.step(s, prod)
        return sum(ces) / len(ces) if ces else 0.0

    def solve(self) -> Tuple[Optional[Node], Optional[Reward]]:
        cfg = self.cfg
        mix, vw = cfg.initial_mix, 0.0
        self.t_start = time.time()
        n_prods = sum(len(v) for v in self.grammar.rules.values())
        warm = " [warm-started heads]" if getattr(self, "_warm", False) else ""
        print(f"\n{'=' * 70}")
        print(f"  TASK: {self.task.name} -- {self.task.description}")
        print(f"  train {len(self.task.examples)} | holdout {len(self.task.examples_holdout)}"
              f" | grammar {n_prods} productions | depth {cfg.max_depth}, size {cfg.max_size}{warm}")
        print("=" * 70)

        for it in range(1, cfg.iterations + 1):
            t0 = time.time()
            if time.time() - self.t_start > cfg.time_budget:
                print("\n   [time budget reached] stopping early.")
                break
            # ---- Phase 1: MCTS (reused tree, Dirichlet noise, Gumbel halving) ----
            samples, results, terminals, ep_best, gum = [], [], 0, -1e18, 0
            for _ in range(cfg.episodes_per_iter):
                res = self.mcts.search(self.env.reset(), cfg.mcts_sims, mix, vw)
                results.append(res)
                samples.extend(res.samples)
                terminals += res.terminals
                gum += self.mcts._gumbel_used
                ep_best = max(ep_best, res.best_value)
                if res.best_reward is not None:
                    self._register(res.best_tree)
            # ---- Phase 2: confidence-weighted distillation ----
            pb = [(p, h, a, d, w) for (p, h, a, d, q, f, w) in samples]
            vb = [(f, q, w) for (p, h, a, d, q, f, w) in samples]
            ce = self.policy.train_batch(pb)
            vmae = self.value.train_batch(vb)
            # ---- Phase 3: policy-only beam decode (inference probe) ----
            beam_val, beam_tree = None, None
            dec = beam_decode(self.env, self.mcts, self.policy, mix,
                              cfg.beam_width, cfg.max_size)
            if dec is not None:
                beam_tree, _lp = dec
                beam_val = self._register(beam_tree).value
            # ---- Phase 4: GA seeded by MCTS elites + beam + archive ----
            pool = [tp for res in results for tp in res.top]
            if self.best_tree is not None and self.best_reward is not None:
                pool.append((self.best_reward.value, self.best_tree))
            if beam_tree is not None and beam_val is not None:
                pool.append((beam_val, beam_tree))
            pool.extend((self.ev.evaluate(t).value, t) for t in self.archive.seeds(cfg.ga_seeds))
            pool.sort(key=lambda x: -x[0])
            elites, seen = [], set()
            for _v, t in pool:
                sig = self.opt.sig(t)
                if sig in seen:
                    continue
                seen.add(sig)
                elites.append(t)
                if len(elites) >= cfg.ga_seeds * 2:
                    break
            ga_tree, _fit, ga_top, ga_stats = self.ga.run(elites, cfg.ga_pop, cfg.ga_gens)
            if ga_tree is not None:
                self._register(ga_tree)
                for f, t in ga_top:
                    self.archive.update(t, f)
            # ---- Phase 5: clone GA + archive champions ----
            champs = [t for _f, t in ga_top][: cfg.bc_k]
            champs += [t for t in self.archive.seeds(cfg.bc_k + 2)
                       if t not in champs][: cfg.bc_k]
            bc = self._distill_champions(champs, cfg.bc_k + 2)

            mix = min(0.95, mix + cfg.mix_ramp)
            vw = min(0.60, vw + cfg.value_ramp)
            self.iters_used = it
            self.curve.append(self.best_reward.value if self.best_reward else -1.0)
            dt = time.time() - t0
            expr = self.grammar.emit(self.best_tree) if self.best_tree else "<none>"
            print(f"\n---- iteration {it}/{cfg.iterations}  ({dt:.1f}s,  mix={mix:.2f},"
                  f"  V*={vw:.2f}) " + "-" * 10)
            print(f"   MCTS    : {cfg.episodes_per_iter} x {cfg.mcts_sims} sims"
                  f" (gumbel {gum}) | best-in-iter {ep_best:+.3f} | terminals {terminals}"
                  f" | root reused (N={self.mcts._root.N if self.mcts._root else 0})")
            print(f"   distill : policy CE {ce:.4f} | value MAE {vmae:.4f} | BC loss {bc:.4f}")
            print(f"   beam    : policy-only width {cfg.beam_width} ->"
                  f" {('' if beam_val is None else format(beam_val, '+.3f'))}"
                  f"{'  <== search competence in policy' if beam_val is not None and beam_val > 0.5 else ''}")
            print(f"   GA      : {ga_stats['gens']} gens | best {ga_stats['best']:+.3f}"
                  f" | pop {ga_stats['pop']} | behav-dups {ga_stats['dups']}"
                  f" | archive {len(self.archive.cells)} cells")
            print(f"   overall : {self.best_reward.value:+.4f}   f(x, y) = {expr}   {_spark(self.curve)}")
            if self.best_reward.value >= cfg.target_fitness:
                print("\n   [converged] perfect program found.")
                break
        return self.best_tree, self.best_reward

    def report(self) -> None:
        tree, rew = self.best_tree, self.best_reward
        if tree is None or rew is None:
            print("  no program found")
            return
        hr = self.ev.evaluate_holdout(tree)
        print(f"\n{'=' * 70}")
        print(f"  BEST PROGRAM   reward {rew.value:+.4f}"
              f"   train {rew.pass_frac * 100:.1f}%   holdout {hr.pass_frac * 100:.1f}%")
        print("=" * 70)
        print(self.ev.source(tree))
        fn = self.ev.compile_tree(tree)
        print(f"\n  {'src':>3} {'x':>4} {'y':>4} | {'expected':>8} | {'got':>8} | verdict")
        print("  " + "-" * 46)
        rows = ([("T", a, e) for a, e in self.task.examples]
                + [("H", a, e) for a, e in self.task.examples_holdout])
        ok = 0
        for tag, (x, y), expected in rows:
            try:
                got = fn(x, y)
            except Exception as e:
                got = f"ERR:{type(e).__name__}"
            good = got == expected
            ok += good
            print(f"  {tag:>3} {x:>4} {y:>4} | {expected:>8} | {str(got):>8} |"
                  f" {'PASS' if good else 'FAIL'}")
        st = self.ev.stats
        print(f"\n  {ok}/{len(rows)} tests passed (T=train, H=holdout)")
        print(f"  compiler: {st['compiles']} unique programs compiled |"
              f" {st['cache_hits']} cache hits | {st['runtime_errors'] if 'runtime_errors' in st else 0}"
              f" runtime faults | {st['unique_traces']} unique behaviors")
        print(f"  canonicalizer: {len(self.opt._memo)} signatures |"
              f" archive: {len(self.archive.cells)} niches")
        print(f"  best-reward curve: {_spark(self.curve)}")
        print(f"  wall time: {time.time() - self.t_start:.1f}s")

    def summary(self) -> dict:
        tree = self.best_tree
        hr = self.ev.evaluate_holdout(tree) if tree is not None else None
        return {"task": self.task.name,
                "reward": self.best_reward.value if self.best_reward else -1e9,
                "train": self.best_reward.pass_frac if self.best_reward else 0.0,
                "hold": hr.pass_frac if hr else 0.0,
                "expr": self.grammar.emit(tree) if tree is not None else "<none>",
                "iters": self.iters_used}


# =============================================================================
# 12. SELF-TESTS & CLI
# =============================================================================

def selftest() -> None:
    print("running pipeline self-tests ...")
    rng = random.Random(0)
    g = build_grammar()
    cfg = Config()
    opt = Optimizer(g)
    ev = Evaluator(g, TASKS["max2"], cfg, opt)
    env = CodeGenEnv(g, ev, cfg)
    ga = GeneticAlgorithm(g, ev, cfg, rng)
    mk = lambda name, *kids: Node(g.by_name[name], list(kids))
    x, y = Node(g.by_name["var_x"]), Node(g.by_name["var_y"])

    for _ in range(200):                                   # 1. canonical semantics
        t = ga.random_tree(T.EXPR, rng.randint(1, cfg.max_depth),
                           full=rng.random() < 0.5)
        if t.size() > cfg.max_size:
            continue
        f1, f2 = ev.compile_tree(t), ev.compile_tree(opt.canonical(t))
        for a in range(-3, 4):
            for b in range(-3, 4):
                try:
                    va = f1(a, b)
                except Exception:
                    va = "ERR"
                try:
                    vb = f2(a, b)
                except Exception:
                    vb = "ERR"
                assert va == vb, (t.serialize(), a, b, va, vb)
    print("  [ok] canonicalization is semantics-preserving (200 programs x 49 inputs)")

    c0, c2, c3, c5 = (Node(g.by_name[f"const_{v}"]) for v in (0, 2, 3, 5))
    tru = Node(g.by_name["true"])
    assert opt.sig(mk("add", x, c0)) == opt.sig(x)                     # x+0 -> x
    assert opt.sig(mk("mul", c0, y)) == opt.sig(c0)                    # 0*y -> 0
    assert opt.sig(mk("add", c2, c3)) == opt.sig(c5)                   # 2+3 -> 5
    assert opt.sig(mk("not_", mk("not_", tru))) == opt.sig(tru)        # not not -> id
    assert opt.sig(mk("ite", tru, x, y)) == opt.sig(x)                 # dead branch
    print("  [ok] algebraic simplification & constant folding rules")

    for _ in range(200):                                   # 2. MDP replay
        t = ga.random_tree(T.EXPR, 5, full=False)
        s = env.reset()
        for prod in t.preorder():
            acts = env.actions(s)
            match = next((a for a in acts if a.name == prod.name), None)
            assert match is not None, "derivation action missing"
            s, _, _ = env.step(s, match)
        assert s.is_terminal()
        assert s.tree.serialize() == t.serialize()
    print("  [ok] MDP derivation replay reproduces trees exactly")

    for _ in range(200):                                   # 3. typed crossover
        a = ga.random_tree(T.EXPR, cfg.max_depth, False)
        b = ga.random_tree(T.EXPR, cfg.max_depth, False)
        c = ga.crossover(a, b)
        if c is None:
            continue
        assert c.depth() <= cfg.max_depth and c.size() <= cfg.max_size
        ev.evaluate(c)
    print("  [ok] strongly-typed crossover always yields compilable programs")

    agent = MCTSAgent(env, PolicyNet(), ValueNet(), cfg, random.Random(1))
    res = agent.search(env.reset(), 80, 0.5, 0.0)          # 4. MCTS + tree reuse
    assert res.best_tree is not None and not res.best_tree.has_holes()
    n1 = agent._root.N
    agent.search(env.reset(), 80, 0.5, 0.3)
    assert agent._root.N > n1, "tree reuse did not accumulate statistics"
    agent.search(env.reset(), 120, 0.5, 0.3)               # exercises Gumbel phase
    print("  [ok] MCTS search, root reuse and Gumbel phase")

    assert ev.trace(mk("add", x, c0)) == ev.trace(x)       # 5. traces
    print("  [ok] execution traces detect behavioral equivalence")

    arc = MAPArchive()                                     # 6. archive & lexicase
    for _ in range(60):
        t = ga.random_tree(T.EXPR, 4)
        r = ev.evaluate(t)
        arc.update(t, r.value)
    assert arc.seeds(3)
    p1 = (ga.random_tree(T.EXPR, 4), 0.5, (True, False))
    p2 = (ga.random_tree(T.EXPR, 4), 0.6, (False, True))
    assert ga._select([p1, p2]) in (p1, p2)
    print("  [ok] MAP-Elites archive + lexicase selection")

    d = beam_decode(env, agent, PolicyNet(), 0.8, 4, 30)   # 7. beam decoder
    assert d is None or not d[0].has_holes()
    print("  [ok] beam decoder returns complete programs")
    print("ALL SELF-TESTS PASSED")


def make_config(args) -> Config:
    cfg = Config(seed=args.seed)
    if args.quick:
        cfg.iterations, cfg.episodes_per_iter, cfg.mcts_sims = 3, 2, 80
        cfg.ga_pop, cfg.ga_gens = 60, 14
    elif args.full:
        cfg.iterations, cfg.episodes_per_iter, cfg.mcts_sims = 10, 6, 320
        cfg.ga_pop, cfg.ga_gens = 180, 40
    if args.sims:
        cfg.mcts_sims = args.sims
    if args.iters:
        cfg.iterations = args.iters
    if args.pop:
        cfg.ga_pop = args.pop
    if args.budget:
        cfg.time_budget = float(args.budget)
    if args.beam:
        cfg.beam_width = args.beam
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hybrid code-generation engine v2: MDP + MCTS + GA + compiler-in-the-loop")
    ap.add_argument("--task", default="max2", choices=sorted(list(TASKS) + ["all"]))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sims", type=int, default=0, help="override MCTS simulations")
    ap.add_argument("--iters", type=int, default=0, help="override outer iterations")
    ap.add_argument("--pop", type=int, default=0, help="override GA population")
    ap.add_argument("--beam", type=int, default=0, help="override beam width")
    ap.add_argument("--budget", type=float, default=0.0, help="time budget (seconds)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)
    if args.selftest:
        selftest()
        return
    cfg = make_config(args)
    names = sorted(TASKS) if args.task == "all" else [args.task]
    heads, summaries = None, []
    for i, name in enumerate(names):
        pol = heads[0] if heads else None
        val = heads[1] if heads else None
        synth = HybridSynthesizer(TASKS[name], cfg, policy=pol, value=val)
        if i > 0:
            synth._warm = True
        synth.solve()
        synth.report()
        summaries.append(synth.summary())
        heads = (synth.policy, synth.value)          # warm start for next task
    if len(summaries) > 1:
        print(f"\n{'=' * 70}\n  SUMMARY ACROSS TASKS (heads warm-started, in order)\n{'=' * 70}")
        print(f"  {'task':<11}{'reward':>8}{'train%':>8}{'holdout%':>10}  expression")
        for s in summaries:
            print(f"  {s['task']:<11}{s['reward']:>8.3f}{s['train'] * 100:>8.1f}"
                  f"{s['hold'] * 100:>10.1f}  {s['expr']}")


if __name__ == "__main__":
    main()