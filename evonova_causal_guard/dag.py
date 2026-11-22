"""Pearl Causal DAG, Bayes-Ball d-separation, do-mutilation, and backdoor adjustment solver."""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Iterable


class CycleError(ValueError):
    """Raised when an edge insertion would introduce a cycle into a CausalDAG."""


class CausalDAG:
    """Directed Acyclic Graph supporting Pearl's do-operator and Backdoor Criterion."""

    def __init__(self, edges: Iterable[tuple[str, str]] | None = None) -> None:
        self._parents: dict[str, set[str]] = {}
        self._children: dict[str, set[str]] = {}
        if edges:
            for parent, child in edges:
                self.add_edge(parent, child)

    def add_node(self, node: str) -> None:
        """Register an isolated node if not already present."""
        self._parents.setdefault(node, set())
        self._children.setdefault(node, set())

    def add_edge(self, parent: str, child: str) -> None:
        """Add directed edge `parent -> child`, rejecting cycles and self-loops."""
        if parent == child:
            raise CycleError(f"Self-loop detected on node '{parent}'.")
        self.add_node(parent)
        self.add_node(child)
        if parent == child or self.has_directed_path(child, parent):
            raise CycleError(f"Adding edge '{parent} -> {child}' would introduce a cycle.")
        self._children[parent].add(child)
        self._parents[child].add(parent)

    @property
    def nodes(self) -> set[str]:
        """Return all node identifiers in the graph."""
        return set(self._parents.keys())

    def parents(self, node: str) -> set[str]:
        """Return direct parents of `node`."""
        return set(self._parents.get(node, set()))

    def children(self, node: str) -> set[str]:
        """Return direct children of `node`."""
        return set(self._children.get(node, set()))

    def ancestors(self, node: str) -> set[str]:
        """Return all proper ancestors of `node`."""
        seen: set[str] = set()
        stack = list(self._parents.get(node, set()))
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(self._parents.get(cur, set()))
        return seen

    def descendants(self, node: str) -> set[str]:
        """Return all proper descendants of `node`."""
        seen: set[str] = set()
        stack = list(self._children.get(node, set()))
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(self._children.get(cur, set()))
        return seen

    def has_directed_path(self, source: str, target: str) -> bool:
        """Return True if a directed path exists from `source` to `target`."""
        if source not in self._children or target not in self._children:
            return False
        if source == target:
            return True
        return target in self.descendants(source)

    def copy(self) -> CausalDAG:
        """Return an independent deep copy of this DAG."""
        clone = CausalDAG()
        for node in self.nodes:
            clone.add_node(node)
        for parent, kids in self._children.items():
            for child in kids:
                clone._children[parent].add(child)
                clone._parents[child].add(parent)
        return clone

    def do(self, interventions: str | Iterable[str]) -> CausalDAG:
        """Apply Pearl's do-operator `do(X)` by deleting incoming edges into intervened nodes."""
        targets = {interventions} if isinstance(interventions, str) else set(interventions)
        mutilated = self.copy()
        for target in targets:
            for parent in list(mutilated._parents.get(target, set())):
                mutilated._children[parent].discard(target)
                mutilated._parents[target].discard(parent)
        return mutilated

    def without_outgoing(self, node: str) -> CausalDAG:
        """Return $G_{\\underline{X}}$ with all outgoing edges from `node` removed."""
        mutilated = self.copy()
        for child in list(mutilated._children.get(node, set())):
            mutilated._parents[child].discard(node)
            mutilated._children[node].discard(child)
        return mutilated

    def d_separated(
        self,
        x: str,
        y: str,
        conditioned: set[str] | Iterable[str] | None = None,
    ) -> bool:
        """Determine exact d-separation `X _|_ Y | Z` using Bayes-Ball reachability."""
        if x not in self._parents or y not in self._parents:
            raise KeyError(f"Unknown node(s) in d_separated query: ({x!r}, {y!r}).")
        z_set = set(conditioned or set())
        if x in z_set or y in z_set:
            return x != y or x in z_set
        if x == y:
            return False

        ancestors_of_z: set[str] = set(z_set)
        for z in z_set:
            if z in self._parents:
                ancestors_of_z.update(self.ancestors(z))

        visited: set[tuple[str, str]] = set()
        queue: deque[tuple[str, str]] = deque([(x, "up"), (x, "down")])

        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node == y and node not in z_set:
                return False

            if direction == "up" and node not in z_set:
                for p in self._parents[node]:
                    if (p, "up") not in visited:
                        queue.append((p, "up"))
                for c in self._children[node]:
                    if (c, "down") not in visited:
                        queue.append((c, "down"))
            elif direction == "down":
                if node not in z_set:
                    for c in self._children[node]:
                        if (c, "down") not in visited:
                            queue.append((c, "down"))
                if node in ancestors_of_z:
                    for p in self._parents[node]:
                        if (p, "up") not in visited:
                            queue.append((p, "up"))

        return True

    def find_backdoor_adjustment_sets(
        self,
        treatment: str,
        outcome: str,
        max_size: int | None = None,
    ) -> list[set[str]]:
        """Return minimal adjustment sets `Z` satisfying Pearl's Backdoor Criterion."""
        if treatment not in self._parents or outcome not in self._parents:
            raise KeyError(f"Unknown node(s): ({treatment!r}, {outcome!r}).")

        desc_of_t = self.descendants(treatment)
        candidates = sorted(n for n in self.nodes if n not in {treatment, outcome} and n not in desc_of_t)
        limit = len(candidates) if max_size is None else min(max_size, len(candidates))
        g_under = self.without_outgoing(treatment)

        valid_minimal: list[set[str]] = []
        for size in range(0, limit + 1):
            for combo in combinations(candidates, size):
                z_set = set(combo)
                if any(prev.issubset(z_set) for prev in valid_minimal):
                    continue
                if g_under.d_separated(treatment, outcome, z_set):
                    valid_minimal.append(z_set)
        return valid_minimal

    def _simple_undirected_paths(self, start: str, end: str) -> list[list[str]]:
        """Enumerate simple undirected paths between `start` and `end`."""
        results: list[list[str]] = []

        def dfs(curr: str, path: list[str], seen: set[str]) -> None:
            if curr == end:
                results.append(list(path))
                return
            neighbors = sorted(self._parents[curr] | self._children[curr])
            for nxt in neighbors:
                if nxt not in seen:
                    seen.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path, seen)
                    path.pop()
                    seen.remove(nxt)

        dfs(start, [start], {start})
        return results

    def detect_collider_bias(
        self,
        treatment: str,
        outcome: str,
        conditioned: set[str] | Iterable[str] | None = None,
    ) -> list[str]:
        """Identify conditioned nodes that unblock collider bias along paths between X and Y."""
        z_set = set(conditioned or set())
        if not z_set or treatment not in self._parents or outcome not in self._parents:
            return []

        biased: set[str] = set()
        for path in self._simple_undirected_paths(treatment, outcome):
            if len(path) < 3:
                continue
            unblocked = True
            activated_z: set[str] = set()
            for idx in range(1, len(path) - 1):
                prev_n, mid_n, next_n = path[idx - 1], path[idx], path[idx + 1]
                is_collider = prev_n in self._parents[mid_n] and next_n in self._parents[mid_n]
                if is_collider:
                    collider_scope = {mid_n} | self.descendants(mid_n)
                    hit = z_set & collider_scope
                    if not hit:
                        unblocked = False
                        break
                    activated_z.update(hit)
                else:
                    if mid_n in z_set:
                        unblocked = False
                        break
            if unblocked and activated_z:
                biased.update(activated_z)

        return sorted(biased)
