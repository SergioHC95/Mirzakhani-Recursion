from typing import Tuple, Dict, Set, List
from itertools import combinations
from fractions import Fraction
from collections import defaultdict

Alpha = Tuple[int, ...]
Key = Tuple[int, int, Alpha]


def degree(g: int, n: int) -> int:
    return 3 * g - 3 + n


def is_stable(g: int, n: int) -> bool:
    return (2 * g - 2 + n) > 0 and n >= 1


def canonical(alpha: Alpha) -> Alpha:
    return tuple(sorted(alpha, reverse=True))


def split_index_sets(n: int):
    indices = list(range(n))
    seen = set()
    for i in range(1, n):
        for combo in combinations(indices, i):
            I = tuple(sorted(combo))
            J = tuple(sorted(set(indices) - set(I)))
            if (J, I) not in seen:
                seen.add((I, J))
                yield I, J


def format_node(key: Key) -> str:
    g, n, alpha = key
    return f"({g},{n},{alpha})"


class DependencyGraph:
    def __init__(self):
        self.graph: Dict[Key, Set[Key]] = defaultdict(set)
        self.visited: Set[Key] = set()

    def build(self, g: int, n: int, alpha: Alpha):
        root = (g, n, canonical(alpha))
        self._dfs(root)
        return self.graph

    def _dfs(self, key: Key):
        if key in self.visited:
            return
        self.visited.add(key)

        g, n, alpha = key
        alpha = canonical(alpha)

        if (g == 0 and n == 3 and alpha == (0, 0, 0)) or \
           (g == 1 and n == 1 and alpha in [(0,), (1,)]):
            return

        if not is_stable(g, n):
            return

        D = degree(g, n) - sum(alpha)
        if D < 0:
            return

        alpha_sorted = list(alpha)
        alpha1 = alpha_sorted[-1] if alpha_sorted else 0

        for k in range(D + 1):
            # (g, n-1)
            if n - 1 >= 1 and is_stable(g, n - 1):
                for j in range(n - 1):
                    if j == len(alpha_sorted) - 1:
                        continue
                    a2 = alpha1 + alpha_sorted[j] + k - 1
                    if a2 >= 0:
                        new_vec = [alpha_sorted[i] for i in range(n - 1) if i != j]
                        new_vec.append(a2)
                        new_alpha = canonical(tuple(new_vec))
                        dep = (g, n - 1, new_alpha)
                        self.graph[key].add(dep)
                        self._dfs(dep)

            # (g-1, n+1)
            if g > 0:
                dm = k + alpha1 - 2
                if dm >= 0 and dm % 2 == 0:
                    dm //= 2
                    for d1 in range(dm + 1):
                        d2 = 2 * dm - d1
                        new_vec = alpha_sorted[:-1] + [d1, d2]
                        new_alpha = canonical(tuple(new_vec))
                        dep = (g - 1, n + 1, new_alpha)
                        self.graph[key].add(dep)
                        self._dfs(dep)

            # Splitting
            up = k + alpha1 - 2
            if up >= 0:
                for I, J in split_index_sets(n - 1):
                    for d in range(up + 1):
                        vecI = [alpha_sorted[i] for i in I]
                        vecJ = [alpha_sorted[j] for j in J]
                        a1 = canonical(tuple(vecI + [d]))
                        a2 = canonical(tuple(vecJ + [up - d]))
                        for g1 in range(0, g + 1):
                            g2 = g - g1
                            n1, n2 = len(a1), len(a2)
                            if all([
                                is_stable(g1, n1),
                                is_stable(g2, n2),
                                degree(g1, n1) - sum(a1) >= 0,
                                degree(g2, n2) - sum(a2) >= 0,
                            ]):
                                dep1 = (g1, n1, a1)
                                dep2 = (g2, n2, a2)
                                self.graph[key].add(dep1)
                                self.graph[key].add(dep2)
                                self._dfs(dep1)
                                self._dfs(dep2)

    def as_edges(self) -> List[Tuple[Key, Key]]:
        return [(src, dst) for src, dsts in self.graph.items() for dst in dsts]

    def to_dot(self) -> str:
        lines = ["digraph G {"]
        for src, dsts in self.graph.items():
            src_label = format_node(src)
            for dst in dsts:
                dst_label = format_node(dst)
                lines.append(f'    "{src_label}" -> "{dst_label}";')
        lines.append("}")
        return "\n".join(lines)

    def print_graph(self):
        for src, dsts in self.graph.items():
            for dst in dsts:
                print(f"{format_node(src)} --> {format_node(dst)}")
