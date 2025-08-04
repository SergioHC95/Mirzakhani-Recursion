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


def split_index_sets(n_minus_1: int):
    """
    Unordered bipartitions (I, J) of {0,..,n_minus_1-1}, unique up to swap.
    Include the empty subset case to match Mathematica's Subsets[..., k].
    """
    inds = tuple(range(n_minus_1))
    out: List[Tuple[List[int], List[int]]] = []
    half = n_minus_1 // 2

    for r in range(0, half + 1):  # <-- start at 0, not 1
        for I in combinations(inds, r):
            Iset = set(I)
            J = tuple(x for x in inds if x not in Iset)
            # keep one representative per unordered pair:
            if (len(I), I) <= (len(J), J):
                out.append((list(I), list(J)))
    return out



def format_node(key: Key) -> str:
    g, n, alpha = key
    return f"({g},{n},{alpha})"


class DependencyGraph:
    """
    Builds the full dependency graph of (g, n, alpha) triples used in the
    finite recursion that computes intersection numbers via rx_rec_opt.

    The graph maps each (g, n, alpha) node to its set of direct recursive dependencies,
    reflecting every rx_lookup call made during the computation.

    This version exactly matches the logic of rx_rec_opt.
    """

    def __init__(self):
        self.graph: Dict[Key, Set[Key]] = defaultdict(set)
        self.visited: Set[Key] = set()

    def build(self, g: int, n: int, alpha: Alpha):
        """Build the full dependency graph rooted at (g, n, alpha)."""
        root = (g, n, canonical(alpha))
        self._dfs(root)
        return self.graph

    def _dfs(self, key: Key):
        """Depth-first traversal to recursively build dependencies."""
        if key in self.visited:
            return
        self.visited.add(key)

        g, n, alpha = key
        alpha = canonical(alpha)

        # Base cases: known seed values, no dependencies
        if (g == 0 and n == 3 and alpha == (0, 0, 0)) or \
           (g == 1 and n == 1 and alpha in [(0,), (1,)]):
            return

        # Moduli space must be stable
        if not is_stable(g, n):
            return

        D = degree(g, n) - sum(alpha)
        if D < 0:
            return

        alpha_sorted = list(alpha)
        alpha1 = alpha_sorted[-1] if alpha_sorted else 0

        for k in range(D + 1):
            # Optional: skip if A_k(k) = 0 (insert real A_k(k) test here if desired)
            # if A_k(k) == 0:
            #     continue

            # --- First branch: (g, n-1)
            if n - 1 >= 1 and is_stable(g, n - 1):
                for j in range(n - 1):
                    a2 = alpha1 + alpha_sorted[j] + k - 1
                    if a2 >= 0:
                        # Drop entries at j and the last (min), add a2
                        new_vec = [alpha_sorted[i] for i in range(n - 1) if i != j]
                        new_vec.append(a2)
                        new_alpha = canonical(tuple(new_vec))
                        dep = (g, n - 1, new_alpha)
                        self.graph[key].add(dep)
                        self._dfs(dep)

            # --- Second branch: (g-1, n+1)
            m = k + alpha1 - 2
            if g > 0 and m >= 0:
                for d1 in range(m // 2 + 1):
                    d2 = m - d1
                    new_vec = alpha_sorted[:-1] + [d1, d2]
                    new_alpha = canonical(tuple(new_vec))
                    dep = (g - 1, n + 1, new_alpha)
                    self.graph[key].add(dep)
                    self._dfs(dep)

            # --- Third branch: splitting into (g1, n1), (g2, n2)
            if n - 1 >= 1 and m >= 0:
                for I, J in split_index_sets(n - 1):
                    for d in range(m + 1):
                        vecI = [alpha_sorted[i] for i in I]
                        vecJ = [alpha_sorted[j] for j in J]
                        a1 = canonical(tuple(vecI + [d]))
                        a2 = canonical(tuple(vecJ + [m - d]))
                        for g1 in range(g + 1):
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
