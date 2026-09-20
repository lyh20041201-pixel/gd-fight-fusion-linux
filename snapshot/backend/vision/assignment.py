"""数据关联的分配算法。

ByteTrack 的两阶段关联都要解一个二部图最小代价完美匹配问题：
给定「轨迹 × 检测」的代价矩阵，选出总代价最小的一组一一对应。

本模块提供两种解法：

- ``hungarian``：匈牙利算法（Kuhn-Munkres，带对偶变量的 O(n²m) 实现），
  **全局最优**。开题报告第三章描述的就是这一种。
- ``greedy``：按代价从小到大贪心占位，**局部最优**，可能给出次优解。
  保留它是为了在报告中做对照，说明为什么要用匈牙利。

不引入 scipy：教室场景每路目标数在数十以内，纯 Python 实现足够快，
且避免为一个几十行的算法拉进一个大依赖。
"""

from __future__ import annotations

from typing import Sequence

Matrix = Sequence[Sequence[float]]

INFEASIBLE = float("inf")


def greedy(cost: Matrix) -> list[tuple[int, int]]:
    """贪心分配：每次取当前最小代价的可行对，占位后继续。

    结果不保证全局最优 —— 先被占走的那一对可能逼得后面只能选很差的配对。
    """
    pairs: list[tuple[float, int, int]] = []
    for i, row in enumerate(cost):
        for j, value in enumerate(row):
            if value < INFEASIBLE:
                pairs.append((float(value), i, j))
    pairs.sort()

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _value, i, j in pairs:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        matches.append((i, j))
    return sorted(matches)


def hungarian(cost: Matrix) -> list[tuple[int, int]]:
    """匈牙利算法（Kuhn-Munkres），返回总代价最小的 (行, 列) 配对。

    实现要点：
    - 逐行增广，用对偶变量 u（行势）、v（列势）维护松弛量，复杂度 O(n²m)；
    - 支持非方阵：行数多于列数时先转置再把结果转回来；
    - 不可行配对（代价为 inf）用一个足够大的有限值代替参与求解，
      求解后再由调用方按阈值剔除 —— 这与 ByteTrack 官方实现的做法一致。
    """
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []

    # 保证 行数 <= 列数
    if rows > cols:
        transposed = [[cost[i][j] for i in range(rows)] for j in range(cols)]
        return sorted((i, j) for j, i in hungarian(transposed))

    big = _finite_upper_bound(cost)
    matrix = [
        [(big if _is_infeasible(value) else float(value)) for value in row] for row in cost
    ]

    n, m = rows, cols
    u = [0.0] * (n + 1)          # 行势
    v = [0.0] * (m + 1)          # 列势
    assigned = [0] * (m + 1)     # assigned[j] = 匹配到第 j 列的行（1 基，0 表示未匹配）
    parent = [0] * (m + 1)       # 增广路径回溯

    for i in range(1, n + 1):
        assigned[0] = i
        j0 = 0
        slack = [INFEASIBLE] * (m + 1)
        visited = [False] * (m + 1)
        while True:
            visited[j0] = True
            i0 = assigned[j0]
            delta = INFEASIBLE
            j1 = 0
            for j in range(1, m + 1):
                if visited[j]:
                    continue
                reduced = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if reduced < slack[j]:
                    slack[j] = reduced
                    parent[j] = j0
                if slack[j] < delta:
                    delta = slack[j]
                    j1 = j
            for j in range(m + 1):
                if visited[j]:
                    u[assigned[j]] += delta
                    v[j] -= delta
                else:
                    slack[j] -= delta
            j0 = j1
            if assigned[j0] == 0:
                break
        # 沿增广路径回溯，翻转匹配
        while j0:
            j1 = parent[j0]
            assigned[j0] = assigned[j1]
            j0 = j1

    matches = [(assigned[j] - 1, j - 1) for j in range(1, m + 1) if assigned[j] > 0]
    # 代价被替换成 big 的配对本来就是不可行的，直接剔除
    return sorted((i, j) for i, j in matches if not _is_infeasible(cost[i][j]))


def solve(cost: Matrix, method: str = "hungarian") -> list[tuple[int, int]]:
    """按名称选择分配算法。未知名称退回匈牙利（全局最优的那个）。"""
    if method == "greedy":
        return greedy(cost)
    return hungarian(cost)


def total_cost(cost: Matrix, matches: Sequence[tuple[int, int]]) -> float:
    return sum(float(cost[i][j]) for i, j in matches)


def _is_infeasible(value: float) -> bool:
    return value >= INFEASIBLE or value != value  # inf 或 NaN


def _finite_upper_bound(cost: Matrix) -> float:
    """比矩阵中任何有限代价都大的值，用来代替 inf 参与求解。"""
    finite = [
        float(value) for row in cost for value in row if not _is_infeasible(value)
    ]
    if not finite:
        return 1.0
    span = max(finite) - min(finite)
    return max(finite) + (span + 1.0) * (len(cost) + len(cost[0]) + 1)
