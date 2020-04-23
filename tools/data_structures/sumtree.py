import numpy as np


class Node:
    def __init__(self, left: Node, right: Node, is_leaf: bool = False, idx: int = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value: float, idx: int):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


class SumTree:
    def __init__(self, inputs: list):
        self.root_node, self.leaf_nodes = self.create_tree(inputs)

    @staticmethod
    def create_tree(input: list):
        nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
        leaves = nodes
        while len(nodes) > 1:
            inodes = iter(nodes)
            nodes = [Node(*pair) for pair in zip(inodes, inodes)]
        return nodes[0], leaves

    def get_node(self, value: float, node: Node):
        if node.is_leaf:
            return node
        if node.left.value >= value:
            return self.get_node(value, node.left)
        else:
            return self.get_node(value - node.left.value, node.right)

    def update_node(self, node: Node, new_value: float):
        change = new_value - node.value
        node.value = new_value
        self.propagate_changes(change, node.parent)

    def propagate_changes(self, change: float, node: Node):
        node.value += change
        if node.parent is not None:
            self.propagate_changes(change, node.parent)


# input = [1, 4, 2, 3]
# root_node, leaf_nodes = create_tree(input)
#
#
# def demonstrate_sampling(root_node: Node):
#     tree_total = root_node.value
#     iterations = 1000000
#     selected_vals = []
#     for i in range(iterations):
#         rand_val = np.random.uniform(0, tree_total)
#         selected_val = get_node(rand_val, root_node).value
#         selected_vals.append(selected_val)
#
#     return selected_vals
#
#
# selected_vals = demonstrate_sampling(root_node)
# # the below print statement should output ~4
# print(f"Should be ~4: {sum([1 for x in selected_vals if x == 4]) / sum([1 for y in selected_vals if y == 1])}")
