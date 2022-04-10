import os
from collections import deque
import torch
import json
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset


# TODO: do numeral decomposition
# special_symbol = ['Symbol',         # variables
#                   'NegativeOne',    # replace to -1
#                   'Pi',             # no extra action needed
#                   'One',            # replece to 1
#                   'Half',           # replace to subtree: Mul,1,Pow,2,-1
#                   'Integer',        # replace to the corresponding integer value
#                   'Rational',       # replace to subtree: Mul, numerator, Pow, denominator, -1
#                   'Float'           # replace to subtree:
#                 ]


class TreeNode:
    def __init__(self, node_idx, node_symbol, left_child=None, right_child=None):
        self.node_idx = node_idx
        self.node_symbol = node_symbol
        self.left_child = left_child
        self.right_child = right_child
        self.left_processed = False
        self.right_processed = False


class BinaryTree:
    def __init__(self, preorder_nodeIdx, preorder_symbol):
        assert len(preorder_nodeIdx) == len(preorder_symbol)
        assert len(preorder_nodeIdx) > 0

        self.root = TreeNode(int(preorder_nodeIdx[0]), preorder_symbol[0])
        stack = [self.root]
        n = len(preorder_nodeIdx)
        i = 1

        while i < n:
            stack_top = len(stack) - 1
            if preorder_nodeIdx[i] == '#' and (not stack[stack_top].left_processed):
                left = None
                stack[stack_top].left_child = left
                stack[stack_top].left_processed = True
            elif preorder_nodeIdx[i] == '#' and stack[stack_top].left_processed:
                right = None
                stack[stack_top].right_child = right
                stack[stack_top].right_processed = True
                stack.pop()
            elif preorder_nodeIdx[i] != '#' and (not stack[stack_top].left_processed):
                left = TreeNode(int(preorder_nodeIdx[i]), preorder_symbol[i])
                stack[stack_top].left_child = left
                stack[stack_top].left_processed = True
                stack.append(left)
            else:
                right = TreeNode(int(preorder_nodeIdx[i]), preorder_symbol[i])
                stack[stack_top].right_child = right
                stack[stack_top].right_processed = True
                stack.pop()
                stack.append(right)

            i += 1

    def preorder(self, node, printSymbol=False):
        if node is None:
            return
        if printSymbol:
            print(node.node_symbol, end=' ')
        else:
            print(node.node_idx, end=' ')
        self.preorder(node.left_child, printSymbol=printSymbol)
        self.preorder(node.right_child, printSymbol=printSymbol)

    def inorder(self, node, printSymbol=False):
        if node is None:
            return
        self.inorder(node.left_child, printSymbol=printSymbol)
        if printSymbol:
            print(node.node_symbol, end=' ')
        else:
            print(node.node_idx, end=' ')
        self.inorder(node.right_child, printSymbol=printSymbol)

    def postorder(self, node, printSymbol=False):
        if node is None:
            return
        self.postorder(node.left_child, printSymbol=printSymbol)
        self.postorder(node.right_child, printSymbol=printSymbol)
        if printSymbol:
            print(node.node_symbol, end=' ')
        else:
            print(node.node_idx, end=' ')

    def levelorder_symbol(self):
        """
        return a list of symbols with the same order as node id
        :return: list
        """
        symbols = []
        queue = deque([self.root])
        while queue:
            cur = queue.popleft()
            symbols.append(cur.node_symbol)
            if cur.left_child:
                queue.append(cur.left_child)
            if cur.right_child:
                queue.append(cur.right_child)
        return symbols


class EquationTree:

    def __init__(self, vars, numNodes, variables, depth, nodeNum, func, label):
        self.vars = vars.split(',')
        self.numNodes = int(numNodes)
        self.variables = {}
        for var_name, var_idx in variables.items():
            self.variables[var_idx] = var_name
        self.nodeDepth = depth.split(',')
        self.depth = 0
        for d in self.nodeDepth:
            if d != '#':
                self.depth = max(self.depth, int(d))

        self.nodeNum = nodeNum.split(',')
        self.func = func.split(',')
        self.label = int(label)

        assert len(self.vars) == len(self.nodeDepth)
        assert len(self.nodeDepth) == len(self.nodeNum)
        assert len(self.nodeNum) == len(self.func)

        self.node_list = []
        # COO format edge index
        self.edge_list = [[], []]

        self._build()

    def _build(self):
        node_idx = []
        node_symbol = []
        for i in range(len(self.func)):
            node_idx.append(self.nodeNum[i])

            if self.func[i] == 'Symbol':
                var_idx = int(self.vars[i].split('_')[1])
                var_name = self.variables[var_idx]
                node_symbol.append(var_name)
            elif self.func[i] == "NegativeOne":
                node_symbol.append(-1)
            elif self.func[i] == "Pi":
                node_symbol.append(self.vars[i])
            elif self.func[i] == "One":
                node_symbol.append(1)
            elif self.func[i] == "Half":
                node_symbol.append(0.5)
            elif self.func[i] == 'Integer':
                node_symbol.append(int(self.vars[i]))
            elif self.func[i] == "Rational":
                numerator, denominator = self.vars[i].split('/')
                value = float(int(numerator) / int(denominator))
                node_symbol.append(value)
            elif self.func[i] == "Float":
                node_symbol.append(float(self.vars[i]))
            else:
                node_symbol.append(self.func[i])

        self.bt = BinaryTree(node_idx, node_symbol)

        # node symbol in the same order as node id
        node_queue = deque([self.bt.root])
        while node_queue:
            cur_node = node_queue.popleft()
            self.node_list.append(cur_node.node_symbol)
            if cur_node.left_child:
                s_node_idx, e_node_idx = cur_node.left_child.node_idx, cur_node.node_idx
                self.edge_list[0].append(s_node_idx)
                self.edge_list[1].append(e_node_idx)
                node_queue.append(cur_node.left_child)
            if cur_node.right_child:
                s_node_idx, e_node_idx = cur_node.right_child.node_idx, cur_node.node_idx
                self.edge_list[0].append(s_node_idx)
                self.edge_list[1].append(e_node_idx)
                node_queue.append(cur_node.right_child)

    def get_result(self):
        return self.node_list, self.edge_list

    def get_label(self):
        return self.label

    def get_depth(self):
        return self.depth

    def get_symbols(self):
        return self.bt.levelorder_symbol()


def load_single_equation(example):
    """
    :param example: a dictionary of schema
        {
            "equation": {
                "vars": value for each constant node 'NegativeOne', 'Pi', 'One',
                        'Half', 'Integer', 'Rational', 'Float'
                "numNodes": number of nodes in this tree, discounting #
                "variables": dictionary of ?,
                "depth": depth of each node in this tree
                "nodeNum": unique ids of each node
                "func": the actual list of nodes in this (binary) equation tree,
                    unary functions are still encoded as having two children,
                    the right one being NULL (#)
            },
            "label": "1" if the lhs of the equation equals rhs else "0"
        }
    :return: An EquationTree corresponding to 'example', paired with it's depth and it's label
    """
    vars = example['equation']['vars']
    numNodes = example['equation']['numNodes']
    variables = example['equation']['variables']
    depth = example['equation']['depth']
    nodeNum = example['equation']['nodeNum']
    func = example['equation']['func']
    label = example['label']

    et = EquationTree(vars, numNodes, variables, depth, nodeNum, func, label)
    return et, et.get_depth(), et.get_label()


def build_equation_tree_examples_list(train_jsonfile, test_jsonfile=None, val_jsonfile=None, depth=None):
    """

    :param train_jsonfile:
    :param test_jsonfile:
    :param val_jsonfile:
    :param depth: if depth is given, only return the equation tree of the given depth
    :return: a list of trio (BinaryTree, depth, label), a dict of all symbols
    """
    train_trios = []
    test_trios = []
    val_trios = []
    symbol_dict = {}

    with open(train_jsonfile, 'rt') as f:
        group_list = json.loads(f.read())

    for i, group in enumerate(group_list):
        for example in group:
            et, d, label = load_single_equation(example)
            if depth is not None and d not in depth:
                continue
            train_trios.append((et, d, label))
            et_symbols = et.get_symbols()
            for s in et_symbols:
                if s not in symbol_dict:
                    symbol_dict[s] = len(symbol_dict)

    if test_jsonfile is not None:
        with open(test_jsonfile, 'rt') as f:
            test_groups = json.loads(f.read())

        for i, group in enumerate(test_groups):
            for example in group:
                et, d, label = load_single_equation(example)
                test_trios.append((et, d, label))
                et_symbols = et.get_symbols()
                for s in et_symbols:
                    if s not in symbol_dict:
                        symbol_dict[s] = len(symbol_dict)

    if val_jsonfile is not None:
        with open(val_jsonfile, 'rt') as f:
            val_groups = json.loads(f.read())

        for i, group in enumerate(val_groups):
            for example in group:
                et, d, label = load_single_equation(example)
                val_trios.append((et, d, label))
                et_symbols = et.get_symbols()
                for s in et_symbols:
                    if s not in symbol_dict:
                        symbol_dict[s] = len(symbol_dict)

    return symbol_dict, train_trios, test_trios, val_trios


class GraphExprDataset(InMemoryDataset):
    """
    Construct the dataset we will use when training the model
    """
    def __init__(self, root, train_filename, test_filename, val_filename):
        self.symbol_vocab = None
        self.train_filename = f'dataset/raw/{train_filename}'
        self.test_filename = f'dataset/raw/{test_filename}'
        self.val_filename = f'dataset/raw/{val_filename}'
        self.train_size = 0
        self.test_size = 0
        self.val_size = 0
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = [self.train_filename, self.test_filename, self.val_filename]
        return files
        
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def _generate_graph(self, equation_tree: EquationTree):
        """

        :param equation_tree:
        :return:
        """
        node_list, edge_list = equation_tree.get_result()

        node_feature = []
        for node_symbol in node_list:
            feature = [0] * len(self.symbol_vocab)
            feature[self.symbol_vocab[node_symbol]] = 1

            node_feature.append(feature)

        x = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long)

        return x, edge_index

    def process(self):
        data_list = []

        symbol_dict, train_trios, test_trios, val_trios = build_equation_tree_examples_list(self.train_filename, self.test_filename, self.val_filename)
        self.symbol_vocab = symbol_dict
        trio_list = train_trios + test_trios + val_trios
        self.train_size = len(train_trios)
        self.test_size = len(test_trios)
        self.val_size = len(val_trios)

        for i, trio in tqdm(enumerate(trio_list)):
            et, d, label = trio

            x, edge_index = self._generate_graph(et)
            y = [label]
            y = torch.tensor(y, dtype=torch.long)

            # feed the graph and label into Data
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # pad the dataset TODO: check here if it's needed to pad the dataset( I guess it's no need to pad)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
	dataset = GraphExprDataset('dataset', '40k_train.json', '40k_test.json', '40k_val_shallow.json')
	print(f'train size: {dataset.train_size}')
	print(f'test size: {dataset.test_size}')
	print(f'val size: {dataset.val_size}')

