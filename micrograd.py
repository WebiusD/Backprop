import code
import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, children=(), op='', label='', parent=None):
        self.data = data
        self.grad = 0
        self.label = label

        self._parent = parent
        self._children = list(children)
        self._op = op

    def _eigen_to_parent_grad(self):
        parent_op = self._parent._op
        siblings = [child for child in self._parent._children if child != self]

        if parent_op == '*':
            # code.interact(local=locals())
            result = math.prod([s.data for s in siblings])
            
            return result
        if parent_op == '+':
            # count how often self is encountered in siblings:
            return sum([1 if s==self else 0 for s in siblings]) + 1
            
    def backward(self):
        if self._parent is None:
            # this is the root node, so it's gradient is 1:
            self.grad = 1.0
        else:
            # this nodes grad is the product of its parent's derivate and its local derivative (chain rule)
            self.grad = self._parent.grad * self._eigen_to_parent_grad()

        # visit all child nodes:
        for child in self._children:
            child.backward()

    def __rassign__(self, other):
        v = Value(other.data, (self, ), '=')
        other.parent = v
        return v
    
    def __add__(self, other):
        # other = other if isinstance(other, Value) else Value(other)
        v = Value(self.data + other.data, (self, other), '+')
        self._parent = other._parent = v
        return v

    def __mul__(self, other):
        # other = other if isinstance(other, Value) else Value(other)
        v = Value(self.data * other.data, (self, other), '*')
        self._parent = other._parent = v
        return v

    # def relu(self):
    #     v = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    #     self.parent = v
    #     return v

    # def __neg__(self): # -self
    #     return self * -1

    # def __radd__(self, other): # other + self
    #     return self + other

    # def __sub__(self, other): # self - other
    #     return self + (-other)

    # def __rsub__(self, other): # other - self
    #     return other + (-self)

    # def __rmul__(self, other): # other * self
    #     return self * other

    # def __truediv__(self, other): # self / other
    #     return self * other**-1

    # def __rtruediv__(self, other): # other / self
    #     return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    

def compute_grad_of(label: str):
    h = 0.001
    value_to_bump = next((value for value in [a, b, c] if value.label==label), None)
    
    L1 = a*b + c
    # bump the value:
    value_to_bump.data += h
    L2 = a*b + c

    grad = (L2.data-L1.data)/h
    return grad

def create_graph(root):
    g = Digraph('NN', filename='nn.gv')
    nodes, edges = set(), set()

    def visit(node):
        nodes.add(node)

        for child in node._children:
            edges.add((child, node))
            visit(child)

    visit(root)

    # build the graph:
    for node in nodes:
        uid = str(id(node))
        g.node(name=uid, label=f"{node.data} | {node.grad}", shape='record')
        if node._op:
            # this node is a result of some operation:
            g.node(name=uid + node._op, label=f"{node._op}")
            g.edge(uid + node._op, uid) 
    
    g.view()
    # for n1, n2 in edges:

    
a = Value(2.0, label='a')
b = Value(-3.5, label='b')
c = Value(1.0, label='c')

L = a*b + c
L.backward()
print(f"L_grad={L.grad}, a_grad={a.grad}, b_grad={b.grad}, c_grad={c.grad}")

a_grad = compute_grad_of('a')
b_grad = compute_grad_of('b')
c_grad = compute_grad_of('c')
print(f"{a_grad=}, {b_grad=}, {c_grad=}")

create_graph(L)