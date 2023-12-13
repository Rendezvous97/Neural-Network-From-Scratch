import math

# Class Value stores the data structures, operations and, most importantly,
# the gradient calculation for each operation using teh chain rule, and, 
# the main backward function that uses topological sort and chains through each
# backward function to get the gradients for each variable

class Value:

    def __init__(self, data, _children=[], _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None # Stores the derivative function for the particular operation

    def __repr__(self):
        return (f"Value={self.data}")
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data + other.data), [self, other], '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data * other.data), [self, other], '*')

        def _backward():
            self.grad += other.data * out.grad # if o = a*b, the do/da = b
            other.grad += self.data * out.grad # if o = a*b, the db/da = a
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def relu(self):
        out = Value((self.data if self.data > 0 else 0), [self,], 'ReLu')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0) * out.grad

        out._backward = _backward

        return out
    
    def __pow__(self, other):

        assert isinstance (other, (float, int))
        out = Value(self.data**other, [self,], f'**{other}')

        def _backward():
            self.grad += other*(self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1-t**2)*out.grad
        out._backward = _backward
        
        return out
    

    # Topological sorting to go from the output and move backwards to the initial input.
    # Then the backward function propagates through each node until the first
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0 # self.grad is 1.0 because self is the main output and the derivative of the output by the output is 1
        for node in reversed(topo):
            node._backward()        