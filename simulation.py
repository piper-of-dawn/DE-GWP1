from collections import namedtuple
import plotly.graph_objects as go
from copy import deepcopy


class Simulation:
    def __init__(self, obj, attribute_name, possible_values, method):
        self.obj = obj
        self.attribute_name = attribute_name
        self.possible_values = possible_values
        self.method = method

    def run(self):
        Result = namedtuple("Result", ["attribute_value", "method_result"])
        self.results = []
        for value in self.possible_values:
            self.obj.reset()
            setattr(self.obj, self.attribute_name, value)
            method_result = getattr(self.obj, self.method)()
            print(f"{self.attribute_name} = {value} => {self.method} = {method_result}")
            self.results.append(Result(value, method_result))
        return self

    def graph(self):
        x = [result.attribute_value for result in self.results]
        y = [result.method_result for result in self.results]
        self.graph = go.Figure(data=go.Scatter(x=x, y=y))
        self.graph.update_layout(title=f"{self.method} vs {self.attribute_name}")
        self.graph.update_xaxes(title_text=str(self.attribute_name))
        self.graph.update_yaxes(title_text=str(self.method))
        self.graph.show()
        return self
