from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field

@dataclass
class Port:
    name: str
    type: str = "DataFrame"

@dataclass
class AlgorithmParameter:
    name: str
    type: str
    default: Any
    label: str
    description: str
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    widget: Optional[str] = None
    priority: str = "non-critical"  # critical or non-critical
    role: str = "parameter" # input, output, or parameter

    def to_dict(self):
        d = {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "label": self.label,
            "description": self.description,
            "priority": self.priority,
            "role": self.role
        }
        if self.min is not None: d["min"] = self.min
        if self.max is not None: d["max"] = self.max
        if self.step is not None: d["step"] = self.step
        if self.options is not None: d["options"] = self.options
        if self.widget is not None: d["widget"] = self.widget
        return d

@dataclass
class Algorithm:
    id: str
    name: str
    category: str # ID of the category, e.g. "load_data"
    prompt: str
    algo_module: Any = None # The module containing the function
    template: str = ""  # Code template for algorithm
    parameters: List[AlgorithmParameter] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    func: Optional[Callable] = field(init=False, default=None)  # The actual function implementing the algorithm

    def initialize(self):
        """
        Initialize parameters and templates from the function.
        This should be called at startup.
        """
        # 0. Resolve function from module if not present
        if self.algo_module and not self.func:
            try:
                self.func = getattr(self.algo_module, self.id)
            except AttributeError:
                print(f"Error: Function '{self.id}' not found in module '{self.algo_module.__name__}'")
                return

        if self.func:
            # 1. Extract parameters if not already provided
            if not self.parameters:
                from .utils import extract_parameters_from_func
                all_params = extract_parameters_from_func(self.func)
                
                # Separate parameters and inputs based on role
                inferred_inputs = []
                inferred_params = []
                
                for param in all_params:
                    if param.role == 'input':
                        inferred_inputs.append(Port(name=param.name, type=param.type))
                    else:
                        inferred_params.append(param)
                
                self.parameters = inferred_params
                
                # If inputs not provided manually, use inferred
                if not self.inputs and inferred_inputs:
                    self.inputs = inferred_inputs
            
            # 2. Extract imports if not provided
            if not self.imports:
                from .utils import extract_imports_from_func
                self.imports = extract_imports_from_func(self.func)
            
            # 3. Infer outputs if not provided
            if not self.outputs:
                import inspect
                import typing
                sig = inspect.signature(self.func)
                if sig.return_annotation is not inspect.Signature.empty:
                    ret_type = sig.return_annotation
                    
                    # Helper function to check if a type is DataFrame-like
                    def is_dataframe_type(t):
                        if isinstance(t, str):
                            return "DataFrame" in t
                        elif hasattr(t, "__name__") and "DataFrame" in t.__name__:
                            return True
                        elif "DataFrame" in str(t):
                            return True
                        return False

                    origin = typing.get_origin(ret_type)
                    if origin is tuple or origin is typing.Tuple:
                        args = typing.get_args(ret_type)
                        for i, arg in enumerate(args):
                            if is_dataframe_type(arg):
                                port_name = f"df_out_{i+1}"
                                self.outputs.append(Port(name=port_name, type="DataFrame"))
                    else:
                        # Single return value
                        if is_dataframe_type(ret_type):
                             self.outputs = [Port(name="df_out", type="DataFrame")]
            
            # 4. Generate templates
            from .template_generator import generate_full_template
            import inspect
            
            try:
                source = inspect.getsource(self.func)
            except OSError:
                source = "# Source not available"

            self.template = generate_full_template(
                self.name, 
                source
            )
  
    def to_prompt_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt
        }
    
    def to_port_dict(self):
        """Convert ports to dictionary format for frontend"""
        return {
            "inputs": [{"name": port.name, "type": port.type} for port in self.inputs],
            "outputs": [{"name": port.name, "type": port.type} for port in self.outputs]
        }
    
    @classmethod
    def from_func(cls, func: Callable, module: Any) -> Optional['Algorithm']:
        """
        从函数对象创建 Algorithm 实例。
        如果函数 Docstring 中没有 Algorithm: 块，则返回 None。
        """
        import inspect
        from .utils import parse_algorithm_metadata
        
        docstring = inspect.getdoc(func)
        metadata = parse_algorithm_metadata(docstring)
        
        if not metadata:
            return None
            
        # 自动推断 ID 为函数名
        algo_id = func.__name__
        
        # 构建实例
        algo = cls(
            id=algo_id,
            name=metadata.get('name', algo_id),
            category=metadata.get('category', 'uncategorized'),
            prompt=metadata.get('prompt', ''),
            imports=metadata.get('imports', []),
            algo_module=module,
            # inputs/outputs 将在 initialize() 中自动推断
        )
        
        # 立即初始化以提取参数和端口
        algo.initialize() 
        return algo

@dataclass
class AlgorithmCategory:
    id: str
    label: str
    algorithms: List[Algorithm] = field(default_factory=list)

    def add_algorithm(self, algo: Algorithm):
        self.algorithms.append(algo)

    def to_prompt_dict(self):
        return {
            "label": self.label,
            "algorithms": [algo.to_prompt_dict() for algo in self.algorithms]
        }
