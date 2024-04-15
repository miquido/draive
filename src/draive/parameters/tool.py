from draive.parameters.function import Function, ParametrizedFunction
from draive.parameters.specification import ToolSpecification


class ParametrizedTool[**Args, Result](ParametrizedFunction[Args, Result]):
    def __init__(
        self,
        /,
        name: str,
        *,
        function: Function[Args, Result],
        description: str | None = None,
    ) -> None:
        super().__init__(function=function)
        self.name: str = name

        if specification := self.parameters.parameters_specification:
            self.specification: ToolSpecification = {
                "type": "function",
                "function": {
                    "name": self.name,
                    "parameters": specification,
                    "description": description or "",
                },
            }
        else:
            raise TypeError(f"{function.__qualname__} can't be represented as a tool")
