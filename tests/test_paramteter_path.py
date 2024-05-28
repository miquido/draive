from draive import DataModel, ParameterPath


class SequenceDataModel(DataModel):
    value: int


class DictDataModel(DataModel):
    key: str


class NestedDataModel(DataModel):
    value: float


class RecursiveDataModel(DataModel):
    more: "RecursiveDataModel | None"


class DataModel(DataModel):
    answer: str
    nested: NestedDataModel
    recursive: RecursiveDataModel
    list_models: list[SequenceDataModel]
    tuple_models: tuple[SequenceDataModel, ...]
    tuple_mixed_models: tuple[SequenceDataModel, DictDataModel, NestedDataModel]
    dict_models: dict[str, DictDataModel]


data_model: DataModel = DataModel(
    answer="testing",
    nested=NestedDataModel(
        value=3.14,
    ),
    recursive=RecursiveDataModel(
        more=RecursiveDataModel(
            more=None,
        ),
    ),
    list_models=[
        SequenceDataModel(value=65),
        SequenceDataModel(value=66),
    ],
    tuple_models=(
        SequenceDataModel(value=42),
        SequenceDataModel(value=21),
    ),
    tuple_mixed_models=(
        SequenceDataModel(value=42),
        DictDataModel(key="C"),
        NestedDataModel(value=3.33),
    ),
    dict_models={
        "A": DictDataModel(key="A"),
        "B": DictDataModel(key="B"),
    },
)


def test_id_path_points_to_self():
    path: ParameterPath[DataModel, DataModel] = DataModel.path(DataModel._)
    assert path(data_model) == data_model
    assert path.__repr__() == "DataModel"
    assert str(path) == ""


def test_attribute_path_points_to_attribute():
    path: ParameterPath[DataModel, str] = DataModel.path(DataModel._.answer)
    assert path(data_model) == data_model.answer
    assert path.__repr__() == "DataModel.answer"
    assert str(path) == "answer"


def test_nested_attribute_path_points_to_nested_attribute():
    path: ParameterPath[DataModel, float] = DataModel.path(DataModel._.nested.value)
    assert path(data_model) == data_model.nested.value
    assert path.__repr__() == "DataModel.nested.value"
    assert str(path) == "nested.value"


def test_recursive_attribute_path_points_to_attribute():
    path: ParameterPath[DataModel, RecursiveDataModel] = DataModel.path(DataModel._.recursive)
    assert path(data_model) == data_model.recursive
    assert path.__repr__() == "DataModel.recursive"
    assert str(path) == "recursive"


def test_list_item_path_points_to_item():
    path: ParameterPath[DataModel, SequenceDataModel] = DataModel.path(DataModel._.list_models[1])
    assert path(data_model) == data_model.list_models[1]
    assert path.__repr__() == "DataModel.list_models[1]"
    assert str(path) == "list_models[1]"


def test_tuple_item_path_points_to_item():
    path: ParameterPath[DataModel, SequenceDataModel] = DataModel.path(DataModel._.tuple_models[1])
    assert path(data_model) == data_model.tuple_models[1]
    assert path.__repr__() == "DataModel.tuple_models[1]"
    assert str(path) == "tuple_models[1]"


def test_mixed_tuple_item_path_points_to_item():
    path: ParameterPath[DataModel, DictDataModel] = DataModel.path(
        DataModel._.tuple_mixed_models[1]
    )
    assert path(data_model) == data_model.tuple_mixed_models[1]
    assert path.__repr__() == "DataModel.tuple_mixed_models[1]"
    assert str(path) == "tuple_mixed_models[1]"


def test_dict_item_path_points_to_item():
    path: ParameterPath[DataModel, DictDataModel] = DataModel.path(DataModel._.dict_models["B"])
    assert path(data_model) == data_model.dict_models["B"]
    assert path.__repr__() == "DataModel.dict_models[B]"
    assert str(path) == "dict_models[B]"
