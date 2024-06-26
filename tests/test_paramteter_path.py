from draive import DataModel, ParameterPath


class SequenceDataModel(DataModel):
    value: int


class DictDataModel(DataModel):
    key: str


class NestedDataModel(DataModel):
    value: float


class RecursiveDataModel(DataModel):
    more: "RecursiveDataModel | None"


class ExampleDataModel(DataModel):
    answer: str
    nested: NestedDataModel
    recursive: RecursiveDataModel
    list_models: list[SequenceDataModel]
    tuple_models: tuple[SequenceDataModel, ...]
    tuple_mixed_models: tuple[SequenceDataModel, DictDataModel, NestedDataModel]
    dict_models: dict[str, DictDataModel]


data_model: ExampleDataModel = ExampleDataModel(
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
    path: ParameterPath[ExampleDataModel, ExampleDataModel] = ExampleDataModel.path(
        ExampleDataModel._
    )
    assert path(data_model) == data_model
    assert path.__repr__() == "ExampleDataModel"
    assert str(path) == ""


def test_attribute_path_points_to_attribute():
    path: ParameterPath[ExampleDataModel, str] = ExampleDataModel.path(ExampleDataModel._.answer)
    assert path(data_model) == data_model.answer
    assert path.__repr__() == "ExampleDataModel.answer"
    assert str(path) == "answer"


def test_nested_attribute_path_points_to_nested_attribute():
    path: ParameterPath[ExampleDataModel, float] = ExampleDataModel.path(
        ExampleDataModel._.nested.value
    )
    assert path(data_model) == data_model.nested.value
    assert path.__repr__() == "ExampleDataModel.nested.value"
    assert str(path) == "nested.value"


def test_recursive_attribute_path_points_to_attribute():
    path: ParameterPath[ExampleDataModel, RecursiveDataModel] = ExampleDataModel.path(
        ExampleDataModel._.recursive
    )
    assert path(data_model) == data_model.recursive
    assert path.__repr__() == "ExampleDataModel.recursive"
    assert str(path) == "recursive"


def test_list_item_path_points_to_item():
    path: ParameterPath[ExampleDataModel, SequenceDataModel] = ExampleDataModel.path(
        ExampleDataModel._.list_models[1]
    )
    assert path(data_model) == data_model.list_models[1]
    assert path.__repr__() == "ExampleDataModel.list_models[1]"
    assert str(path) == "list_models[1]"


def test_tuple_item_path_points_to_item():
    path: ParameterPath[ExampleDataModel, SequenceDataModel] = ExampleDataModel.path(
        ExampleDataModel._.tuple_models[1]
    )
    assert path(data_model) == data_model.tuple_models[1]
    assert path.__repr__() == "ExampleDataModel.tuple_models[1]"
    assert str(path) == "tuple_models[1]"


def test_mixed_tuple_item_path_points_to_item():
    path: ParameterPath[ExampleDataModel, DictDataModel] = ExampleDataModel.path(
        ExampleDataModel._.tuple_mixed_models[1]
    )
    assert path(data_model) == data_model.tuple_mixed_models[1]
    assert path.__repr__() == "ExampleDataModel.tuple_mixed_models[1]"
    assert str(path) == "tuple_mixed_models[1]"


def test_dict_item_path_points_to_item():
    path: ParameterPath[ExampleDataModel, DictDataModel] = ExampleDataModel.path(
        ExampleDataModel._.dict_models["B"]
    )
    assert path(data_model) == data_model.dict_models["B"]
    assert path.__repr__() == "ExampleDataModel.dict_models[B]"
    assert str(path) == "dict_models[B]"


def test_id_path_set_updates_self():
    path: ParameterPath[ExampleDataModel, ExampleDataModel] = ExampleDataModel.path(
        ExampleDataModel._
    )
    assert path(data_model, updated=data_model) == data_model
    assert path.__repr__() == "ExampleDataModel"
    assert str(path) == ""


def test_attribute_path_set_updates_attribute():
    path: ParameterPath[ExampleDataModel, str] = ExampleDataModel.path(ExampleDataModel._.answer)
    updated: ExampleDataModel = path(data_model, updated="changed")
    assert updated != data_model
    assert updated.answer == "changed"


def test_nested_attribute_path_set_updates_nested_attribute():
    path: ParameterPath[ExampleDataModel, float] = ExampleDataModel.path(
        ExampleDataModel._.nested.value
    )
    updated: ExampleDataModel = path(data_model, updated=11.0)
    assert updated != data_model
    assert updated.nested.value == 11


def test_recursive_attribute_set_updates_attribute():
    path: ParameterPath[ExampleDataModel, RecursiveDataModel] = ExampleDataModel.path(
        ExampleDataModel._.recursive
    )
    updated: ExampleDataModel = path(data_model, updated=RecursiveDataModel(more=None))
    assert updated != data_model
    assert updated.recursive == RecursiveDataModel(more=None)


def test_list_item_path_set_updates_item():
    path: ParameterPath[ExampleDataModel, SequenceDataModel] = ExampleDataModel.path(
        ExampleDataModel._.list_models[1]
    )
    updated: ExampleDataModel = path(data_model, updated=SequenceDataModel(value=11))
    assert updated != data_model
    assert updated.list_models[1] == SequenceDataModel(value=11)


def test_tuple_item_path_set_updates_item():
    path: ParameterPath[ExampleDataModel, SequenceDataModel] = ExampleDataModel.path(
        ExampleDataModel._.tuple_models[1]
    )
    updated: ExampleDataModel = path(data_model, updated=SequenceDataModel(value=11))
    assert updated != data_model
    assert updated.tuple_models[1] == SequenceDataModel(value=11)


def test_mixed_tuple_item_set_updates_item():
    path: ParameterPath[ExampleDataModel, DictDataModel] = ExampleDataModel.path(
        ExampleDataModel._.tuple_mixed_models[1]
    )
    updated: ExampleDataModel = path(data_model, updated=DictDataModel(key="updated"))
    assert updated != data_model
    assert updated.tuple_mixed_models[1] == DictDataModel(key="updated")


def test_dict_item_path_set_updates_item():
    path: ParameterPath[ExampleDataModel, DictDataModel] = ExampleDataModel.path(
        ExampleDataModel._.dict_models["B"]
    )
    updated: ExampleDataModel = path(data_model, updated=DictDataModel(key="updated"))
    assert updated != data_model
    assert updated.dict_models["B"] == DictDataModel(key="updated")
