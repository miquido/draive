from draive.lmm.types import LMMOutputDecoder, LMMOutputInvalid, LMMOutputSelection
from draive.multimodal import MultimodalContent
from draive.parameters.model import DataModel

__all__ = ("lmm_output_decoder",)


def lmm_output_decoder(  # noqa: PLR0911
    output: LMMOutputSelection,
    /,
) -> LMMOutputDecoder:
    match output:
        case "auto":
            return _auto_output_conversion

        case "text":
            return _text_output_conversion

        case "json":
            return _json_output_conversion

        case "image":
            return _image_output_conversion

        case "audio":
            return _audio_output_conversion

        case "video":
            return _video_output_conversion

        case [*_]:  # we could prepare more specific method
            return _auto_output_conversion

        case model:
            return _prepare_model_output_conversion(model)


def _auto_output_conversion(
    content: MultimodalContent,
) -> MultimodalContent:
    return content


def _text_output_conversion(
    content: MultimodalContent,
) -> MultimodalContent:
    return content.text()


def _image_output_conversion(
    content: MultimodalContent,
) -> MultimodalContent:
    return content.images()


def _audio_output_conversion(
    content: MultimodalContent,
) -> MultimodalContent:
    return content.audio()


def _video_output_conversion(
    content: MultimodalContent,
) -> MultimodalContent:
    return content.video()


def _json_output_conversion(
    content: MultimodalContent,
) -> MultimodalContent:
    try:
        return MultimodalContent.of(DataModel.from_json(content.text().to_str()))

    except ValueError as exc:
        raise LMMOutputInvalid("Failed to decode JSON output", content) from exc


def _prepare_model_output_conversion(
    model: type[DataModel],
) -> LMMOutputDecoder:
    def _model_output_conversion(
        content: MultimodalContent,
    ) -> MultimodalContent:
        try:
            return MultimodalContent.of(model.from_json(content.text().to_str()))

        except ValueError as exc:
            raise LMMOutputInvalid("Failed to decode JSON output", content) from exc

    return _model_output_conversion
