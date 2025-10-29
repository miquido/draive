import builtins
from collections.abc import Collection, Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, Self, cast, final, overload

from haiway import META_EMPTY, MISSING, BasicValue, Meta, MetaValues, Missing

from draive.multimodal.artifact import ArtifactContent
from draive.multimodal.text import TextContent
from draive.parameters import DataModel
from draive.resources import MimeType, ResourceContent, ResourceReference

__all__ = (
    "Multimodal",
    "MultimodalContent",
    "MultimodalContentPart",
    "MultimodalTag",
)

MultimodalContentPart = TextContent | ResourceReference | ResourceContent | ArtifactContent


@final
class MultimodalContent(DataModel):
    """
    Immutable sequence of multimodal parts with normalization utilities.

    ``MultimodalContent`` holds an ordered collection of text blocks,
    resources, and artifacts. Construction via ``of(...)`` performs light
    normalization such as merging adjacent textual elements when possible.

    Attributes
    ----------
    parts : Sequence[MultimodalContentPart]
        The ordered, immutable parts of this content.
    """

    empty: ClassVar[Self]  # defined after the class

    @classmethod
    def of(
        cls,
        *elements: "Multimodal",
    ) -> Self:
        """
        Construct a normalized ``MultimodalContent``.

        Parameters
        ----------
        *elements : Multimodal
            Elements of the content in order.

        Returns
        -------
        Self
            A new content instance. If only a single existing
            ``MultimodalContent`` is provided, it is returned as-is.
        """
        if not elements:
            return cls.empty

        if len(elements) == 1 and isinstance(elements[0], MultimodalContent):
            return cast(Self, elements[0])

        return cls(parts=tuple(_parts_from_elements(elements)))

    type: Literal["content"] = "content"
    parts: Sequence[MultimodalContentPart]

    def texts(self) -> Sequence[TextContent]:
        """
        Return only textual parts.

        Returns
        -------
        Sequence[TextContent]
            All parts that are instances of ``TextContent`` in order.
        """
        return tuple(part for part in self.parts if isinstance(part, TextContent))

    @property
    def contains_resources(self) -> bool:
        """
        Indicate whether any resource parts are present.

        Returns
        -------
        bool
            ``True`` when there is at least one ``ResourceReference`` or
            ``ResourceContent`` part.
        """
        return any(isinstance(part, ResourceReference | ResourceContent) for part in self.parts)

    def resources(
        self,
        *,
        mime_type: MimeType | None = None,
    ) -> Sequence[ResourceReference | ResourceContent]:
        """
        Return resource parts, optionally filtered by MIME type.

        Parameters
        ----------
        mime_type : MimeType | None, optional
            When provided, only resources whose ``mime_type`` equals this value
            are returned.

        Returns
        -------
        Sequence[ResourceReference | ResourceContent]
            Resource parts in order, optionally filtered by type.
        """
        if mime_type is None:
            return tuple(
                part for part in self.parts if isinstance(part, ResourceReference | ResourceContent)
            )

        else:
            return tuple(
                part
                for part in self.parts
                if isinstance(part, ResourceReference | ResourceContent)
                and part.mime_type == mime_type
            )

    def without_resources(self) -> Self:
        """
        Create a copy with all resource parts removed.

        Returns
        -------
        Self
            A content instance containing only non-resource parts.
        """
        return self.__class__(
            parts=tuple(
                part
                for part in self.parts
                if not isinstance(part, ResourceReference | ResourceContent)
            ),
        )

    def images(self) -> Sequence[ResourceReference | ResourceContent]:
        """
        Return image resources (``mime_type`` starting with ``image``).

        Returns
        -------
        Sequence[ResourceReference | ResourceContent]
            Image resources in order.
        """
        return tuple(
            part
            for part in self.parts
            if isinstance(part, ResourceReference | ResourceContent)
            and part.mime_type is not None
            and part.mime_type.startswith("image")
        )

    def audio(self) -> Sequence[ResourceReference | ResourceContent]:
        """
        Return audio resources (``mime_type`` starting with ``audio``).

        Returns
        -------
        Sequence[ResourceReference | ResourceContent]
            Audio resources in order.
        """
        return tuple(
            part
            for part in self.parts
            if isinstance(part, ResourceReference | ResourceContent)
            and part.mime_type is not None
            and part.mime_type.startswith("audio")
        )

    def video(self) -> Sequence[ResourceReference | ResourceContent]:
        """
        Return video resources (``mime_type`` starting with ``video``).

        Returns
        -------
        Sequence[ResourceReference | ResourceContent]
            Video resources in order.
        """
        return tuple(
            part
            for part in self.parts
            if isinstance(part, ResourceReference | ResourceContent)
            and part.mime_type is not None
            and part.mime_type.startswith("video")
        )

    @property
    def contains_artifacts(self) -> bool:
        """
        Indicate whether any artifact parts are present.

        Returns
        -------
        bool
            ``True`` when there is at least one ``ArtifactContent`` part.
        """
        return any(isinstance(part, ArtifactContent) for part in self.parts)

    @overload
    def artifacts(
        self,
        /,
        *,
        category: str | None = None,
    ) -> Sequence[ArtifactContent[DataModel]]: ...

    @overload
    def artifacts[Artifact: DataModel](
        self,
        model: builtins.type[Artifact],
        /,
        *,
        category: str | None = None,
    ) -> Sequence[ArtifactContent[Artifact]]: ...

    def artifacts[Artifact: DataModel](
        self,
        model: builtins.type[Artifact] | None = None,
        /,
        *,
        category: str | None = None,
    ) -> Sequence[ArtifactContent[Artifact]] | Sequence[ArtifactContent[DataModel]]:
        """
        Return artifact parts, optionally filtered by model type and category.

        Parameters
        ----------
        model : type[Artifact] | None, optional
            When provided, only artifacts whose wrapped ``artifact`` is an
            instance of this type are returned.
        category : str | None, optional
            When provided, only artifacts whose ``category`` equals this value
            are returned.

        Returns
        -------
        Sequence[ArtifactContent[Artifact]] | Sequence[ArtifactContent[DataModel]]
            Artifact parts in order, narrowed by the provided filters.
        """
        if model is None:
            if category is None:
                return tuple(part for part in self.parts if isinstance(part, ArtifactContent))

            else:
                return tuple(
                    part
                    for part in self.parts
                    if isinstance(part, ArtifactContent) and part.category == category
                )

        elif category is None:
            return tuple(
                part
                for part in self.parts
                if isinstance(part, ArtifactContent) and isinstance(part.artifact, model)
            )

        else:
            return tuple(
                part
                for part in self.parts
                if isinstance(part, ArtifactContent)
                and part.category == category
                and isinstance(part.artifact, model)
            )

    def without_artifacts(self) -> Self:
        """
        Create a copy with all artifacts removed.

        Returns
        -------
        Self
            A content instance containing only non-artifact parts.
        """
        return self.__class__(
            parts=tuple(part for part in self.parts if not isinstance(part, ArtifactContent)),
        )

    def to_str(self) -> str:
        """
        Produce a string representation of the content sequence.

        Returns
        -------
        str
            Concatenation of ``to_str()`` across all parts in order.
        """
        return "".join(part.to_str() for part in self.parts)

    def appending(
        self,
        *parts: "Self | MultimodalTag | MultimodalContentPart | str",
    ) -> Self:
        """
        Return a new content instance with parts appended.

        Parameters
        ----------
        *parts : Self | MultimodalTag | MultimodalContentPart | str
            Additional parts to append.

        Returns
        -------
        Self
            A new content instance containing existing and appended parts.
        """
        if not parts:
            return self

        return self.__class__(parts=tuple(_parts_from_elements((*self.parts, *parts))))

    def matching_meta(
        self,
        **values: BasicValue,
    ) -> Self:
        """
        Select parts whose metadata exactly matches the provided values.

        Parameters
        ----------
        **values : BasicValue
            Key-value pairs to match. The entire ``Meta`` object of a part must
            equal the provided mapping for it to be included.

        Returns
        -------
        Self
            A content instance containing only parts with equal metadata.
        """
        if not values or not self.parts:
            return self

        meta: Meta = Meta.of(values)

        return self.__class__(parts=tuple(part for part in self.parts if part.meta == meta))

    def split_by_meta(
        self,
        *,
        key: str,
    ) -> Sequence[Self]:
        """
        Split content into groups whenever the value for a meta key changes.

        Parameters
        ----------
        key : str
            Metadata key to track while scanning parts.

        Returns
        -------
        Sequence[Self]
            A tuple of content groups, each preserving original order and
            containing consecutive parts that share the same value for ``key``.
        """
        if not self.parts:
            return ()

        result: list[Self] = []
        current_group: list[MultimodalContentPart] = []
        current_value: BasicValue | Missing = MISSING

        for part in self.parts:
            # Get metadata value for this part
            part_value: BasicValue | Missing = part.meta.get(key, MISSING)
            if part_value is MISSING:
                continue  # skip elements without explicit value

            # If this is the first part or value changed, start new group
            if current_value != part_value:
                # Save current group if it has parts
                if current_group:
                    result.append(
                        self.__class__(
                            parts=tuple(current_group),
                        )
                    )

                # Start new group
                current_group = [part]
                current_value = part_value

            else:  # Same value, add to current group
                current_group.append(part)

        # Add final group if it has parts
        if current_group:
            result.append(
                self.__class__(
                    parts=tuple(current_group),
                )
            )

        return tuple(result)

    def tag(
        self,
        name: str | None = None,
        /,
    ) -> "MultimodalTag | None":
        """
        Return the first tag whose name matches ``name``.

        Parameters
        ----------
        name : str | None, optional
            Tag name to search for. When ``None``, the first parsed tag is
            returned regardless of its name.

        Returns
        -------
        MultimodalTag | None
            The matching tag, or ``None`` when no tag satisfies the filter.
        """
        tags = _collect_tags(self, name)
        return tags[0] if tags else None

    def tags(
        self,
        name: str | None = None,
        /,
    ) -> "Sequence[MultimodalTag]":
        """
        Return all tags whose name matches ``name``.

        Parameters
        ----------
        name : str | None, optional
            Tag name to search for. When ``None``, all parsed tags are
            returned.

        Returns
        -------
        Sequence[MultimodalTag]
            Parsed tags in order of appearance that satisfy the filter.
        """
        return _collect_tags(self, name)

    def replacing_tag(  # noqa: C901, PLR0912, PLR0915
        self,
        name: str,
        /,
        replacement: "Multimodal",
        *,
        strip_tags: bool = False,
        exhaustive: bool = False,
    ) -> Self:
        """
        Replace occurrences of a tag with provided content.

        Parameters
        ----------
        name : str
            Tag name to replace.
        replacement : Multimodal
            Content to insert in place of the tag.
        strip_tags : bool, optional
            When True, remove the tag markers and keep only inner content.
        exhaustive : bool, optional
            When True, replace all occurrences; otherwise, replace a single
            occurrence.

        Returns
        -------
        Self
            A content instance with the replacement applied.
        """
        if not name:
            return self

        replacement_parts: tuple[MultimodalContentPart, ...] = tuple(
            _parts_from_elements((replacement,))
        )

        tokens = _tokenize_content(self.parts)
        result_parts: list[MultimodalContentPart] = []
        stack: list[_ActiveContext] = []
        replaced_count = 0

        for index, token in enumerate(tokens):
            current_parts = _current_parts(stack, result_parts)

            if isinstance(token, _TextToken):
                if token.text:
                    current_parts.append(TextContent(text=token.text, meta=token.meta))

            elif isinstance(token, ResourceReference | ResourceContent | ArtifactContent):
                current_parts.append(token)

            elif isinstance(token, _OpenToken):
                stack.append(
                    _ActiveContext(
                        name=token.name,
                        attrs=token.attrs,
                        opening_meta=token.meta,
                        opening_raw=token.raw,
                        start_index=index,
                        parts=[],
                    )
                )

            elif isinstance(token, _SelfClosingToken):
                if token.name == name and (exhaustive or replaced_count == 0):
                    if not strip_tags:
                        current_parts.append(
                            TextContent(
                                text=f"<{token.name}{_tag_attributes(token.attrs)}>",
                                meta=token.meta,
                            )
                        )

                    current_parts.extend(replacement_parts)

                    if not strip_tags:
                        current_parts.append(TextContent(text=f"</{token.name}>", meta=token.meta))

                    replaced_count += 1

                else:
                    current_parts.append(TextContent(text=token.raw, meta=token.meta))

            elif isinstance(token, _CloseToken):
                if not stack:
                    current_parts.append(TextContent(text=token.raw, meta=token.meta))
                    continue

                match_index = _find_matching_context(stack, token.name)
                if match_index is None:
                    current_parts.append(TextContent(text=token.raw, meta=token.meta))
                    continue

                while len(stack) - 1 > match_index:
                    unmatched = stack.pop()
                    parent_parts = _current_parts(stack, result_parts)
                    parent_parts.append(
                        TextContent(text=unmatched.opening_raw, meta=unmatched.opening_meta)
                    )
                    parent_parts.extend(unmatched.parts)

                context = stack.pop()
                target_parts = _current_parts(stack, result_parts)

                if context.name == name and (exhaustive or replaced_count == 0):
                    if not strip_tags:
                        target_parts.append(
                            TextContent(text=context.opening_raw, meta=context.opening_meta)
                        )

                    target_parts.extend(replacement_parts)

                    if not strip_tags:
                        target_parts.append(TextContent(text=token.raw, meta=token.meta))

                    replaced_count += 1

                else:
                    target_parts.append(
                        TextContent(text=context.opening_raw, meta=context.opening_meta)
                    )
                    target_parts.extend(context.parts)
                    target_parts.append(TextContent(text=token.raw, meta=token.meta))

            else:  # pragma: no cover - future-proofing for new token types
                raise TypeError(f"Unsupported token: {type(token)!r}")

        while stack:
            context = stack.pop()
            target_parts = _current_parts(stack, result_parts)
            target_parts.append(TextContent(text=context.opening_raw, meta=context.opening_meta))
            target_parts.extend(context.parts)

        return self.__class__.of(*result_parts)

    def __bool__(self) -> bool:
        """
        Indicate whether the content contains at least one truthy part.

        Returns
        -------
        bool
            ``True`` when ``parts`` is non-empty and any part is truthy.
        """
        return bool(self.parts) and any(self.parts)


MultimodalContent.empty = MultimodalContent(parts=())


@final
class MultimodalTag(DataModel):
    """
    Lightweight tag that wraps multimodal content with attributes.

    ``MultimodalTag`` provides a minimal, XML-like tag representation useful
    for marking content segments with attributes in string projections while
    still supporting conversion back into a ``MultimodalContent`` sequence.

    Attributes
    ----------
    name : str
        Tag name used for string projection (e.g., ``<name ...>``).
    content : MultimodalContent
        The multimodal content enclosed by this tag instance.
    meta : Meta
        Tag attributes expressed as structured metadata. Values are escaped
        appropriately in string rendering.
    """

    @classmethod
    def of(
        cls,
        *elements: Self | MultimodalContentPart | str,
        name: str,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Construct a ``MultimodalTag`` from one or more elements.

        Parameters
        ----------
        *elements : Self | MultimodalContentPart | str
            Content elements to include in order.
        name : str
            The tag name.
        meta : Meta | MetaValues | None, optional
            Optional tag attributes expressed as ``Meta`` or a mapping. When
            ``None``, an empty metadata object is used.

        Returns
        -------
        Self
            A new tag wrapping a normalized ``MultimodalContent``.
        """
        return cls(
            name=name,
            content=MultimodalContent.of(*elements),
            meta=Meta.of(meta),
        )

    name: str
    content: MultimodalContent
    meta: Meta

    def to_str(self) -> str:
        """
        Produce a string representation of the tag and its content.

        Returns
        -------
        str
            A self-closing form when ``content`` is empty, otherwise an
            opening/closing tag with the content rendered between them.
        """
        if not self.content.parts:
            return f"<{self.name}{_tag_attributes(self.meta)}/>"

        return f"<{self.name}{_tag_attributes(self.meta)}>{self.content.to_str()}</{self.name}>"

    def parts(self) -> Sequence[MultimodalContentPart]:
        if not self.content.parts:
            return (TextContent(text=f"<{self.name}{_tag_attributes(self.meta)}/>"),)

        return (
            TextContent(text=f"<{self.name}{_tag_attributes(self.meta)}>"),
            *self.content.parts,
            TextContent(text=f"</{self.name}>"),
        )

    def to_multimodal(self) -> MultimodalContent:
        """
        Convert the tag to an equivalent ``MultimodalContent`` sequence.

        The result contains three parts when the tag is non-empty: the
        opening tag text, the wrapped content, and the closing tag text. For an
        empty tag, a single self-closing tag text is produced.

        Returns
        -------
        MultimodalContent
            A normalized multimodal content sequence representing this tag.
        """
        if not self.content.parts:
            return MultimodalContent.of(
                TextContent(text=f"<{self.name}{_tag_attributes(self.meta)}/>")
            )

        return MultimodalContent.of(
            TextContent(text=f"<{self.name}{_tag_attributes(self.meta)}>"),
            self.content,
            TextContent(text=f"</{self.name}>"),
        )


Multimodal = (
    MultimodalContent
    | MultimodalTag
    | TextContent
    | ResourceReference
    | ResourceContent
    | ArtifactContent
    | str
)


type Token = "_TextToken | _OpenToken | _CloseToken | _SelfClosingToken | MultimodalContentPart"


@dataclass(frozen=True)
class _TextToken:
    text: str
    meta: Meta


@dataclass(frozen=True)
class _OpenToken:
    name: str
    attrs: Meta
    meta: Meta
    raw: str


@dataclass(frozen=True)
class _CloseToken:
    name: str
    meta: Meta
    raw: str


@dataclass(frozen=True)
class _SelfClosingToken:
    name: str
    attrs: Meta
    meta: Meta
    raw: str


@dataclass
class _ActiveContext:
    name: str
    attrs: Meta
    opening_meta: Meta
    opening_raw: str
    start_index: int
    parts: list[MultimodalContentPart]


def _collect_tags(  # noqa: C901, PLR0912
    content: MultimodalContent,
    name: str | None,
) -> tuple[MultimodalTag, ...]:
    tokens = _tokenize_content(content.parts)
    stack: list[_ActiveContext] = []
    top_level: list[MultimodalContentPart] = []
    collected: list[tuple[int, MultimodalTag]] = []

    for index, token in enumerate(tokens):
        current_parts = _current_parts(stack, top_level)

        if isinstance(token, _TextToken):
            if token.text:
                current_parts.append(TextContent(text=token.text, meta=token.meta))

        elif isinstance(token, ResourceReference | ResourceContent | ArtifactContent):
            current_parts.append(token)

        elif isinstance(token, _OpenToken):
            stack.append(
                _ActiveContext(
                    name=token.name,
                    attrs=token.attrs,
                    opening_meta=token.meta,
                    opening_raw=token.raw,
                    start_index=index,
                    parts=[],
                )
            )

        elif isinstance(token, _SelfClosingToken):
            tag = MultimodalTag(
                name=token.name,
                content=MultimodalContent.empty,
                meta=token.attrs,
            )
            if name is None or tag.name == name:
                collected.append((index, tag))

            current_parts.append(TextContent(text=token.raw, meta=token.meta))

        elif isinstance(token, _CloseToken):
            if not stack:
                current_parts.append(TextContent(text=token.raw, meta=token.meta))
                continue

            match_index = _find_matching_context(stack, token.name)
            if match_index is None:
                current_parts.append(TextContent(text=token.raw, meta=token.meta))
                continue

            while len(stack) - 1 > match_index:
                unmatched = stack.pop()
                parent_parts = _current_parts(stack, top_level)
                parent_parts.append(
                    TextContent(text=unmatched.opening_raw, meta=unmatched.opening_meta)
                )
                parent_parts.extend(unmatched.parts)

            context = stack.pop()
            tag_content = (
                MultimodalContent.empty
                if not context.parts
                else MultimodalContent.of(*context.parts)
            )
            tag = MultimodalTag(name=context.name, content=tag_content, meta=context.attrs)

            if name is None or tag.name == name:
                collected.append((context.start_index, tag))

            parent_parts = _current_parts(stack, top_level)
            parent_parts.append(TextContent(text=context.opening_raw, meta=context.opening_meta))
            parent_parts.extend(context.parts)
            parent_parts.append(TextContent(text=token.raw, meta=token.meta))

        else:  # pragma: no cover - future-proofing for new token types
            raise TypeError(f"Unsupported token: {type(token)!r}")

    while stack:
        context = stack.pop()
        target_parts = _current_parts(stack, top_level)
        target_parts.append(TextContent(text=context.opening_raw, meta=context.opening_meta))
        target_parts.extend(context.parts)

    collected.sort(key=lambda item: item[0])
    return tuple(tag for _, tag in collected)


def _tokenize_content(
    parts: Sequence[MultimodalContentPart],
) -> list[Token]:
    tokens: list[Token] = []
    pending_text: str | None = None
    pending_meta: Meta | None = None

    def flush_pending() -> None:
        nonlocal pending_text, pending_meta
        if pending_text is None or pending_meta is None or not pending_text:
            pending_text = None
            pending_meta = None
            return

        tokens.extend(_tokenize_text(pending_text, pending_meta))
        pending_text = None
        pending_meta = None

    for part in parts:
        if isinstance(part, TextContent):
            if pending_meta is None:
                pending_text = part.text
                pending_meta = part.meta

            elif pending_meta == part.meta:
                if pending_text is None:
                    pending_text = part.text
                else:
                    pending_text = f"{pending_text}{part.text}"

            else:
                flush_pending()
                pending_text = part.text
                pending_meta = part.meta

        else:
            flush_pending()
            tokens.append(part)

    flush_pending()
    return tokens


def _tokenize_text(  # noqa: PLR0912
    text: str,
    meta: Meta,
) -> list[Token]:
    tokens: list[Token] = []
    i = 0

    while i < len(text):
        lt = text.find("<", i)
        if lt == -1:
            if i < len(text):
                tokens.append(_TextToken(text=text[i:], meta=meta))
            break

        if lt > i:
            tokens.append(_TextToken(text=text[i:lt], meta=meta))

        if lt == len(text) - 1:
            tokens.append(_TextToken(text="<", meta=meta))
            break

        if text[lt + 1] == "/":
            closing = _parse_closing_tag_token(text, lt, meta)
            if closing is None:
                tokens.append(_TextToken(text="<", meta=meta))
                i = lt + 1
            else:
                token, end_pos = closing
                tokens.append(token)
                i = end_pos

        else:
            match = _match_opening_tag(text, lt, filter_name=None)
            if match is None:
                tokens.append(_TextToken(text="<", meta=meta))
                i = lt + 1
            else:
                tag_name, attrs_meta, is_self_closing, end_pos = match
                raw = text[lt:end_pos]
                if is_self_closing:
                    tokens.append(
                        _SelfClosingToken(
                            name=tag_name,
                            attrs=attrs_meta,
                            meta=meta,
                            raw=raw,
                        )
                    )
                else:
                    tokens.append(
                        _OpenToken(
                            name=tag_name,
                            attrs=attrs_meta,
                            meta=meta,
                            raw=raw,
                        )
                    )

                i = end_pos

    return tokens


def _parse_closing_tag_token(
    text: str,
    pos: int,
    meta: Meta,
) -> tuple[_CloseToken, int] | None:
    if pos + 2 > len(text) or text[pos : pos + 2] != "</":
        return None

    name, name_end = _extract_tag_name(text, pos + 2)
    if not name:
        return None

    end_pos = name_end
    while end_pos < len(text) and text[end_pos].isspace():
        end_pos += 1

    if end_pos >= len(text) or text[end_pos] != ">":
        return None

    closing_end = end_pos + 1
    return _CloseToken(name=name, meta=meta, raw=text[pos:closing_end]), closing_end


def _current_parts(
    stack: Sequence[_ActiveContext],
    top_level: list[MultimodalContentPart],
) -> list[MultimodalContentPart]:
    return stack[-1].parts if stack else top_level


def _find_matching_context(
    stack: Sequence[_ActiveContext],
    name: str,
) -> int | None:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index].name == name:
            return index
    return None


def _parse_attributes_from_string(
    attr_text: str,
    /,
) -> Meta:
    """Parse tag attributes from text using strict char-by-char parsing."""
    if not attr_text.strip():
        return META_EMPTY

    attrs: dict[str, str] = {}
    i = 0

    while i < len(attr_text):
        i = _skip_whitespace(attr_text, i)
        if i >= len(attr_text):
            break

        # Parse single attribute
        name, value, i = _parse_single_attribute(attr_text, i)
        attrs[name] = value

    return Meta.of(attrs)


def _parse_single_attribute(
    attr_text: str,
    pos: int,
    /,
) -> tuple[str, str, int]:
    """Parse a single attribute and return (name, value, new_pos)."""
    # Parse attribute name
    name, pos = _parse_strict_attr_name(attr_text, pos)

    # Expect equals
    pos = _expect_equals_sign(attr_text, pos)

    # Parse quoted value
    value, pos = _parse_quoted_attr_value(attr_text, pos)

    return name, value, pos


def _parse_strict_attr_name(
    attr_text: str,
    pos: int,
    /,
) -> tuple[str, int]:
    """Parse attribute name with strict validation."""
    if pos >= len(attr_text) or not (attr_text[pos].isalpha() or attr_text[pos] in "_"):
        raise ValueError("Invalid attribute name")

    start = pos
    while pos < len(attr_text) and (attr_text[pos].isalnum() or attr_text[pos] in "_-"):
        pos += 1

    if pos == start:
        raise ValueError("Empty attribute name")

    return attr_text[start:pos], pos


def _expect_equals_sign(
    attr_text: str,
    pos: int,
    /,
) -> int:
    """Expect equals sign and return position after it."""
    if pos >= len(attr_text):
        raise ValueError("Attribute without value")

    if attr_text[pos] != "=":
        if not attr_text[pos].isspace():
            raise ValueError("Invalid character after attribute name")

        pos = _skip_whitespace(attr_text, pos)
        if pos >= len(attr_text) or attr_text[pos] != "=":
            raise ValueError("Attribute without value")

    return pos + 1


def _parse_quoted_attr_value(
    attr_text: str,
    pos: int,
    /,
) -> tuple[str, int]:
    """Parse quoted attribute value."""
    if pos >= len(attr_text) or attr_text[pos] not in "\"'":
        raise ValueError("Attribute value must be quoted")

    quote_char = attr_text[pos]
    pos += 1  # Skip opening quote
    value = ""

    while pos < len(attr_text):
        if attr_text[pos] == quote_char:
            return value, pos + 1
        elif attr_text[pos] == "\\":
            value, pos = _handle_escape_sequence(attr_text, pos, value)
        elif attr_text[pos] == "\n":
            raise ValueError("Literal newline in attribute value")
        else:
            value += attr_text[pos]
            pos += 1

    raise ValueError("Unterminated attribute value")


def _handle_escape_sequence(
    attr_text: str,
    pos: int,
    value: str,
    /,
) -> tuple[str, int]:
    """Handle escape sequence in attribute value."""
    if pos + 1 >= len(attr_text):
        raise ValueError("Unterminated escape sequence")

    pos += 1
    next_char = attr_text[pos]

    if next_char == "n":
        value += "\n"
    elif next_char == "t":
        value += "\t"
    elif next_char == "r":
        value += "\r"
    elif next_char in '"\\':
        value += next_char
    else:
        value += "\\" + next_char

    return value, pos + 1


def _skip_whitespace(
    text: str,
    pos: int,
    /,
) -> int:
    """Skip whitespace characters and return new position."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _tag_attributes(
    meta: Meta,
    /,
) -> str:
    if not meta:
        return ""

    return " " + " ".join(
        f'{key}="{_formatted_tag_attribute_value(value)}"' for key, value in meta.items()
    )


_ATTRIBUTE_ESCAPE_MAP: Mapping[str, str] = {
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
    '"': '\\"',
    "\\": "\\\\",
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
}


def _escape_attribute_string(value: str) -> str:
    return "".join(_ATTRIBUTE_ESCAPE_MAP.get(char, char) for char in value)


def _formatted_tag_attribute_value(
    value: BasicValue,
    /,
) -> str:
    if isinstance(value, str):
        return _escape_attribute_string(value)

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return ",".join(_formatted_tag_attribute_value(item) for item in value)

    if isinstance(value, Mapping):
        raise ValueError("Mappings are not allowed as tag attributes")

    return str(value)


def _parse_attrs(
    attrs_text: str,
    /,
) -> Meta:
    # Use the new char-by-char parser
    return _parse_attributes_from_string(attrs_text)


def _match_opening_tag(
    text: str,
    pos: int,
    *,
    filter_name: str | None,
) -> tuple[str, Meta, bool, int] | None:
    """Parse opening tag at position and return (tag_name, attributes, is_self_closing, end_pos)."""
    if pos >= len(text) or text[pos] != "<":
        return None

    i = pos + 1
    if i >= len(text) or text[i] == "/":
        return None  # Closing tag or incomplete

    # Parse tag name
    tag_name, i = _extract_tag_name(text, i)
    if not tag_name or (filter_name is not None and tag_name != filter_name):
        return None

    # Parse rest of tag
    attrs_text, is_self_closing, end_pos = _parse_tag_remainder(text, i)
    if end_pos is None:
        return None

    try:
        attrs_meta = _parse_attrs(attrs_text)
    except Exception:
        return None

    return (tag_name, attrs_meta, is_self_closing, end_pos)


def _extract_tag_name(
    text: str,
    pos: int,
    /,
) -> tuple[str, int]:
    """Extract tag name starting at position."""
    if pos >= len(text) or not (text[pos].isalpha() or text[pos] in "_:"):
        return "", pos

    start = pos
    while pos < len(text) and (text[pos].isalnum() or text[pos] in "_:.-"):
        pos += 1

    return text[start:pos], pos


def _is_quote_escaped(
    text: str,
    pos: int,
    /,
) -> bool:
    """Check if quote at pos is escaped by counting preceding backslashes."""
    if pos == 0 or text[pos - 1] != "\\":
        return False
    # Count consecutive backslashes
    backslash_count = 0
    check_pos = pos - 1
    while check_pos >= 0 and text[check_pos] == "\\":
        backslash_count += 1
        check_pos -= 1
    # If odd number of backslashes, the quote is escaped
    return backslash_count % 2 == 1


def _handle_quoted_char(
    text: str,
    pos: int,
    in_quote: str | None,
    /,
) -> tuple[str | None, int]:
    """Handle character when inside a quoted string."""
    char = text[pos]
    if char == in_quote and not _is_quote_escaped(text, pos):
        return None, pos + 1  # End quote
    return in_quote, pos + 1  # Continue in quote


def _parse_tag_remainder(
    text: str,
    pos: int,
    /,
) -> tuple[str, bool, int | None]:
    """Parse attributes and closing of tag, return (attrs_text, is_self_closing, end_pos)."""
    # Skip whitespace
    while pos < len(text) and text[pos].isspace():
        pos += 1

    attrs_start = pos
    is_self_closing = False
    in_quote = None  # Track if we're inside a quoted string

    # Find end of tag, being aware of quoted attribute values
    while pos < len(text):
        char = text[pos]

        if in_quote:
            in_quote, pos = _handle_quoted_char(text, pos, in_quote)
        elif char in "\"'":
            # Start of a quoted string
            in_quote = char
            pos += 1
        elif char == ">":
            # End of tag (only if not in quotes)
            break
        elif char == "/" and _is_self_closing_at(text, pos):
            is_self_closing = True
            # Skip to the >
            while pos < len(text) and text[pos] != ">":
                pos += 1
            break
        else:
            pos += 1

    if pos >= len(text):
        return "", False, None

    # Extract attributes text
    attrs_text = text[attrs_start:pos].rstrip("/ \t\n\r")
    return attrs_text, is_self_closing, pos + 1


def _is_self_closing_at(
    text: str,
    pos: int,
    /,
) -> bool:
    """Check if the '/' at pos indicates a self-closing tag."""
    next_pos = pos + 1
    while next_pos < len(text) and text[next_pos].isspace():
        next_pos += 1
    return next_pos < len(text) and text[next_pos] == ">"


def _parts_from_elements(  # noqa: C901, PLR0912, PLR0915
    elements: Collection[Multimodal],
    /,
) -> Generator[MultimodalContentPart]:
    last_text: TextContent | None = None
    for element in elements:
        if isinstance(element, str):
            if last_text is None:
                last_text = TextContent(
                    text=element,
                    meta=META_EMPTY,
                )

            elif not last_text.meta:
                last_text = TextContent(
                    text=last_text.text + element,
                    meta=META_EMPTY,
                )

            else:
                yield last_text
                last_text = TextContent(
                    text=element,
                    meta=META_EMPTY,
                )

        elif isinstance(element, TextContent):
            if last_text is None:
                last_text = element

            elif last_text.meta == element.meta:
                # Preserve metadata when merging text parts with identical meta
                last_text = TextContent(
                    text=last_text.text + element.text,
                    meta=last_text.meta,
                )

            else:
                yield last_text
                last_text = element

        elif isinstance(element, MultimodalContent):
            for part in element.parts:
                if isinstance(part, TextContent):
                    if last_text is None:
                        last_text = part

                    elif last_text.meta == part.meta:
                        # Preserve metadata when merging text parts with identical meta
                        last_text = TextContent(
                            text=last_text.text + part.text,
                            meta=last_text.meta,
                        )

                    else:
                        yield last_text
                        last_text = part

                else:
                    assert isinstance(  # nosec: B101
                        part,
                        ResourceReference | ResourceContent | ArtifactContent,
                    )

                    if last_text is not None:
                        yield last_text
                        last_text = None

                    yield part

        elif isinstance(element, MultimodalTag):
            for part in element.parts():
                if isinstance(part, TextContent):
                    if last_text is None:
                        last_text = part

                    elif last_text.meta == part.meta:
                        # Preserve metadata when merging text parts with identical meta
                        last_text = TextContent(
                            text=last_text.text + part.text,
                            meta=last_text.meta,
                        )

                    else:
                        yield last_text
                        last_text = part

                else:
                    assert isinstance(  # nosec: B101
                        part,
                        ResourceReference | ResourceContent | ArtifactContent,
                    )

                    if last_text is not None:
                        yield last_text
                        last_text = None

                    yield part

        else:
            assert isinstance(  # nosec: B101
                element,
                ResourceReference | ResourceContent | ArtifactContent,
            )

            if last_text is not None:
                yield last_text
                last_text = None

            yield element

    if last_text is not None:
        yield last_text
