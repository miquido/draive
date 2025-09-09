from collections.abc import Collection, Generator, Sequence
from typing import ClassVar, Self, cast, final, overload

from haiway import MISSING, Meta, MetaValue, MetaValues, Missing
from haiway.utils.metadata import META_EMPTY

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
type MultimodalConvertible = "MultimodalTag | MultimodalContentPart | str"


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
        Construct a normalized ``MultimodalContent`` sequence.

        Parameters
        ----------
        element : MultimodalConvertible | Self | None
            The first element to include. When ``None``, returns
            ``MultimodalContent.empty``.
        *elements : MultimodalConvertible | Self
            Additional elements to append in order.

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
        model: type[Artifact],
        /,
        *,
        category: str | None = None,
    ) -> Sequence[ArtifactContent[Artifact]]: ...

    def artifacts[Artifact: DataModel](
        self,
        model: type[Artifact] | None = None,
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
        *parts: Self | MultimodalConvertible,
    ) -> Self:
        """
        Return a new content instance with parts appended.

        Parameters
        ----------
        *parts : Self | MultimodalConvertible
            Additional parts to append. Accepts any value supported by
            ``MultimodalContent.of(...)`` including another content instance.

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
        **values: MetaValue,
    ) -> Self:
        """
        Select parts whose metadata exactly matches the provided values.

        Parameters
        ----------
        **values : MetaValue
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
        current_value: MetaValue | Missing = MISSING

        for part in self.parts:
            # Get metadata value for this part
            part_value: MetaValue | None = part.meta.get(key, default=None)

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
        """Return the first tag matching the name using custom parser."""
        return next(_parse_tags_from_content(content=self, name=name), None)

    def tags(
        self,
        name: str | None = None,
        /,
    ) -> "Sequence[MultimodalTag]":
        """Return all tags matching the name using custom parser."""
        return tuple(_parse_tags_from_content(content=self, name=name))

    def replacing_tag(  # noqa: C901, PLR0915
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
        if not name or replacement is None:
            return self

        replacement_parts: tuple[MultimodalContentPart, ...] = tuple(
            _parts_from_elements((replacement,))
        )

        result: list[MultimodalContentPart] = []

        class _Current:
            def __init__(self, attrs: Meta, opening_meta: Meta) -> None:
                self.attrs = attrs
                self.opening_meta = opening_meta
                self.parts: list[MultimodalContentPart] = []

        class _NestingTracker:
            def __init__(self) -> None:
                self.level = 0  # Track nesting level across parts

        def _replace_in_text_part(  # noqa: C901, PLR0912, PLR0915
            text: str,
            meta: Meta,
            current: _Current | None,
            replaced_count: int,
            nesting: _NestingTracker,
        ) -> tuple[list[MultimodalContentPart], _Current | None, int]:
            # Fast check: if no opening tags exist for our name and we're not in current tag,
            # return original text unchanged
            if current is None and f"<{name}" not in text:
                return [TextContent(text=text, meta=meta)], None, replaced_count

            # Additional check: if we have potential tags but none are complete within this part,
            # and we're not continuing a tag from a previous part, return unchanged
            if current is None:
                # Check if any complete tags exist in this text part that are not nested
                # in other tags
                has_complete_tag = False
                temp_i = 0
                temp_nesting_level = nesting.level  # Start with cross-part nesting level

                while temp_i < len(text):
                    temp_lt = text.find("<", temp_i)
                    if temp_lt == -1:
                        break

                    # Check if it's a closing tag
                    if temp_lt + 1 < len(text) and text[temp_lt + 1] == "/":
                        # Find the tag name in closing tag
                        close_end = text.find(">", temp_lt)
                        if close_end != -1:
                            temp_nesting_level = max(0, temp_nesting_level - 1)
                        temp_i = temp_lt + 1
                        continue

                    # Check if it's an opening tag
                    match_result = _match_opening_tag(
                        text, temp_lt, filter_name=None
                    )  # Check any tag
                    if match_result:
                        tag_name, _attrs, is_self, end_pos = match_result

                        if is_self:
                            # Self-closing tag doesn't affect nesting, but check if it's our target
                            if tag_name == name and temp_nesting_level == 0:
                                has_complete_tag = True
                                break
                        else:
                            # Opening tag - check if it's our target at nesting level 0
                            if tag_name == name and temp_nesting_level == 0:
                                # Check if closing tag exists
                                closing_pos = _closing_search(text, end_pos, name=name)
                                if closing_pos:
                                    has_complete_tag = True
                                    break
                            temp_nesting_level += 1
                        temp_i = end_pos
                    else:
                        temp_i = temp_lt + 1

                if not has_complete_tag:
                    # Update nesting level for next parts
                    temp_i = 0
                    while temp_i < len(text):
                        temp_lt = text.find("<", temp_i)
                        if temp_lt == -1:
                            break
                        if temp_lt + 1 < len(text) and text[temp_lt + 1] == "/":
                            close_end = text.find(">", temp_lt)
                            if close_end != -1:
                                nesting.level = max(0, nesting.level - 1)
                            temp_i = temp_lt + 1
                        else:
                            match_result = _match_opening_tag(text, temp_lt, filter_name=None)
                            if match_result:
                                _, _, is_self, end_pos = match_result
                                if not is_self:
                                    nesting.level += 1
                                temp_i = end_pos
                            else:
                                temp_i = temp_lt + 1
                    return [TextContent(text=text, meta=meta)], None, replaced_count

            out: list[MultimodalContentPart] = []
            i = 0
            while i < len(text):
                if current is None:
                    lt = text.find("<", i)
                    if lt == -1:
                        if i < len(text):
                            out.append(TextContent(text=text[i:], meta=meta))
                        break

                    if lt > i:
                        out.append(TextContent(text=text[i:lt], meta=meta))

                    # Check if it's a closing tag first
                    if lt + 1 < len(text) and text[lt + 1] == "/":
                        close_end = text.find(">", lt)
                        if close_end != -1:
                            nesting.level = max(0, nesting.level - 1)
                            out.append(TextContent(text=text[lt : close_end + 1], meta=meta))
                            i = close_end + 1
                        else:
                            out.append(TextContent(text=text[lt : lt + 1], meta=meta))
                            i = lt + 1
                        continue

                    match_result = _match_opening_tag(text, lt, filter_name=name)
                    if not match_result:
                        # Check if it's any other opening tag to track nesting
                        other_match = _match_opening_tag(text, lt, filter_name=None)
                        if other_match:
                            _, _, is_self_other, end_pos_other = other_match
                            if not is_self_other:
                                nesting.level += 1
                            out.append(TextContent(text=text[lt:end_pos_other], meta=meta))
                            i = end_pos_other
                        else:
                            out.append(TextContent(text=text[lt : lt + 1], meta=meta))
                            i = lt + 1
                        continue

                    _tag_name, attrs_meta, is_self, _end_pos = match_result

                    # Only process target tags when not nested inside other tags
                    if nesting.level > 0:
                        # We're nested inside another tag, treat as plain text
                        out.append(TextContent(text=text[lt:_end_pos], meta=meta))
                        if not is_self:
                            nesting.level += 1
                        i = _end_pos
                        continue

                    if is_self:
                        if exhaustive or replaced_count == 0:
                            if not strip_tags:
                                out.append(
                                    TextContent(
                                        text=f"<{name}{_tag_attributes(attrs_meta)}>",
                                        meta=meta,
                                    )
                                )
                            out.extend(replacement_parts)
                            if not strip_tags:
                                out.append(TextContent(text=f"</{name}>", meta=meta))
                            replaced_count += 1
                        else:
                            out.append(TextContent(text=text[lt:_end_pos], meta=meta))

                        i = _end_pos
                        continue

                    # Check if closing tag exists in this text part
                    closing_pos = _closing_search(text, _end_pos, name=name)
                    if not closing_pos:
                        # No closing tag found in this part, treat as plain text
                        out.append(TextContent(text=text[lt : lt + 1], meta=meta))
                        i = lt + 1
                        continue

                    current = _Current(attrs_meta, meta)
                    nesting.level += 1  # We're now inside our target tag
                    i = _end_pos

                else:
                    cm = _closing_search(text, i, name=name)
                    if not cm:
                        if i < len(text):
                            current.parts.append(TextContent(text=text[i:], meta=meta))
                        break

                    # Find the start position of the closing tag
                    close_tag_start = text.rfind(f"</{name}", 0, cm)
                    if close_tag_start > i:
                        current.parts.append(TextContent(text=text[i:close_tag_start], meta=meta))

                    if exhaustive or replaced_count == 0:
                        if not strip_tags:
                            out.append(
                                TextContent(
                                    text=f"<{name}{_tag_attributes(current.attrs)}>",
                                    meta=current.opening_meta,
                                )
                            )
                        out.extend(replacement_parts)
                        if not strip_tags:
                            out.append(TextContent(text=f"</{name}>", meta=meta))
                        replaced_count += 1
                    else:
                        out.append(
                            TextContent(
                                text=f"<{name}{_tag_attributes(current.attrs)}>",
                                meta=current.opening_meta,
                            )
                        )
                        out.extend(current.parts)
                        out.append(TextContent(text=f"</{name}>", meta=meta))

                    current = None
                    nesting.level = max(0, nesting.level - 1)  # Exiting our target tag
                    i = cm
                    continue

            return out, current, replaced_count

        current: _Current | None = None
        replaced_count = 0
        nesting = _NestingTracker()
        for part in self.parts:
            if isinstance(part, TextContent):
                out, current, replaced_count = _replace_in_text_part(
                    part.text,
                    part.meta,
                    current,
                    replaced_count,
                    nesting,
                )
                result.extend(out)

            elif current is None:
                result.append(part)

            else:
                current.parts.append(part)

        if current is not None:
            result.append(
                TextContent(
                    text=f"<{name}{_tag_attributes(current.attrs)}>",
                    meta=current.opening_meta,
                )
            )
            result.extend(current.parts)

        return self.__class__.of(*result)

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
        The content wrapped by the tag. It is the content wrapped by the tag.
    meta : Meta
        Tag attributes expressed as structured metadata. Values are escaped
        appropriately in string rendering.
    """

    @classmethod
    def of(
        cls,
        element: Self | MultimodalContentPart | str,
        *elements: Self | MultimodalContentPart | str,
        name: str,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Construct a ``MultimodalTag`` from one or more elements.

        Parameters
        ----------
        element : Self | MultimodalContentPart | str
            First element to include. May be a tag, a content part, or a raw
            string which will be converted to ``TextContent``.
        *elements : Self | MultimodalContentPart | str
            Additional elements to include in order.
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
            content=MultimodalContent.of(element, *elements),
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

    def _parts(self) -> Sequence[MultimodalContentPart]:
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
        return MultimodalContent.of(*self._parts())


Multimodal = (
    MultimodalContent
    | MultimodalTag
    | TextContent
    | ResourceReference
    | ResourceContent
    | ArtifactContent
    | str
)


def _parse_attributes_from_string(attr_text: str) -> Meta:
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


def _parse_single_attribute(attr_text: str, pos: int) -> tuple[str, str, int]:
    """Parse a single attribute and return (name, value, new_pos)."""
    # Parse attribute name
    name, pos = _parse_strict_attr_name(attr_text, pos)

    # Expect equals
    pos = _expect_equals_sign(attr_text, pos)

    # Parse quoted value
    value, pos = _parse_quoted_attr_value(attr_text, pos)

    return name, value, pos


def _parse_strict_attr_name(attr_text: str, pos: int) -> tuple[str, int]:
    """Parse attribute name with strict validation."""
    if pos >= len(attr_text) or not (attr_text[pos].isalpha() or attr_text[pos] in "_"):
        raise ValueError("Invalid attribute name")

    start = pos
    while pos < len(attr_text) and (attr_text[pos].isalnum() or attr_text[pos] in "_-"):
        pos += 1

    if pos == start:
        raise ValueError("Empty attribute name")

    return attr_text[start:pos], pos


def _expect_equals_sign(attr_text: str, pos: int) -> int:
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


def _parse_quoted_attr_value(attr_text: str, pos: int) -> tuple[str, int]:
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


def _handle_escape_sequence(attr_text: str, pos: int, value: str) -> tuple[str, int]:
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


def _skip_whitespace(text: str, pos: int) -> int:
    """Skip whitespace characters and return new position."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _find_closing_tag(text: str, start_pos: int, tag_name: str) -> int | None:
    """Find the end position of a closing tag using char-by-char search."""
    search_pattern = f"</{tag_name}"
    pos = start_pos

    while True:
        pos = text.find(search_pattern, pos)
        if pos == -1:
            return None

        # Check if it's followed by whitespace or >
        end_pos = pos + len(search_pattern)
        if end_pos >= len(text):
            return None

        # Closing tags should not have attributes - only whitespace then >
        if text[end_pos] == ">":
            return end_pos + 1
        elif text[end_pos] in " \t\n\r":
            # Skip whitespace and ensure we only find >
            while end_pos < len(text) and text[end_pos] in " \t\n\r":
                end_pos += 1
            if end_pos < len(text) and text[end_pos] == ">":
                return end_pos + 1
            else:
                # Malformed closing tag (has attributes or other content)
                pos += 1
                continue
        else:
            # Not a valid closing tag
            pos += 1
            continue


def _tag_attributes(
    meta: Meta,
    /,
) -> str:
    if not meta:
        return ""

    return " " + " ".join(
        f'{key}="{_formatted_tag_attribute_value(value)}"' for key, value in meta.items()
    )


def _formatted_tag_attribute_value(
    value: MetaValue,
) -> str:
    match value:
        case str() as string:
            accumulator: list[str] = []
            for char in string:
                match char:
                    case "\n":
                        accumulator.append("\\n")

                    case "\t":
                        accumulator.append("\\t")

                    case "\r":
                        accumulator.append("\\r")

                    case '"':
                        accumulator.append('\\"')

                    case "\\":
                        accumulator.append("\\\\")

                    # TODO: handle <, >, &

                    case other:
                        accumulator.append(other)

            return "".join(accumulator)

        case [*values]:
            return ",".join(_formatted_tag_attribute_value(v) for v in values)

        case {}:
            raise ValueError("Mappings are not allowed as tag attributes")

        case other:
            return str(other)


def _unescape_tag_attribute_value(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _parse_attrs(attrs_text: str) -> Meta:
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


def _extract_tag_name(text: str, pos: int) -> tuple[str, int]:
    """Extract tag name starting at position."""
    if pos >= len(text) or not (text[pos].isalpha() or text[pos] in "_:"):
        return "", pos

    start = pos
    while pos < len(text) and (text[pos].isalnum() or text[pos] in "_:.-"):
        pos += 1

    return text[start:pos], pos


def _is_quote_escaped(text: str, pos: int) -> bool:
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


def _handle_quoted_char(text: str, pos: int, in_quote: str | None) -> tuple[str | None, int]:
    """Handle character when inside a quoted string."""
    char = text[pos]
    if char == in_quote and not _is_quote_escaped(text, pos):
        return None, pos + 1  # End quote
    return in_quote, pos + 1  # Continue in quote


def _parse_tag_remainder(text: str, pos: int) -> tuple[str, bool, int | None]:
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


def _is_self_closing_at(text: str, pos: int) -> bool:
    """Check if the '/' at pos indicates a self-closing tag."""
    next_pos = pos + 1
    while next_pos < len(text) and text[next_pos].isspace():
        next_pos += 1
    return next_pos < len(text) and text[next_pos] == ">"


def _closing_search(text: str, pos: int, *, name: str) -> int | None:
    return _find_closing_tag(text, pos, name)


def _parse_tags_from_content(  # noqa: C901
    *,
    content: "MultimodalContent",
    name: str | None,
) -> Generator[MultimodalTag]:
    """
    Stream-parse tags from mixed ``MultimodalContent`` preserving multimodal parts.

    The parser detects XML-like tags only within text parts. When a matching
    opening tag is found, all subsequent parts (including resources and
    artifacts) are collected until a matching closing tag appears. Self-closing
    tags immediately yield an empty ``content``.
    """

    class _Current:
        def __init__(self, name: str, attrs: Meta) -> None:
            self.name = name
            self.attrs = attrs
            self.parts: list[MultimodalContentPart] = []

    def _parse_text_part(
        text: str,
        meta: Meta,
        current: _Current | None,
    ) -> tuple[list[MultimodalTag], _Current | None]:
        tags: list[MultimodalTag] = []
        i = 0
        while i < len(text):
            if current is None:
                lt = text.find("<", i)
                if lt == -1:
                    break

                match_result = _match_opening_tag(text, lt, filter_name=name)
                if match_result is None:
                    i = lt + 1
                    continue

                tag_name, attrs_meta, is_self, end_pos = match_result
                if is_self:
                    tags.append(
                        MultimodalTag(
                            name=tag_name,
                            content=MultimodalContent.empty,
                            meta=attrs_meta,
                        )
                    )
                    i = end_pos
                    continue

                # Start capturing content within this tag and continue scanning
                current = _Current(tag_name, attrs_meta)
                i = end_pos
                continue

            else:
                cm = _closing_search(text, i, name=current.name)
                if not cm:
                    current.parts.append(TextContent(text=text[i:], meta=meta))
                    break

                # Find the start position of the closing tag
                close_tag_start = text.rfind(f"</{current.name}", 0, cm)
                if close_tag_start > i:
                    current.parts.append(TextContent(text=text[i:close_tag_start], meta=meta))

                inner_parts = tuple(current.parts)
                tags.append(
                    MultimodalTag(
                        name=current.name,
                        content=(
                            MultimodalContent.empty
                            if not inner_parts
                            else MultimodalContent.of(*inner_parts)
                        ),
                        meta=current.attrs,
                    )
                )
                current = None
                i = cm
                continue

        return tags, current

    current: _Current | None = None
    for part in content.parts:
        if isinstance(part, TextContent):
            tags, current = _parse_text_part(part.text, part.meta, current)
            yield from tags

        elif current is not None:
            current.parts.append(part)

    # if unclosed tag at the end -> ignore (no yield)


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
            for part in element._parts():
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
