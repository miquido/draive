from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, NoReturn, cast, final

from haiway import Meta, MetaValues, Paginated, Pagination, as_dict, cache, ctx

from draive.multimodal.templates.repository import TemplatesRepository
from draive.multimodal.templates.types import TemplateDeclaration
from draive.surreal.state import Surreal
from draive.surreal.types import SurrealObject

__all__ = ("SurrealTemplatesRepository",)


@final
class SurrealTemplatesRepository:
    @staticmethod
    def prepare(  # noqa: C901
        cache_limit: int = 32,
        cache_expiration: float = 600.0,
        meta: Meta | MetaValues | None = None,
    ) -> TemplatesRepository:
        """Return a SurrealDB-backed templates repository with caching.

        Parameters
        ----------
        cache_limit
            Maximum number of loaded template payloads cached concurrently.
        cache_expiration
            Lifetime in seconds for cached entries before reloading from SurrealDB.
        meta
            Repository metadata exposed through the returned repository state.

        Returns
        -------
        TemplatesRepository
            Repository facade operating on the ``templates`` SurrealDB table.
        """

        async def listing(
            pagination: Pagination | None,
            **extra: Any,
        ) -> Paginated[TemplateDeclaration]:
            ctx.log_info("Listing templates...")
            pagination = pagination or Pagination.of(limit=32)
            if pagination.limit <= 0:
                return Paginated[TemplateDeclaration].of(
                    (),
                    pagination=pagination.with_token(None),
                )

            results: tuple[SurrealObject, ...]
            match pagination.token:
                case str() as token:
                    results = tuple(
                        await _fetch_template_rows(
                            after_identifier=token,
                            limit=pagination.limit + 1,
                        )
                    )

                case None:
                    results = tuple(
                        await _fetch_template_rows(
                            after_identifier=None,
                            limit=pagination.limit + 1,
                        )
                    )

                case _:
                    raise ValueError("Invalid SurrealDB templates pagination token")

            page_results: tuple[SurrealObject, ...] = results[: pagination.limit]
            ctx.log_info(f"...{len(page_results)} results found!")
            next_token: str | None = None
            if len(results) > pagination.limit:
                next_token = f"{cast(str, page_results[-1]['identifier'])}"

            return Paginated[TemplateDeclaration].of(
                tuple(
                    TemplateDeclaration(
                        identifier=cast(str, result["identifier"]),
                        description=cast(str | None, result.get("description")),
                        variables=cast(
                            Mapping[str, str],
                            result.get("variables") or {},
                        ),
                        meta=Meta.of(cast(MetaValues | None, result.get("meta"))),
                    )
                    for result in page_results
                ),
                pagination=pagination.with_token(next_token),
            )

        @cache(
            limit=cache_limit,
            expiration=cache_expiration,
        )
        async def load(
            identifier: str,
            /,
        ) -> str | None:
            results: Sequence[SurrealObject] = await Surreal.execute(
                """
                SELECT
                    id,
                    updated,
                    content

                FROM
                    templates

                WHERE
                    identifier = $identifier

                ORDER BY
                    updated DESC,
                    id DESC

                LIMIT 1;
                """,
                identifier=identifier,
            )

            if not results:
                return None

            content: str | None = cast(str | None, results[0].get("content"))
            if content is None:
                return None

            return content

        async def loading(
            identifier: str,
            meta: Meta,
            **extra: Any,
        ) -> str | None:
            return await load(identifier)

        async def defining(
            identifier: str,
            description: str | None,
            content: str,
            variables: Mapping[str, str],
            meta: Meta,
            **extra: Any,
        ) -> None:
            updated: datetime = datetime.now(UTC)
            await Surreal.execute(
                """
                CREATE templates CONTENT {
                    identifier: $identifier,
                    updated: $updated,
                    description: $description,
                    content: $content,
                    variables: $variables,
                    meta: $meta
                };
                """,
                identifier=identifier,
                updated=updated,
                description=description,
                content=content,
                variables=as_dict(variables),
                meta=as_dict(meta),
            )
            await load.clear_cache()

        return TemplatesRepository(
            listing=listing,
            loading=loading,
            defining=defining,
            meta=Meta.of(meta if meta is not None else {"source": "surrealdb"}),
        )

    __slots__ = ()

    def __init__(self) -> NoReturn:
        raise RuntimeError("SurrealTemplatesRepository instantiation is forbidden")


async def _fetch_template_rows(
    *,
    after_identifier: str | None,
    limit: int,
) -> Sequence[SurrealObject]:
    return await Surreal.execute(
        """
        SELECT
            id,
            identifier,
            updated,
            description,
            variables,
            meta

        FROM
            templates

        WHERE
            identifier IN (
                SELECT VALUE identifier
                FROM templates
                WHERE
                    $after_identifier = NONE
                    OR identifier > $after_identifier
                GROUP BY
                    identifier
                ORDER BY
                    identifier ASC
                LIMIT $limit
            )
            AND id IN (
                SELECT VALUE id
                FROM templates
                WHERE
                    identifier = $parent.identifier
                ORDER BY
                    updated DESC,
                    id DESC
                LIMIT 1
            )

        ORDER BY
            identifier ASC,
            updated DESC,
            id DESC
        LIMIT $limit;
        """,
        after_identifier=after_identifier,
        limit=limit,
    )
