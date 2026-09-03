from collections.abc import Mapping, MutableMapping, Sequence
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
    """SurrealDB-backed templates repository factory.

    Exposes static helpers for schema migration and for creating a
    :class:`~draive.multimodal.templates.repository.TemplatesRepository`
    persisted in SurrealDB.

    Examples
    --------
    ```python
    from draive import ctx
    from draive.surreal import SurrealClient, SurrealTemplatesRepository

    async def bootstrap_templates() -> None:
        async with SurrealClient(url="ws://localhost:8000/rpc") as surreal:
            async with ctx.scope("templates", surreal):
                await SurrealTemplatesRepository.migrate()
                repository = SurrealTemplatesRepository.prepare()
    ```
    """

    @staticmethod
    async def migrate() -> None:
        """Define the database structures required by the templates repository.

        Defines the `templates` table and its supporting index when not already
        defined. A SurrealDB server refuses to read from a table which was never
        defined nor written to, therefore it has to be defined upfront.

        Returns
        -------
        None
            Completes when the schema definition statements finish.

        Raises
        ------
        SurrealException
            Raised when SurrealDB statement execution fails.
        """
        # one statement per call - a multi-statement query reports its errors
        # per statement, executing them separately keeps failures attributable
        await Surreal.execute("DEFINE TABLE IF NOT EXISTS templates SCHEMALESS TYPE NORMAL;")
        await Surreal.execute(
            "DEFINE INDEX IF NOT EXISTS templates_identifier_idx "
            "ON TABLE templates FIELDS identifier, updated;"
        )

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
    # Selecting the latest revision of each identifier within a single statement
    # would require a `$parent`-correlated subquery, which a SurrealDB server
    # evaluates unreliably - it silently dropped a varying subset of the rows on
    # each run (verified live). The page of identifiers and their revisions are
    # fetched separately instead, keeping the latest revision of each here.
    identifiers: Sequence[SurrealObject] = await Surreal.execute(
        """
        -- `SELECT VALUE identifier ... GROUP BY identifier` yields a single
        -- NONE instead of the distinct values, the grouping has to be nested
        SELECT VALUE identifier FROM (
            SELECT identifier
            FROM templates
            WHERE
                $after_identifier = NONE
                OR identifier > $after_identifier
            GROUP BY
                identifier
            ORDER BY
                identifier ASC
            LIMIT $limit
        );
        """,
        after_identifier=after_identifier,
        limit=limit,
    )

    if not identifiers:
        return ()

    revisions: Sequence[SurrealObject] = await Surreal.execute(
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
            identifier IN $identifiers

        ORDER BY
            identifier ASC,
            updated DESC,
            id DESC;
        """,
        identifiers=[cast(str, row["value"]) for row in identifiers],
    )

    latest: MutableMapping[str, SurrealObject] = {}
    for revision in revisions:
        # rows are ordered by identifier and revision, the first one wins
        latest.setdefault(cast(str, revision["identifier"]), revision)

    return tuple(latest.values())
