from haiway import State

from draive.prompts.types import PromptFetching, PromptListing

__all__ = [
    "PromptRepository",
]


class PromptRepository(State):
    list: PromptListing
    fetch: PromptFetching
