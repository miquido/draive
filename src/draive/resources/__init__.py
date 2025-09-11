from draive.resources.state import ResourcesRepository
from draive.resources.template import ResourceAvailabilityCheck, ResourceTemplate, resource
from draive.resources.types import (
    MimeType,
    Resource,
    ResourceContent,
    ResourceCorrupted,
    ResourceDeleting,
    ResourceFetching,
    ResourceListFetching,
    ResourceMissing,
    ResourceReference,
    ResourceReferenceTemplate,
    ResourceUploading,
)

# Backwards-compatible/expected aliases
ResourceException = ResourceCorrupted
ResourceTemplateDeclaration = ResourceReferenceTemplate

__all__ = (
    "MimeType",
    "Resource",
    "ResourceAvailabilityCheck",
    "ResourceContent",
    "ResourceCorrupted",
    "ResourceDeleting",
    "ResourceFetching",
    "ResourceListFetching",
    "ResourceMissing",
    "ResourceReference",
    "ResourceReferenceTemplate",
    "ResourceTemplate",
    "ResourceUploading",
    "ResourcesRepository",
    "resource",
)
