import json
from typing import Any

import pytest
from haiway import Missing, ValidationError

from draive.claude import (
    ClaudeCodeApiKey,
    ClaudeCodeConfig,
    ClaudeCodeLoadAdditionalClaudeMd,
    ClaudeCodeNoClaudeMd,
    ClaudeCodeOAuthToken,
    ClaudeCodeProjectClaudeMd,
    ClaudeCodeSkipAdditionalClaudeMd,
    ClaudeCodeUnavailable,
)
from draive.claude.launch import (
    build_launch,
    claude_arguments,
    claude_environment,
    claude_settings,
    verify_claude_md,
)
from draive.claude.workspace import ClaudeCodeWorkspace

WORKSPACE = ClaudeCodeWorkspace(root="/private", directory="/private/work")
VARIABLE = "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD"


def _add_directories(arguments: Any) -> list[str]:
    return [arguments[index + 1] for index, item in enumerate(arguments) if item == "--add-dir"]


def _settings(config: ClaudeCodeConfig, **arguments: Any) -> Any:
    return json.loads(claude_settings(config, **arguments))


def _environment(config: ClaudeCodeConfig) -> Any:
    return claude_environment(config, workspace=WORKSPACE, parent={})


@pytest.mark.parametrize(
    "credential", [ClaudeCodeApiKey(value="k"), ClaudeCodeOAuthToken(value="t")]
)
def test_no_claude_md_is_the_default_and_changes_nothing(credential: Any) -> None:
    config = ClaudeCodeConfig(directory="/project", additional_directories=("/data",))
    explicit = config.updating(claude_md=ClaudeCodeNoClaudeMd())

    assert config.claude_md == ClaudeCodeNoClaudeMd()
    launch = build_launch(
        config, binary="/bin/claude", prompt=[], workspace=WORKSPACE, credential=credential
    )
    assert launch.arguments == claude_arguments(explicit, directory=WORKSPACE.directory)
    assert _add_directories(launch.arguments) == ["/data"]
    assert VARIABLE not in launch.environment
    assert _settings(config) == {
        "enabledPlugins": _settings(ClaudeCodeConfig())["enabledPlugins"],
        "autoMemoryEnabled": False,
    }
    assert claude_settings(config) == claude_settings(ClaudeCodeConfig())


def test_project_claude_md_adds_the_working_directory_and_enables_the_variable() -> None:
    config = ClaudeCodeConfig(directory="/private/work", claude_md=ClaudeCodeProjectClaudeMd())

    launch = build_launch(
        config, binary="/bin/claude", prompt=[], workspace=WORKSPACE, credential=Missing
    )

    assert _add_directories(launch.arguments) == ["/private/work"]
    assert launch.environment[VARIABLE] == "1"
    assert _environment(config)[VARIABLE] == "1"
    # nothing is excluded without additional directories
    assert "claudeMdExcludes" not in _settings(config, directory="/private/work")
    arguments = claude_arguments(config, directory="/project")
    # the permissions are the ones of the working directory only
    assert arguments[arguments.index("--allowedTools") + 1] == "Read(./**),Grep(./**),Glob(./**)"


def test_the_working_directory_of_the_configuration_is_used_without_a_given_one() -> None:
    config = ClaudeCodeConfig(directory="/project", claude_md=ClaudeCodeProjectClaudeMd())

    assert _add_directories(claude_arguments(config)) == ["/project"]


def test_the_working_directory_is_required_to_load_its_claude_md() -> None:
    config = ClaudeCodeConfig(claude_md=ClaudeCodeProjectClaudeMd())

    with pytest.raises(ValueError, match="working directory"):
        claude_arguments(config)


def test_the_working_directory_is_not_added_twice() -> None:
    config = ClaudeCodeConfig(
        additional_directories=("/data", "/project/"),
        claude_md=ClaudeCodeProjectClaudeMd(),
    )

    arguments = claude_arguments(config, directory="/project")

    assert _add_directories(arguments) == ["/data", "/project/"]
    # the working directory is not an additional one when it comes to excluding
    assert _settings(config, directory="/project")["claudeMdExcludes"] == [
        "/data/CLAUDE.md",
        "/data/.claude/CLAUDE.md",
    ]


def test_permissions_of_additional_directories_are_not_changed_by_the_working_one() -> None:
    base = ClaudeCodeConfig(tools=("Read",), additional_directories=("/data",))
    config = base.updating(claude_md=ClaudeCodeProjectClaudeMd())

    def allowed(value: ClaudeCodeConfig) -> str:
        arguments = claude_arguments(value, directory="/project")
        return arguments[arguments.index("--allowedTools") + 1]

    assert allowed(config) == allowed(base) == "Read(./**),Read(//data/**)"


@pytest.mark.parametrize(
    ("additional", "excluded"),
    [
        (
            ClaudeCodeSkipAdditionalClaudeMd(),
            [
                "/data/CLAUDE.md",
                "/data/.claude/CLAUDE.md",
                "/more/CLAUDE.md",
                "/more/.claude/CLAUDE.md",
            ],
        ),
        (ClaudeCodeLoadAdditionalClaudeMd(), None),
    ],
)
def test_the_claude_md_of_additional_directories_is_excluded_unless_wanted(
    additional: Any,
    excluded: list[str] | None,
) -> None:
    config = ClaudeCodeConfig(
        directory="/project",
        additional_directories=("/data", "/more", "/data"),
        claude_md=ClaudeCodeProjectClaudeMd(additional_directories=additional),
    )

    settings = _settings(config)

    assert settings.get("claudeMdExcludes") == excluded
    assert _environment(config)[VARIABLE] == "1"


def test_the_default_for_additional_directories_is_to_skip_them() -> None:
    assert ClaudeCodeProjectClaudeMd().additional_directories == ClaudeCodeSkipAdditionalClaudeMd()


def test_configured_excludes_are_added_to_the_derived_ones_and_other_settings_win() -> None:
    config = ClaudeCodeConfig(
        directory="/project",
        additional_directories=("/data",),
        claude_md=ClaudeCodeProjectClaudeMd(),
        settings={
            "claudeMdExcludes": ["/data/CLAUDE.md", "**/secret/CLAUDE.md"],
            "autoMemoryEnabled": True,
        },
    )

    settings = _settings(config)

    assert settings["claudeMdExcludes"] == [
        "/data/CLAUDE.md",
        "/data/.claude/CLAUDE.md",
        "**/secret/CLAUDE.md",
    ]
    assert settings["autoMemoryEnabled"] is True


def test_configured_excludes_stay_when_nothing_is_derived() -> None:
    config = ClaudeCodeConfig(settings={"claudeMdExcludes": ["/x/CLAUDE.md"]})

    assert _settings(config)["claudeMdExcludes"] == ["/x/CLAUDE.md"]


def test_the_working_directory_is_required_to_run_with_its_claude_md() -> None:
    config = ClaudeCodeConfig(claude_md=ClaudeCodeProjectClaudeMd())

    with pytest.raises(ClaudeCodeUnavailable, match="directory of the configuration"):
        verify_claude_md(config)

    with pytest.raises(ClaudeCodeUnavailable, match="directory of the configuration"):
        build_launch(config, binary="/bin/claude", prompt=[], workspace=WORKSPACE)

    # also what is completed after the creation, nothing of it is checked before a run
    completed = config.updating(directory="/project")
    verify_claude_md(completed)
    assert _add_directories(claude_arguments(completed)) == ["/project"]


@pytest.mark.parametrize("directory", ["/data/a*b", "/data/a?b", "/data/[x]", "/data/{a,b}"])
def test_additional_directories_with_glob_characters_are_refused_when_skipped(
    directory: str,
) -> None:
    config = ClaudeCodeConfig(
        directory="/project",
        additional_directories=(directory,),
        claude_md=ClaudeCodeProjectClaudeMd(),
    )

    with pytest.raises(ClaudeCodeUnavailable, match="glob characters"):
        build_launch(config, binary="/bin/claude", prompt=[], workspace=WORKSPACE)

    # nothing is excluded when they are loaded, and without the loading they are not special
    for allowed in (
        config.updating(
            claude_md=ClaudeCodeProjectClaudeMd(
                additional_directories=ClaudeCodeLoadAdditionalClaudeMd()
            )
        ),
        config.updating(claude_md=ClaudeCodeNoClaudeMd()),
    ):
        verify_claude_md(allowed)


def test_directories_are_compared_as_given() -> None:
    config = ClaudeCodeConfig(
        additional_directories=("/link-to-project",),
        claude_md=ClaudeCodeProjectClaudeMd(),
    )

    # symlinks are not resolved - the real path has to be passed
    assert _add_directories(claude_arguments(config, directory="/project")) == [
        "/link-to-project",
        "/project",
    ]


@pytest.mark.parametrize("value", ["/data/CLAUDE.md", {"a": "b"}, 1])
def test_configured_excludes_have_to_be_lists(value: Any) -> None:
    with pytest.raises(ValidationError):
        ClaudeCodeConfig(settings={"claudeMdExcludes": value})


@pytest.mark.parametrize(
    "config",
    [
        ClaudeCodeConfig(claude_md=ClaudeCodeProjectClaudeMd()),
        ClaudeCodeConfig(
            claude_md=ClaudeCodeProjectClaudeMd(
                additional_directories=ClaudeCodeLoadAdditionalClaudeMd()
            )
        ),
    ],
)
def test_claude_md_survives_serialization(config: ClaudeCodeConfig) -> None:
    assert ClaudeCodeConfig.from_mapping(config.to_mapping()) == config
