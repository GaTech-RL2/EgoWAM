"""Unit tests for the legacy `robot_name` -> live-schema filter remap.

The `app.episodes` table dropped `robot_name` and gained `rig_name`, so data configs written against
the old schema have to be translated before the SQL select. The subtle case is the human side:
`aria_bimanual` is not an embodiment value -- Aria episodes are `human_bimanual` on the `aria_gen1`
rig, and without the rig clause the filter also matches thousands of Mecka episodes.

The pure-python tests here need no credentials. The live-DB test is skipped unless
`EGOWAM_TEST_LIVE_DB=1`, since it queries the real episode table.
"""

import os

import pytest

from egowam.rldb.zarr.zarr_dataset_multi import remap_legacy_filters

LIVE_COLUMNS = [
    "episode_hash",
    "operator",
    "lab",
    "num_frames",
    "task",
    "task_description",
    "scene",
    "objects",
    "is_deleted",
    "embodiment",
    "rig_name",
    "zarr_processed_path",
]


def test_robot_embodiment_keeps_its_value():
    out = remap_legacy_filters(
        {"task": "EgoWAM_cup_on_saucer", "robot_name": "eva_bimanual"}, LIVE_COLUMNS
    )
    assert out == {"task": "EgoWAM_cup_on_saucer", "embodiment": "eva_bimanual"}


def test_aria_maps_to_human_bimanual_and_pins_the_rig():
    """Without rig_name the same filter also matches Mecka/Scale human episodes."""
    out = remap_legacy_filters(
        {"task": "cup_on_saucer", "robot_name": "aria_bimanual"}, LIVE_COLUMNS
    )
    assert out == {
        "task": "cup_on_saucer",
        "embodiment": "human_bimanual",
        "rig_name": "aria_gen1",
    }


def test_noop_when_the_table_still_has_robot_name():
    """Stays correct if the schema is ever rolled back."""
    filters = {"task": "cup_on_saucer", "robot_name": "aria_bimanual"}
    assert remap_legacy_filters(filters, LIVE_COLUMNS + ["robot_name"]) == filters


def test_noop_without_a_robot_name_filter():
    filters = {"task": "cup_on_saucer", "embodiment": "human_bimanual"}
    assert remap_legacy_filters(filters, LIVE_COLUMNS) == filters


def test_caller_supplied_values_win():
    """An explicit embodiment/rig_name is not overwritten by the mapping."""
    out = remap_legacy_filters(
        {"robot_name": "aria_bimanual", "rig_name": "mecka"}, LIVE_COLUMNS
    )
    assert out["rig_name"] == "mecka"
    assert out["embodiment"] == "human_bimanual"


def test_input_is_not_mutated():
    filters = {"task": "cup_on_saucer", "robot_name": "aria_bimanual"}
    remap_legacy_filters(filters, LIVE_COLUMNS)
    assert filters == {"task": "cup_on_saucer", "robot_name": "aria_bimanual"}


def test_missing_target_column_raises_with_context():
    with pytest.raises(ValueError, match="no 'rig_name' column"):
        remap_legacy_filters({"robot_name": "aria_bimanual"}, ["task", "embodiment"])


# --- live episode table -------------------------------------------------------------------------

# Robot splits were relabelled <task>_success[_1] -> EgoWAM_<task>; the episode sets are
# unchanged, so every count below is exactly what the old names returned.
#
# These are RESOLVER counts, not raw table counts. _get_filtered_paths also requires a
# non-null zarr_processed_path, so the table holds a few more rows than resolve: e.g.
# cup_on_saucer/aria is 554 in app.episodes but 552 resolvable. Compare against the
# resolver, not against SELECT COUNT(*).
EXPECTED_COUNTS = [
    (
        {
            "task": "EgoWAM_cup_on_saucer",
            "scene": "white_wall",
            "robot_name": "eva_bimanual",
        },
        363,
    ),
    ({"task": "cup_on_saucer", "robot_name": "aria_bimanual"}, 552),
    ({"task": "EgoWAM_bag_groceries", "robot_name": "eva_bimanual"}, 300),
    ({"task": "bag_groceries", "robot_name": "aria_bimanual"}, 143),
    ({"task": "EgoWAM_fold_clothes", "robot_name": "eva_bimanual"}, 300),
    ({"task": "fold_clothes", "robot_name": "aria_bimanual"}, 709),
]


@pytest.mark.skipif(
    os.environ.get("EGOWAM_TEST_LIVE_DB") != "1",
    reason="needs episode-table credentials; set EGOWAM_TEST_LIVE_DB=1 to run",
)
@pytest.mark.parametrize("filters,expected", EXPECTED_COUNTS)
def test_live_episode_counts(filters, expected):
    """The paper's six train splits resolve to the expected episode counts."""
    from egowam.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver

    resolved = S3EpisodeResolver._get_filtered_paths(
        filters={**filters, "is_deleted": False}
    )
    assert len(resolved) == expected
