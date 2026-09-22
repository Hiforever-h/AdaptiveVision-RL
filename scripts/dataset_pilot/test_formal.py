import unittest

from .prepare_formal import choose, choose_use_tool_stratified, stable_score


def candidate(group, pixel, dhash, use_tool=False):
    return {"group_key": group, "pixel_sha256": pixel, "dhash": dhash,
            "source_info": {"use_tool": use_tool}}


class FormalSelectionTests(unittest.TestCase):
    def test_stable_score_is_deterministic(self):
        self.assertEqual(stable_score(7, "train", "doc:1"),
                         stable_score(7, "train", "doc:1"))
        self.assertNotEqual(stable_score(7, "train", "doc:1"),
                            stable_score(8, "train", "doc:1"))

    def test_choose_blocks_groups_and_exact_or_near_duplicates(self):
        groups = sorted((f"doc:{index}" for index in range(4)),
                        key=lambda group: stable_score(9, "train", group))
        rows = [candidate(groups[0], "p0", "ffffffffffffffff"),
                candidate(groups[1], "used", "ffffffffffffffff"),
                candidate(groups[2], "p2", "0000000000000001"),
                candidate(groups[3], "p3", "f0f0f0f0f0f0f0f0")]
        selected, rejected = choose(
            rows, 1, 9, "train", set(), {"used"},
            ["0000000000000000"], {groups[0]})
        self.assertEqual([row["group_key"] for row in selected], [groups[3]])
        self.assertGreaterEqual(rejected["group_overlap"], 1)
        self.assertGreaterEqual(rejected["exact_duplicate_image"], 1)
        self.assertGreaterEqual(rejected["near_duplicate_image"], 1)

    def test_choose_use_tool_stratified_meets_exact_targets(self):
        rows = [candidate("doc:0", "p0", "0000000000000000", False),
                candidate("doc:1", "p1", "ffffffffffffffff", False),
                candidate("doc:2", "p2", "aaaaaaaaaaaaaaaa", True),
                candidate("doc:3", "p3", "5555555555555555", True)]
        selected, _ = choose_use_tool_stratified(
            rows, {False: 2, True: 2}, 11, "train", set(), set(), [])
        self.assertEqual(len(selected), 4)
        self.assertEqual(sum(row["source_info"]["use_tool"] is True for row in selected), 2)
        self.assertTrue(all(row["split"] == "train" for row in selected))

    def test_choose_use_tool_stratified_fails_when_a_stratum_is_short(self):
        rows = [candidate("doc:1", "p1", "0000000000000000", False),
                candidate("doc:2", "p2", "ffffffffffffffff", True)]
        with self.assertRaises(RuntimeError):
            choose_use_tool_stratified(
                rows, {False: 2, True: 1}, 11, "train", set(), set(), [])


if __name__ == "__main__":
    unittest.main()
