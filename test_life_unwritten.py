"""Regression tests for Life Unwritten state handling.

Covers: bond/mood sync after key conversations, transactional rollback on
failure, failure not polluting later actions, deterministic results that do
not depend on dict iteration order, single day advancement, and deep-copy
save/load snapshots.
"""

import copy
import json
import os
import tempfile
import unittest

import life_unwritten as lu


def fresh_game():
    game = lu.LifeUnwritten()
    game.state.player_name = "Tester"
    return game


def state_key(state):
    return state.to_dict()


# Table-driven: (character, option_index, bond_delta, mood_delta, expected_npc_mood)
INTERACTION_CASES = [
    ("Maya", 0, 15, 5, "warm"),
    ("Maya", 1, 8, 3, "warm"),
    ("Maya", 2, 5, 2, "warm"),
    ("David", 0, 20, 8, "warm"),
    ("David", 1, 12, 5, "hopeful"),
    ("David", 2, -5, -2, "guarded"),
    ("Sarah", 0, 18, 6, "hopeful"),
    ("Sarah", 1, 5, 1, "guarded"),
    ("Sarah", 2, 8, 3, "guarded"),
    ("Alex", 0, 10, -5, "guarded"),
    ("Alex", 1, 15, 5, "guarded"),
    ("Alex", 2, 8, 2, "guarded"),
]


class TestInteractionTable(unittest.TestCase):
    def test_interaction_outcomes_table(self):
        for name, idx, bond_delta, mood_delta, npc_mood in INTERACTION_CASES:
            with self.subTest(character=name, option=idx):
                game = fresh_game()
                char = game.state.characters[name]
                old_bond, old_mood = char.bond_level, game.state.mood
                option = game.get_interaction_scenarios(char)["options"][idx]

                outcome = game.apply_interaction(name, option)

                self.assertEqual(char.bond_level, old_bond + bond_delta)
                self.assertEqual(game.state.mood, old_mood + mood_delta)
                # Key-conversation sync: NPC mood follows the new bond level.
                self.assertEqual(char.current_mood, npc_mood)
                self.assertEqual(char.current_mood, lu.mood_for_bond(char.bond_level))
                self.assertEqual(char.last_interaction, option["text"])
                self.assertEqual(outcome["new_bond"], char.bond_level)
                self.assertEqual(outcome["new_mood"], game.state.mood)
                # Choice log recorded exactly once with synced values.
                self.assertEqual(len(game.state.choices_made), 1)
                entry = game.state.choices_made[0]
                self.assertEqual(entry["mood_at_time"], game.state.mood)
                self.assertIn(f"{old_bond} → {char.bond_level}", entry["impact"])


class TestSnapshotDeepCopy(unittest.TestCase):
    def test_snapshot_restore_roundtrip(self):
        game = fresh_game()
        game.apply_interaction("Maya", game.get_interaction_scenarios(
            game.state.characters["Maya"])["options"][0])
        snap = game.state.snapshot()

        # Mutate everything, then restore.
        game.state.mood = 1
        game.state.day = 6
        game.state.characters["Maya"].bond_level = 0
        game.state.characters["Maya"].current_mood = "estranged"
        game.state.choices_made.append({"day": 6, "choice": "x", "impact": "y",
                                        "mood_at_time": 1})
        game.state.restore(snap)

        self.assertEqual(state_key(game.state), state_key(snap))
        self.assertEqual(game.state.characters["Maya"].bond_level, 75)
        self.assertEqual(game.state.characters["Maya"].current_mood, "warm")
        self.assertEqual(len(game.state.choices_made), 1)

    def test_snapshot_is_deep_copy_not_alias(self):
        game = fresh_game()
        snap = game.state.snapshot()
        # Mutating the live state must not leak into the snapshot...
        game.state.characters["Sarah"].bond_level = 99
        game.state.choices_made.append({"day": 1, "choice": "c", "impact": "i",
                                        "mood_at_time": 50})
        self.assertEqual(snap.characters["Sarah"].bond_level, 30)
        self.assertEqual(len(snap.choices_made), 0)
        # ...and mutating the snapshot must not leak back either.
        snap.characters["Sarah"].bond_level = 5
        snap.choices_made.append({"day": 9, "choice": "z", "impact": "z",
                                  "mood_at_time": 0})
        self.assertEqual(game.state.characters["Sarah"].bond_level, 99)
        self.assertEqual(len(game.state.choices_made), 1)

    def test_restore_detaches_from_snapshot(self):
        game = fresh_game()
        snap = game.state.snapshot()
        game.state.restore(snap)
        # After restore, further mutation of the snapshot must not affect state.
        snap.characters["Alex"].bond_level = 100
        self.assertEqual(game.state.characters["Alex"].bond_level, 20)


class TestTransactionalRollback(unittest.TestCase):
    def test_failure_rolls_back_everything(self):
        game = fresh_game()
        before = state_key(game.state)
        option = game.get_interaction_scenarios(
            game.state.characters["Maya"])["options"][0]

        def boom(description, impact):
            raise RuntimeError("simulated mid-interaction failure")

        game.state.save_choice = boom  # fail after bond/mood were mutated
        with self.assertRaises(RuntimeError):
            game.apply_interaction("Maya", option)

        # Full rollback: bond, mood, NPC mood, last interaction, choice log.
        self.assertEqual(state_key(game.state), before)
        self.assertEqual(game.state.characters["Maya"].bond_level, 60)
        self.assertEqual(game.state.characters["Maya"].current_mood, "distant")
        self.assertEqual(game.state.mood, 50)
        self.assertEqual(game.state.choices_made, [])

    def test_failure_does_not_pollute_later_actions(self):
        game = fresh_game()
        option = game.get_interaction_scenarios(
            game.state.characters["David"])["options"][0]

        game.state.save_choice = lambda d, i: (_ for _ in ()).throw(ValueError("x"))
        with self.assertRaises(ValueError):
            game.apply_interaction("David", option)

        # A later, healthy interaction behaves exactly as on a fresh game.
        del game.state.save_choice
        clean = fresh_game()
        outcome = game.apply_interaction("David", option)
        clean_outcome = clean.apply_interaction(
            "David", clean.get_interaction_scenarios(
                clean.state.characters["David"])["options"][0])

        self.assertEqual(outcome, clean_outcome)
        self.assertEqual(state_key(game.state), state_key(clean.state))


class TestDeterminism(unittest.TestCase):
    def test_same_script_same_result(self):
        def script():
            game = fresh_game()
            for name, idx, *_ in INTERACTION_CASES[:6]:
                char = game.state.characters[name]
                game.apply_interaction(
                    name, game.get_interaction_scenarios(char)["options"][idx])
            game.advance_day()
            return state_key(game.state)

        self.assertEqual(script(), script())

    def test_result_independent_of_iteration_order(self):
        game_a = fresh_game()
        game_b = fresh_game()
        # Rebuild characters dict in a different insertion order.
        chars = game_b.state.characters
        game_b.state.characters = dict(reversed(list(chars.items())))

        for name, idx, *_ in INTERACTION_CASES:
            for game in (game_a, game_b):
                char = game.state.characters[name]
                game.apply_interaction(
                    name, game.get_interaction_scenarios(char)["options"][idx])

        self.assertEqual(state_key(game_a.state), state_key(game_b.state))

    def test_follow_up_is_deterministic(self):
        game = fresh_game()
        char = game.state.characters["Maya"]
        first = game.generate_follow_up_response(char, 15)
        second = game.generate_follow_up_response(char, 15)
        self.assertEqual(first, second)


class TestDayAdvance(unittest.TestCase):
    def test_day_advances_exactly_once(self):
        game = fresh_game()
        game.state.reflection_count = 2
        self.assertTrue(game.advance_day())
        self.assertEqual(game.state.day, 2)
        self.assertEqual(game.state.reflection_count, 0)

    def test_no_advance_after_game_over(self):
        game = fresh_game()
        game.state.game_over = True
        self.assertFalse(game.advance_day())
        self.assertEqual(game.state.day, 1)


class TestSaveLoad(unittest.TestCase):
    def test_save_load_roundtrip_preserves_state(self):
        game = fresh_game()
        for name, idx, *_ in INTERACTION_CASES[:4]:
            char = game.state.characters[name]
            game.apply_interaction(
                name, game.get_interaction_scenarios(char)["options"][idx])
        before = state_key(game.state)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "save.json")
            self.assertTrue(game.save_game(path))

            # Mutate live state, then load: no rollback of saved progress.
            game.state.characters["Maya"].bond_level = 3
            game.state.mood = 5
            game.state.day = 7
            self.assertTrue(game.load_game(path))

        self.assertEqual(state_key(game.state), before)
        self.assertEqual(game.state.characters["Maya"].bond_level, 88)
        self.assertEqual(game.state.characters["Maya"].current_mood, "devoted")

    def test_saved_file_is_deep_snapshot(self):
        game = fresh_game()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "save.json")
            game.save_game(path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Editing the file's in-memory copy must not touch live state.
            data["characters"]["Maya"]["bond_level"] = 1
            self.assertEqual(game.state.characters["Maya"].bond_level, 60)

    def test_failed_load_keeps_current_state(self):
        game = fresh_game()
        game.apply_interaction("Sarah", game.get_interaction_scenarios(
            game.state.characters["Sarah"])["options"][0])
        before = state_key(game.state)

        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "corrupt.json")
            with open(bad, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            self.assertFalse(game.load_game(bad))
            missing = os.path.join(tmp, "missing.json")
            self.assertFalse(game.load_game(missing))

        # Corrupt/missing saves must not roll back or dirty live state.
        self.assertEqual(state_key(game.state), before)

    def test_to_dict_sorted_and_deep_copied(self):
        game = fresh_game()
        data = game.state.to_dict()
        self.assertEqual(list(data["characters"].keys()),
                         sorted(data["characters"].keys()))
        data["characters"]["Maya"]["bond_level"] = 0
        data["choices_made"].append({"day": 1, "choice": "c", "impact": "i",
                                     "mood_at_time": 50})
        self.assertEqual(game.state.characters["Maya"].bond_level, 60)
        self.assertEqual(game.state.choices_made, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
