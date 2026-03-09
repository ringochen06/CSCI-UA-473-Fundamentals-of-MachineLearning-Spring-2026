import json
import os

import numpy as np
import streamlit as st
import torch

# Use path by script location so refresh always finds the same file
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_FILE = os.path.join(_ROOT, "lab_json", "lab1_game_save.json")

# Only save/load Lab 1 game keys (no other labs' session_state)
_LAB1_GAME_KEYS = frozenset({
    "player_name", "magic_number", "game_started", "level", "hp", "max_hp",
    "xp", "gold", "revival_count", "dice_count", "inventory", "logs",
    "in_shop", "dungeon_map", "answered_mcqs", "merchant_count",
    "merchant_dice_rolled", "last_roll", "shop_available",
    "ev_questions_solved", "prob_question_solved", "rng_offset",
    "level_complete", "shop_from_boss_lobby", "tensor_x",
})


def _is_lab1_game_key(key):
    if not isinstance(key, str):
        return False
    if key in _LAB1_GAME_KEYS:
        return True
    if key.startswith(("code_", "mcq_", "frq_", "saved_frq_", "boss_code_")):
        return True
    if key.startswith("boss_") and key.endswith("_solved"):
        return True
    return False


def serialize_value(val):
    """Recursive serialization for nested lists/dicts and special types."""
    if isinstance(val, torch.Tensor):
        return {
            "__tensor__": True,
            "data": val.tolist(),
            "dtype": str(val.dtype).replace("torch.", ""),
            "device": "cpu",  # Always save as CPU
        }
    elif isinstance(val, (set, tuple)):
        # Convert set/tuple to list, mark the type
        return {
            "__type__": type(val).__name__,
            "data": [serialize_value(item) for item in val],
        }
    elif isinstance(val, list):
        return [serialize_value(item) for item in val]
    elif isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    elif isinstance(val, (np.integer, np.floating)):
        return val.item()  # Convert numpy scalars to python scalars

    return val


def deserialize_value(val):
    """Recursive deserialization."""
    if isinstance(val, dict):
        if val.get("__tensor__"):
            dtype_str = val["dtype"]
            # specific safe attribute lookup
            try:
                dtype = getattr(torch, dtype_str)
            except AttributeError:
                dtype = torch.float32  # fallback

            # create tensor
            try:
                return torch.tensor(val["data"], dtype=dtype)
            except Exception as e:
                print(
                    f"Warning: Failed to create tensor with dtype {dtype_str}, falling back to default. Error: {e}"
                )
                return torch.tensor(val["data"])  # fallback without dtype

        if val.get("__type__") == "set":
            return set(deserialize_value(item) for item in val["data"])
        elif val.get("__type__") == "tuple":
            return tuple(deserialize_value(item) for item in val["data"])

        # Regular dict recursion
        return {k: deserialize_value(v) for k, v in val.items()}

    elif isinstance(val, list):
        return [deserialize_value(item) for item in val]

    return val


def _has_progress(state_dict):
    """True if state has level > 0 or any in-level answers (mcq/code/frq/boss)."""
    if not state_dict:
        return False
    level = state_dict.get("level")
    if isinstance(level, (int, float)) and level > 0:
        return True
    for key in state_dict:
        if key.startswith(("code_", "mcq_", "frq_", "saved_frq_", "boss_code_")) or (
            key.startswith("boss_") and key.endswith("_solved")
        ):
            return True
    return False


def save_game_silent():
    """Saves Lab 1 game state only (only lab1 keys); only when changed.
    Does not overwrite an existing save that has progress with a fresh "just entered" state
    (so Restart -> Enter Dungeon again does not wipe the file)."""
    state_to_save = {}
    for key, val in st.session_state.items():
        if not _is_lab1_game_key(key):
            continue
        try:
            state_to_save[key] = serialize_value(val)
        except Exception:
            pass
    try:
        existing = {}
        if os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, "r") as f:
                existing = json.load(f)
        if state_to_save == existing:
            return True
        # Don't overwrite a save that has progress with initial state (level=0, no answers)
        if existing and _has_progress(existing) and not _has_progress(state_to_save):
            return True
        os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
        with open(SAVE_FILE, "w") as f:
            json.dump(state_to_save, f, indent=2)
        return True
    except Exception:
        return False


def load_game_silent():
    """Loads Lab 1 game state from lab1_game_save.json into session_state (no toast/rerun)."""
    if not os.path.exists(SAVE_FILE):
        return False
    try:
        with open(SAVE_FILE, "r") as f:
            saved_state = json.load(f)
        for key, val in saved_state.items():
            if _is_lab1_game_key(key):
                st.session_state[key] = deserialize_value(val)
        return True
    except Exception:
        return False


def save_game():
    """Saves Lab 1 game state only to lab1_game_save.json."""
    state_to_save = {}
    for key, val in st.session_state.items():
        if not _is_lab1_game_key(key):
            continue
        try:
            state_to_save[key] = serialize_value(val)
        except Exception as e:
            print(f"Skipping key '{key}' during save due to serialization error: {e}")

    try:
        os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
        with open(SAVE_FILE, "w") as f:
            json.dump(state_to_save, f, indent=2)
        st.toast("Game Saved Successfully!", icon="💾")
    except Exception as e:
        st.error(f"Failed to save game: {e}")


def load_game():
    """Loads game state from file."""
    if not os.path.exists(SAVE_FILE):
        st.error("No save file found!")
        return False

    try:
        with open(SAVE_FILE, "r") as f:
            saved_state = json.load(f)

        # Clear current state or update?
        # Update is safer to keep default keys if not present in save?
        # But if we want to "Reload", we probably want to wipe current state and replace with saved.
        # But `init_game` might have set defaults.
        # Let's use st.session_state.update

        for key, val in saved_state.items():
            if _is_lab1_game_key(key):
                st.session_state[key] = deserialize_value(val)

        st.toast("Game Loaded Successfully!", icon="📂")
        st.rerun()
        return True
    except Exception as e:
        st.error(f"Failed to load game: {e}")
        return False
