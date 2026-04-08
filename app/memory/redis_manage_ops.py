"""
Redis maintenance helpers for M3 session message lists (read / selective clear).

Design notes (conceptually similar to external ops scripts such as
``IC-Self-Study/redis/redis_clear_intent_sentence_ops.py`` in IC-RAG-Agent docs:
argparse CLI, explicit Redis URL, batch key operations) but **key layout and logic
are specific to this app** — ``{prefix}session:{id}:meta`` / ``:messages``.

Run from repo root (loads ``PROJECT_ROOT/.env``):

    python -m app.memory.redis_manage_ops --session-id <uuid> -n 10
    python -m app.memory.redis_manage_ops --user-id local-dev -n 5 --type query
    python -m app.memory.redis_manage_ops --session-id <uuid> --clear
    python -m app.memory.redis_manage_ops --session-id <uuid> --clear -n 3 --type answer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import redis
from dotenv import load_dotenv

from app.config import get_redis_settings
from app.memory.session_store import (
    normalize_stored_message,
    session_meta_key,
    session_messages_key,
)

logger = logging.getLogger(__name__)


def _project_root_env_path() -> Path:
    """``app/memory`` -> parents[2] == repo root."""
    return Path(__file__).resolve().parents[2] / ".env"


def connect_redis_for_ops() -> Tuple[redis.Redis, str]:
    """
    Load ``.env`` from repo root and connect using ``REDIS_URL``.

    Returns:
        (client, key_prefix)

    Raises:
        SystemExit: When ``REDIS_URL`` is missing or connection fails.
    """
    env_path = _project_root_env_path()
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    rs = get_redis_settings()
    if not (rs.url or "").strip():
        raise SystemExit("REDIS_URL is not set. Set it in .env or the environment.")
    try:
        client = redis.Redis.from_url(rs.url.strip(), decode_responses=True)
        client.ping()
    except redis.RedisError as exc:
        raise SystemExit(f"Redis connection failed: {exc}") from exc
    return client, rs.key_prefix


def session_id_from_meta_key(key: str, prefix: str) -> Optional[str]:
    """Parse ``session_id`` from ``{prefix}session:{id}:meta``."""
    needle = f"{prefix}session:"
    if not key.startswith(needle) or not key.endswith(":meta"):
        return None
    return key[len(needle) : -len(":meta")]


def iter_session_meta_keys(client: redis.Redis, prefix: str) -> Iterator[str]:
    """SCAN Redis for session meta keys (non-blocking vs KEYS *)."""
    pattern = f"{prefix}session:*:meta"
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=pattern, count=256)
        for k in keys:
            yield k
        if cursor == 0:
            break


def meta_user_id(client: redis.Redis, meta_key: str) -> Optional[str]:
    """Return ``user_id`` field from session meta hash."""
    try:
        uid = client.hget(meta_key, "user_id")
    except redis.RedisError as exc:
        logger.warning("hget failed %s: %s", meta_key, exc)
        return None
    if uid is None:
        return None
    return str(uid).strip()


def find_session_ids_for_user(client: redis.Redis, prefix: str, user_id: str) -> List[str]:
    """Collect ``session_id`` values whose meta ``user_id`` matches."""
    want = (user_id or "").strip()
    if not want:
        return []
    out: List[str] = []
    for meta_k in iter_session_meta_keys(client, prefix):
        sid = session_id_from_meta_key(meta_k, prefix)
        if not sid:
            continue
        owner = meta_user_id(client, meta_k)
        if owner == want:
            out.append(sid)
    return sorted(set(out))


def assert_session_owner(client: redis.Redis, prefix: str, session_id: str, user_id: str) -> str:
    """
    Ensure meta exists and ``user_id`` matches.

    Returns:
        Owner ``user_id`` from meta.

    Raises:
        ValueError: On missing session or owner mismatch.
    """
    meta_k = session_meta_key(prefix, session_id)
    owner = meta_user_id(client, meta_k)
    if owner is None:
        raise ValueError(f"No session meta for session_id={session_id!r}")
    if owner != (user_id or "").strip():
        raise ValueError(
            f"session_id={session_id!r} belongs to user_id={owner!r}, not {user_id!r}"
        )
    return owner


def load_raw_and_normalized(
    client: redis.Redis,
    prefix: str,
    session_id: str,
    owner_user_id: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Read messages list as raw JSON strings plus normalized dicts (same order).

    Args:
        client: Redis client.
        prefix: Key prefix.
        session_id: Session id.
        owner_user_id: Meta owner (for normalization of legacy rows).

    Returns:
        (raw_strings, normalized_dicts) aligned by index where parse succeeded;
        failed JSON rows are dropped from both in sync — actually we keep raw
        for rewrite; simpler: parallel lists only for successfully parsed.

    For rewrite we need full raw list with indices. Return:
        list of tuples (index, raw, optional_norm)
    """
    msg_k = session_messages_key(prefix, session_id)
    raw_list = client.lrange(msg_k, 0, -1)
    norms: List[Optional[Dict[str, Any]]] = []
    for raw in raw_list:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            norms.append(None)
            continue
        norms.append(normalize_stored_message(obj, session_id, owner_user_id))
    return raw_list, norms


def filter_tail_messages(
    normalized: List[Optional[Dict[str, Any]]],
    *,
    message_type: Optional[str],
    count: int,
) -> List[Dict[str, Any]]:
    """
    From chronological list, keep entries that match ``message_type`` (if set),
    then take the last ``count`` non-None normalized dicts.

    If ``message_type`` is None, use all non-None entries, last ``count``.
    """
    typed = message_type.strip().lower() if message_type else None
    candidates: List[Dict[str, Any]] = []
    for n in normalized:
        if n is None:
            continue
        if typed and (n.get("type") or "").strip().lower() != typed:
            continue
        candidates.append(n)
    if count <= 0:
        return candidates
    return candidates[-count:]


def merge_recent_across_sessions(
    client: redis.Redis,
    prefix: str,
    session_ids: List[str],
    owner_user_id: str,
    *,
    message_type: Optional[str],
    count: int,
) -> List[Dict[str, Any]]:
    """
    Merge messages from multiple sessions, sort by ``timestamp`` descending,
    apply optional type filter, return up to ``count`` items.
    """
    merged: List[Dict[str, Any]] = []
    for sid in session_ids:
        try:
            _, norms = load_raw_and_normalized(client, prefix, sid, owner_user_id)
        except redis.RedisError as exc:
            logger.warning("skip session %s: %s", sid, exc)
            continue
        for n in norms:
            if n is None:
                continue
            if message_type and (n.get("type") or "").strip().lower() != message_type.strip().lower():
                continue
            row = dict(n)
            row["_session_id"] = sid
            merged.append(row)
    merged.sort(key=lambda x: (x.get("timestamp") or ""), reverse=True)
    if count <= 0:
        return merged
    return merged[:count]


def clear_messages_list(
    client: redis.Redis,
    prefix: str,
    session_id: str,
    owner_user_id: str,
    *,
    remove_last_n_matching: Optional[int],
    message_type: Optional[str],
) -> Tuple[int, int]:
    """
    Rewrite ``:messages`` list after removing items from the tail.

    Args:
        remove_last_n_matching: If ``None``, delete entire messages key.
        message_type: If set, only count/remove entries whose ``type`` matches.

    Returns:
        (removed_count, remaining_count)
    """
    assert_session_owner(client, prefix, session_id, owner_user_id)
    msg_k = session_messages_key(prefix, session_id)
    raw_list, norms = load_raw_and_normalized(client, prefix, session_id, owner_user_id)

    if remove_last_n_matching is None:
        deleted = len(raw_list)
        try:
            client.delete(msg_k)
        except redis.RedisError as exc:
            raise RuntimeError(f"Redis delete failed: {exc}") from exc
        return deleted, 0

    typed = message_type.strip().lower() if message_type else None
    # Indices to remove: walk from end, mark last N matching norm indices
    remove_idx: set[int] = set()
    need = remove_last_n_matching
    if need <= 0:
        return 0, len(raw_list)

    for i in range(len(norms) - 1, -1, -1):
        if need <= 0:
            break
        n = norms[i]
        if n is None:
            continue
        if typed and (n.get("type") or "").strip().lower() != typed:
            continue
        remove_idx.add(i)
        need -= 1

    new_raw = [raw_list[i] for i in range(len(raw_list)) if i not in remove_idx]
    try:
        pipe = client.pipeline(transaction=True)
        pipe.delete(msg_k)
        if new_raw:
            pipe.rpush(msg_k, *new_raw)
        pipe.execute()
    except redis.RedisError as exc:
        raise RuntimeError(f"Redis rewrite failed: {exc}") from exc
    return len(remove_idx), len(new_raw)


def clear_all_sessions_for_user(
    client: redis.Redis,
    prefix: str,
    user_id: str,
    *,
    remove_last_n_matching: Optional[int],
    message_type: Optional[str],
) -> List[Tuple[str, int, int]]:
    """
    Apply :func:`clear_messages_list` to every session owned by ``user_id``.

    Returns:
        List of (session_id, removed, remaining) per session.
    """
    sids = find_session_ids_for_user(client, prefix, user_id)
    results: List[Tuple[str, int, int]] = []
    for sid in sids:
        rem, left = clear_messages_list(
            client,
            prefix,
            sid,
            user_id,
            remove_last_n_matching=remove_last_n_matching,
            message_type=message_type,
        )
        results.append((sid, rem, left))
    return results


def _print_messages(rows: List[Dict[str, Any]]) -> None:
    for i, row in enumerate(rows, start=1):
        m = dict(row)
        sid = m.pop("_session_id", None)
        extra = f" session_id={sid}" if sid else ""
        print(
            f"{i}.{extra}\n"
            f"   type={m.get('type')!r} ts={m.get('timestamp')!r}\n"
            f"   content={m.get('content')!r}\n"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect or clear IC-AI-Chat-Client session messages in Redis.",
    )
    p.add_argument(
        "--user-id",
        default=None,
        help="Owner user_id (default: USER_ID from .env, else local-dev).",
    )
    p.add_argument(
        "--session-id",
        default=None,
        help="Restrict to one session (recommended for --clear).",
    )
    p.add_argument(
        "-n",
        "--count",
        type=int,
        default=None,
        help="List: max rows (default 20; 0 = no limit). "
        "Clear: omit to wipe entire message list; else remove last N rows matching --type "
        "(or any type if --type omitted).",
    )
    p.add_argument(
        "--type",
        dest="message_type",
        default=None,
        help="Filter by message type (e.g. query, answer).",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Delete messages. Without -n/--count: empty entire list. "
        "With -n: remove last N entries matching --type (or any type if --type omitted).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry: list or clear session messages."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    import os

    effective_user = (args.user_id or os.getenv("USER_ID") or "local-dev").strip()

    client, prefix = connect_redis_for_ops()

    try:
        if args.clear:
            # No -n / --count => delete entire :messages key(s). With -n N => drop last N matches.
            if args.count is None or args.count == 0:
                remove_n = None
            else:
                remove_n = args.count

            if args.session_id:
                rem, left = clear_messages_list(
                    client,
                    prefix,
                    args.session_id.strip(),
                    effective_user,
                    remove_last_n_matching=remove_n,
                    message_type=args.message_type,
                )
                print(f"session_id={args.session_id!r} removed={rem} remaining={left}")
            else:
                rows = clear_all_sessions_for_user(
                    client,
                    prefix,
                    effective_user,
                    remove_last_n_matching=remove_n,
                    message_type=args.message_type,
                )
                for sid, rem, left in rows:
                    print(f"session_id={sid!r} removed={rem} remaining={left}")
            return 0

        list_limit = 20 if args.count is None else args.count
        count = 10**9 if list_limit == 0 else list_limit

        if args.session_id:
            sid = args.session_id.strip()
            owner = assert_session_owner(client, prefix, sid, effective_user)
            _, norms = load_raw_and_normalized(client, prefix, sid, owner)
            rows = filter_tail_messages(norms, message_type=args.message_type, count=count)
            _print_messages([dict(m) for m in rows])
        else:
            sids = find_session_ids_for_user(client, prefix, effective_user)
            if not sids:
                print(f"No sessions for user_id={effective_user!r}")
                return 0
            rows = merge_recent_across_sessions(
                client,
                prefix,
                sids,
                effective_user,
                message_type=args.message_type,
                count=count,
            )
            _print_messages([dict(m) for m in rows])
        return 0
    except (ValueError, RuntimeError, SystemExit) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            client.close()
        except redis.RedisError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
