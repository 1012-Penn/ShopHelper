"""ch14 后台管理员 CLI：创建/列出/启停/改密/解锁。

用法示例：
  python -m scripts.adminctl create --username admin --role admin
  python -m scripts.adminctl list
  python -m scripts.adminctl disable --username alice
  python -m scripts.adminctl reset-password --username alice
  python -m scripts.adminctl unlock --username alice

口令只通过 getpass 或 --password-stdin 读取，不提供明文 --password 参数。
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys

from app.admin_store import ROLES, AdminStore, normalize_username
from app.config import Settings
from app.db import make_engine, make_session_factory


def build_store() -> AdminStore:
    # CLI 不需要模型 key；显式兜底，避免 Settings 因缺 OPENAI_API_KEY 启动失败
    settings = Settings(openai_api_key=os.getenv("OPENAI_API_KEY", "unused"))
    factory = make_session_factory(make_engine(settings))
    return AdminStore(factory, password_iterations=settings.admin_password_iterations)


def read_password(args) -> str:
    if getattr(args, "password_stdin", False):
        password = sys.stdin.readline().rstrip("\n")
        if not password:
            raise SystemExit("从 stdin 读取的口令为空")
        return password
    first = getpass.getpass("口令: ")
    second = getpass.getpass("确认口令: ")
    if first != second:
        raise SystemExit("两次口令不一致")
    return first


def cmd_create(store: AdminStore, args) -> int:
    password = read_password(args)
    try:
        user = store.create_user(args.username, password, args.role)
    except ValueError as exc:
        print(f"创建失败:{exc}", file=sys.stderr)
        return 1
    print(f"已创建管理员:{user.username} 角色={user.role}")
    return 0


def cmd_list(store: AdminStore) -> int:
    users = store.list_users()
    if not users:
        print("暂无管理员，请先 create。")
        return 0
    print("username\trole\tactive\tlast_login\tlocked_until")
    for user in users:
        print(f"{user['username']}\t{user['role']}\t"
              f"{'yes' if user['is_active'] else 'no'}\t"
              f"{user['last_login_at'] or '-'}\t{user['locked_until'] or '-'}")
    return 0


def cmd_set_active(store: AdminStore, args, active: bool) -> int:
    name = normalize_username(args.username)
    target = next((u for u in store.list_users() if u["username"] == name), None)
    if target is None:
        print(f"管理员不存在:{name}", file=sys.stderr)
        return 1
    if not active and target["is_active"] and store.count_admins(active_only=True) <= 1:
        print("拒绝:不能禁用最后一个启用中的管理员", file=sys.stderr)
        return 1
    if not store.set_active(name, active):
        print(f"操作失败:{name}", file=sys.stderr)
        return 1
    print(f"已{'启用' if active else '禁用'}:{name}（旧会话已吊销）")
    return 0


def cmd_reset_password(store: AdminStore, args) -> int:
    password = read_password(args)
    try:
        ok = store.reset_password(args.username, password)
    except ValueError as exc:
        print(f"重置失败:{exc}", file=sys.stderr)
        return 1
    if not ok:
        print(f"管理员不存在:{normalize_username(args.username)}", file=sys.stderr)
        return 1
    print(f"已重置口令:{normalize_username(args.username)}（旧会话已吊销）")
    return 0


def cmd_unlock(store: AdminStore, args) -> int:
    if not store.unlock(args.username):
        print(f"管理员不存在:{normalize_username(args.username)}", file=sys.stderr)
        return 1
    print(f"已解锁:{normalize_username(args.username)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShopHelper 后台管理员管理")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="创建管理员")
    create.add_argument("--username", required=True)
    create.add_argument("--role", choices=ROLES, default="viewer")
    create.add_argument("--password-stdin", action="store_true",
                        help="从 stdin 读一行口令（自动化用）")

    sub.add_parser("list", help="列出管理员")

    disable = sub.add_parser("disable", help="禁用管理员并吊销旧会话")
    disable.add_argument("--username", required=True)

    enable = sub.add_parser("enable", help="启用管理员")
    enable.add_argument("--username", required=True)

    reset = sub.add_parser("reset-password", help="重置口令并吊销旧会话")
    reset.add_argument("--username", required=True)
    reset.add_argument("--password-stdin", action="store_true",
                       help="从 stdin 读一行口令（自动化用）")

    unlock = sub.add_parser("unlock", help="清除登录失败锁定")
    unlock.add_argument("--username", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = build_store()
        if args.command == "create":
            return cmd_create(store, args)
        if args.command == "list":
            return cmd_list(store)
        if args.command == "disable":
            return cmd_set_active(store, args, active=False)
        if args.command == "enable":
            return cmd_set_active(store, args, active=True)
        if args.command == "reset-password":
            return cmd_reset_password(store, args)
        if args.command == "unlock":
            return cmd_unlock(store, args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"数据库操作失败:{exc}", file=sys.stderr)
        print("请确认已 apply db/ch14-admin-auth.sql 且 DATABASE_URL 可用。",
              file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
