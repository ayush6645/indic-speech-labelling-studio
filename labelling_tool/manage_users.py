"""CLI tool to manage reviewer/admin users for remote ngrok access.

Usage:
    python -m labelling_tool.manage_users add alice --name "Alice" --role reviewer
    python -m labelling_tool.manage_users list
    python -m labelling_tool.manage_users passwd gyan
    python -m labelling_tool.manage_users disable alice
    python -m labelling_tool.manage_users enable alice
"""
import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labelling_tool import auth, db  # noqa: E402


def main():
    db.init_db()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Create or update a user")
    p_add.add_argument("username")
    p_add.add_argument("--name", default="", help="Display name")
    p_add.add_argument("--role", choices=("admin", "reviewer"), default="reviewer")
    p_add.add_argument("--password", default="", help="Password (prompts securely if omitted)")

    p_pw = sub.add_parser("passwd", help="Change password (for enabling remote login)")
    p_pw.add_argument("username")
    p_pw.add_argument("--password", default="")

    p_dis = sub.add_parser("disable", help="Disable user and revoke active sessions")
    p_dis.add_argument("username")

    p_en = sub.add_parser("enable", help="Enable a disabled user")
    p_en.add_argument("username")

    sub.add_parser("list", help="List all users")

    args = ap.parse_args()
    if args.cmd == "list":
        users = auth.list_users()
        if not users:
            print("No users.")
            return
        for u in users:
            st = "active" if u["active"] else "disabled"
            print(f"{u['username']:16} {u['role']:10} {st:10} {u['display_name']}")
        return

    if args.cmd == "add":
        pw = args.password or getpass.getpass(f"Password for {args.username}: ")
        res = auth.create_user(args.username, pw, display_name=args.name or args.username, role=args.role)
        print(f"Saved user {res['username']} ({res['role']})")
        return

    if args.cmd == "passwd":
        users = {u["username"]: u for u in auth.list_users()}
        if args.username not in users:
            print(f"Unknown user: {args.username}", file=sys.stderr)
            sys.exit(1)
        u = users[args.username]
        pw = args.password or getpass.getpass(f"New password for {args.username}: ")
        auth.create_user(args.username, pw, display_name=u["display_name"], role=u["role"])
        print(f"Updated password for {args.username}")
        return

    if args.cmd == "disable":
        auth.set_user_active(args.username, False)
        print(f"Disabled {args.username}")
        return

    if args.cmd == "enable":
        auth.set_user_active(args.username, True)
        print(f"Enabled {args.username}")


if __name__ == "__main__":
    main()
