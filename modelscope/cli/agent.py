# Copyright (c) Alibaba, Inc. and its affiliates.
"""``modelscope agent`` — upload, download, watch, restore, and stop agent files.

Thin integration layer that delegates to ``ultron.cli`` commands while using
modelscope's own endpoint and token (no separate ``--server`` needed).
"""

import sys
from argparse import ArgumentParser

from modelscope_hub.cli.base import CLICommand
from modelscope_hub.config import HubConfig


def _get_config(args) -> HubConfig:
    """Build a HubConfig using the global --endpoint/--token from the CLI."""
    return HubConfig(
        endpoint=getattr(args, "endpoint", None),
        token=getattr(args, "token", None),
    )


def _get_username(config: HubConfig) -> str:
    """Resolve current username via /openapi/v1/users/me."""
    from modelscope_hub._openapi import OpenAPIClient
    client = OpenAPIClient(config=config)
    data = client.get_current_user()
    return data.get("username", data.get("Username", ""))


def _fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


class AgentCMD(CLICommand):
    """Command for managing agent resources (upload/download/watch/restore/stop)."""

    name = 'agent'

    @staticmethod
    def register(subparsers: ArgumentParser) -> None:
        parser = subparsers.add_parser(
            AgentCMD.name, help='Manage agent files (upload, download, watch, restore, stop).')
        sub = parser.add_subparsers(dest='agent_action', help='agent subcommands')

        # ---- upload ----
        p_upload = sub.add_parser('upload', help='Upload local agent files to remote repository')
        p_upload.add_argument('-f', '--framework', required=True, help='Agent framework name')
        p_upload.add_argument('-n', '--name', required=True, help='Sub-agent name')
        p_upload.add_argument('--local_dir', default=None, help='Override local workspace root')
        p_upload.add_argument('--dry-run', action='store_true', help='Show what would be uploaded without uploading')
        p_upload.add_argument('--list', action='store_true', help='List discoverable sub-agents and exit')

        # ---- download ----
        p_download = sub.add_parser('download', help='Download agent files from remote repository')
        p_download.add_argument('-f', '--framework', required=True, help='Agent framework name')
        p_download.add_argument('-n', '--name', required=True, help='Sub-agent name')
        p_download.add_argument('--local_dir', default=None, help='Override local workspace root')
        p_download.add_argument('--target', default=None, help='Convert to a different framework on download')
        p_download.add_argument('--dry-run', action='store_true', help='Show what would be written without writing')

        # ---- watch ----
        p_watch = sub.add_parser('watch', help='Start background sync for agent files')
        p_watch.add_argument('-f', '--framework', required=True, help='Agent framework name')
        p_watch.add_argument('-n', '--name', default=None, help='Sub-agent name (default: all)')
        p_watch.add_argument('--local_dir', default=None, help='Override local workspace root')
        p_watch.add_argument('--pull', action='store_true', help='Enable bidirectional sync (pull remote changes)')

        # ---- restore ----
        p_restore = sub.add_parser('restore', help='Restore agent files from a backup')
        p_restore.add_argument('target', nargs='?', default=None, help="'last' or a backup filename")
        p_restore.add_argument('-f', '--framework', default=None, help='Agent framework name')
        p_restore.add_argument('-n', '--name', default=None, help='Sub-agent name')
        p_restore.add_argument('--local_dir', default=None, help='Override local workspace root')
        p_restore.add_argument('--list', action='store_true', help='List available backups')

        # ---- stop ----
        sub.add_parser('stop', help='Stop background watch process')

        parser.set_defaults(_command=AgentCMD)

    def execute(self):
        action = getattr(self.args, 'agent_action', None)
        if not action:
            print('Usage: modelscope agent <upload|download|watch|restore|stop>')
            return

        handler = {
            'upload': self._upload,
            'download': self._download,
            'watch': self._watch,
            'restore': self._restore,
            'stop': self._stop,
        }.get(action)

        if handler:
            handler()
        else:
            _fail(f"unknown action: {action}")

    # ------------------------------------------------------------------
    # Subcommand implementations
    # ------------------------------------------------------------------

    def _upload(self):
        from ultron.cli.client import ApiError, UltronClient
        from ultron.cli.commands import (
            ALL_AGENT_NAME, _build_allowlist, _frameworks, _repo_name,
        )
        from ultron.services.harness.allowlist import ALLOWLIST_REGISTRY

        framework = self.args.framework
        if framework not in ALLOWLIST_REGISTRY:
            _fail(f"unknown framework '{framework}'. Available: {_frameworks()}")

        if self.args.list:
            spec = _build_allowlist(framework, self.args.name or "default", self.args.local_dir)
            agents = spec.list_agents()
            print(f"Sub-agents for {framework}:")
            for a in agents:
                print(f"  {a}")
            return

        if not self.args.name:
            _fail("--name is required (the internal sub-agent name)")

        spec = _build_allowlist(framework, self.args.name, self.args.local_dir)
        resources = spec.collect()
        if not resources:
            _fail(f"no files found for {framework}/{self.args.name} under {spec.workspace_root}.")

        total_bytes = sum(len(c.encode("utf-8")) for c in resources.values())
        print(f"Found {len(resources)} file(s) ({total_bytes} bytes):")
        for rel in sorted(resources):
            print(f"  {rel} ({len(resources[rel].encode('utf-8'))} B)")

        if self.args.dry_run:
            print("\n[dry-run] nothing uploaded.")
            return

        config = _get_config(self.args)
        if not config.token:
            _fail("not logged in. Run 'modelscope login' first.")
        username = _get_username(config)
        client = UltronClient(config.endpoint, config.token)
        repo = _repo_name(framework, self.args.name)

        try:
            file_id = client.upload_file(resources)
            client.create_repo(username, repo, framework, system_prompt_files=file_id)
        except ApiError as e:
            _fail(f"upload failed (HTTP {e.status}: {e.detail})")

        print(f"\nUploaded {len(resources)} file(s) to {username}/{repo}.")

    def _download(self):
        from ultron.cli.client import ApiError, UltronClient
        from ultron.cli.commands import (
            _build_allowlist, _convert, _frameworks, _repo_name,
        )
        from ultron.services.harness.allowlist import ALLOWLIST_REGISTRY

        framework = self.args.framework
        if framework not in ALLOWLIST_REGISTRY:
            _fail(f"unknown framework '{framework}'. Available: {_frameworks()}")

        if not self.args.name:
            _fail("--name is required")

        config = _get_config(self.args)
        if not config.token:
            _fail("not logged in. Run 'modelscope login' first.")
        username = _get_username(config)
        client = UltronClient(config.endpoint, config.token)
        repo = _repo_name(framework, self.args.name)

        try:
            info = client.repo_info(username, repo)
            if info is None:
                _fail(f"repository {username}/{repo} not found.")
            paths = client.list_repo_files(username, repo)
            if not paths:
                _fail(f"repository {username}/{repo} has no files.")
            resources = {p: client.download_repo_file(username, repo, p) for p in paths}
        except ApiError as e:
            _fail(f"download failed (HTTP {e.status}: {e.detail})")

        target_fw = self.args.target or framework
        if target_fw not in ALLOWLIST_REGISTRY:
            _fail(f"unknown target framework '{target_fw}'. Available: {_frameworks()}")
        if target_fw != framework:
            resources = _convert(resources, framework, target_fw)
            print(f"Converted {framework} -> {target_fw} ({len(resources)} file(s)).")

        spec = _build_allowlist(target_fw, self.args.name, self.args.local_dir)
        root = spec.workspace_root
        print(f"{len(resources)} file(s) for {username}/{repo} (framework={target_fw}):")
        for rel in sorted(resources):
            print(f"  {rel} -> {root / rel}")

        if self.args.dry_run:
            print("\n[dry-run] nothing written.")
            return

        written = spec.apply(resources)
        print(f"\nWrote {len(written)} file(s) under {root}.")

    def _watch(self):
        from ultron.cli.cache import pid_file
        from ultron.cli.client import ApiError, UltronClient
        from ultron.cli.commands import (
            ALL_AGENT_NAME, _build_allowlist, _frameworks, _repo_name,
        )
        from ultron.cli.watcher import daemonize, stop_daemon, watch_loop
        from ultron.services.harness.allowlist import ALLOWLIST_REGISTRY

        framework = self.args.framework
        if framework not in ALLOWLIST_REGISTRY:
            _fail(f"unknown framework '{framework}'. Available: {_frameworks()}")

        name = self.args.name or ALL_AGENT_NAME

        config = _get_config(self.args)
        if not config.token:
            _fail("not logged in. Run 'modelscope login' first.")
        username = _get_username(config)
        client = UltronClient(config.endpoint, config.token)

        # Clean up stale processes
        pf = pid_file()
        if pf.exists():
            stop_daemon()

        spec = _build_allowlist(framework, name, self.args.local_dir)

        if not spec.supports_individual_watch and name != ALL_AGENT_NAME:
            _fail(
                f"'{framework}' has shared files across sub-agents; "
                f"watch only supports '--name all'."
            )

        repo = _repo_name(framework, name)

        # Framework mismatch guard
        try:
            info = client.repo_info(username, repo)
            if info:
                remote_fw = info.get("Framework") or info.get("framework") or ""
                if remote_fw and remote_fw != framework:
                    _fail(
                        f"framework mismatch: local={framework}, remote={remote_fw}. "
                        f"Use 'modelscope agent download --target' for cross-framework sync."
                    )
        except ApiError as e:
            if e.status in (403, 401):
                _fail(f"authentication failed (HTTP {e.status})")

        interval = 120
        push_only = not self.args.pull

        print(f"Starting sync for {username}/{repo} (interval={interval}s)...")
        print(f"  Framework: {framework}")
        print(f"  Root: {spec.workspace_root}")
        if push_only:
            print(f"  Mode: push-only (local -> remote, will NOT pull remote changes)")
        else:
            print(f"  Mode: bidirectional (local <-> remote, WILL pull remote changes)")
        print(f"  Stop: modelscope agent stop")

        daemonize(watch_loop, spec, client, username, repo, framework, interval, push_only=push_only)
        print(f"  Watch started (PID file: {pf}).")

    def _restore(self):
        from ultron.cli.commands import cmd_recover
        from types import SimpleNamespace

        # Map our args to what cmd_recover expects
        ns = SimpleNamespace(
            target=self.args.target,
            framework=self.args.framework,
            name=self.args.name,
            local_dir=self.args.local_dir,
            list=self.args.list,
        )
        rc = cmd_recover(ns)
        if rc:
            sys.exit(rc)

    def _stop(self):
        from ultron.cli.watcher import stop_daemon

        stopped = stop_daemon()
        if stopped:
            print("Watch process stopped.")
        else:
            print("No watch process running.")
