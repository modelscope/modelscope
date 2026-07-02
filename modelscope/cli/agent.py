# Copyright (c) Alibaba, Inc. and its affiliates.
"""``modelscope agent`` — upload, download, watch, restore, and stop agent files.

Integration layer that provides agent file management using modelscope's own
endpoint and token (no separate server configuration needed).

Cache data is stored under ``$MODELSCOPE_CACHE/agent/`` by default
(typically ``~/.cache/modelscope/hub/agent/``).
"""

import os
import sys
from argparse import ArgumentParser

from modelscope_hub.cli.base import CLICommand
from modelscope_hub.config import HubConfig


def _init_agent_data_dir():
    """Set agent cache directory under modelscope cache hierarchy.

    Uses the standard ``MODELSCOPE_CACHE`` env var (default ``~/.cache/modelscope/hub``),
    placing agent data at ``$MODELSCOPE_CACHE/agent/``.
    """
    if os.environ.get('ULTRON_DATA_DIR'):
        return  # already explicitly configured by caller
    from pathlib import Path
    cache_root = os.environ.get('MODELSCOPE_CACHE', '').strip()
    if not cache_root:
        cache_root = str(Path.home() / '.cache' / 'modelscope' / 'hub')
    os.environ['ULTRON_DATA_DIR'] = str(Path(cache_root) / 'agent')


def _normalize_endpoint(endpoint: str) -> str:
    """Ensure the endpoint URL has a scheme (default https)."""
    if not endpoint:
        return endpoint
    if not endpoint.startswith(('http://', 'https://')):
        endpoint = f'https://{endpoint}'
    return endpoint.rstrip('/')


def _get_config(args) -> HubConfig:
    """Build a HubConfig using the global --endpoint/--token from the CLI."""
    config = HubConfig(
        endpoint=getattr(args, 'endpoint', None),
        token=getattr(args, 'token', None),
    )
    # HubConfig may resolve endpoint from env (MODELSCOPE_ENDPOINT) without scheme
    if config.endpoint:
        config.endpoint = _normalize_endpoint(config.endpoint)
    return config


def _get_username(config: HubConfig) -> str:
    """Resolve current username via /openapi/v1/users/me."""
    from modelscope_hub._openapi import OpenAPIClient
    client = OpenAPIClient(config=config)
    try:
        data = client.get_current_user()
    except Exception as e:
        _fail(f'failed to resolve current user: {e}')
    if not data:
        _fail('failed to resolve current user: empty response from server.')
    username = data.get('username', data.get('Username', ''))
    if not username:
        _fail('failed to resolve current user: server returned empty username.')
    return username


def _fail(message: str) -> None:
    print(f'Error: {message}', file=sys.stderr)
    sys.exit(1)


def _resolve_repo(repo: str, username: str = '') -> 'tuple[str, str]':
    """Parse --repo value into (group, repo_name).

    - repo contains '/' → split into (group, repo_name)
    - repo without '/' → (username, repo)
    """
    if '/' in repo:
        parts = repo.split('/', 1)
        return parts[0], parts[1]
    return username, repo


class AgentCMD(CLICommand):
    """Command for managing agent resources (upload/download/watch/list/status/backups/convert/restore/stop)."""

    name = 'agent'

    @staticmethod
    def register(subparsers: ArgumentParser) -> None:
        parser = subparsers.add_parser(
            AgentCMD.name,
            help='Manage agent files (upload, download, watch, list, status, restore, backups, convert, stop).')
        sub = parser.add_subparsers(
            dest='agent_action', help='agent subcommands')

        # ---- upload ----
        p_upload = sub.add_parser(
            'upload', help='Upload local agent files to remote repository')
        p_upload.add_argument(
            '-f', '--framework', required=True, help='Agent framework name')
        p_upload.add_argument(
            '-n', '--name', default=None,
            help='Agent name (auto-selects if only one exists locally)')
        p_upload.add_argument(
            '-r', '--repo', required=True,
            help='Remote repo name (required). Supports owner/name format.')
        p_upload.add_argument(
            '--local-dir', default=None, help='Override local workspace root')
        p_upload.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be uploaded without uploading')

        # ---- download ----
        p_download = sub.add_parser(
            'download', help='Download agent files from remote repository')
        p_download.add_argument(
            '-f', '--framework', required=True, help='Agent framework name')
        p_download.add_argument(
            '-r', '--repo', required=True,
            help='Remote repo name (required). Supports owner/name format.')
        p_download.add_argument(
            '-n', '--name', default=None,
            help='Agent name to write as locally (default: default)')
        p_download.add_argument(
            '--local-dir', default=None, help='Override local workspace root')
        p_download.add_argument(
            '--target-framework', default=None,
            help='Convert to a different framework on download')
        p_download.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be written without writing')

        # ---- watch ----
        p_watch = sub.add_parser(
            'watch', help='Start background sync for agent files')
        p_watch.add_argument(
            '-f', '--framework', required=True, help='Agent framework name')
        p_watch.add_argument(
            '-n', '--name', default=None,
            help='Agent name (default: sync all agents)')
        p_watch.add_argument(
            '-r', '--repo', required=True,
            help='Remote repo name (required). Supports owner/name format.')
        p_watch.add_argument(
            '--local-dir', default=None, help='Override local workspace root')
        p_watch.add_argument(
            '--pull', action='store_true',
            help='Enable bidirectional sync (pull remote changes)')

        # ---- list (remote) ----
        p_list = sub.add_parser(
            'list', help='List remote agent repositories')
        p_list.add_argument(
            '--owner', default=None,
            help='Filter by owner (user or organization)')
        p_list.add_argument(
            '--page', dest='page_number', type=int, default=1,
            help='Page number (default: 1)')
        p_list.add_argument(
            '--page-size', dest='page_size', type=int, default=10,
            help='Items per page (default: 10)')

        # ---- status (local) ----
        p_status = sub.add_parser(
            'status', help='Show local agent status for a framework')
        p_status.add_argument(
            '-f', '--framework', required=True, help='Agent framework name')
        p_status.add_argument(
            '--local-dir', default=None, help='Override local workspace root')

        # ---- backups ----
        p_backups = sub.add_parser(
            'backups', help='List available backups')
        p_backups.add_argument(
            '-f', '--framework', default=None, help='Agent framework name')
        p_backups.add_argument(
            '-n', '--name', default=None, help='Agent name')
        p_backups.add_argument(
            '--local-dir', default=None, help='Override local workspace root')

        # ---- restore ----
        p_restore = sub.add_parser(
            'restore', help='Restore agent files from a backup')
        p_restore.add_argument(
            '--from-backup', required=True,
            help="Backup to restore: 'last' or a specific backup filename")
        p_restore.add_argument(
            '-f', '--framework', default=None, help='Agent framework name')
        p_restore.add_argument(
            '-n', '--name', default=None, help='Agent name')
        p_restore.add_argument(
            '--local-dir', default=None, help='Override local workspace root')

        # ---- convert (local only, no network) ----
        p_convert = sub.add_parser(
            'convert', help='Convert local agent files between frameworks')
        p_convert.add_argument(
            '--from-framework', required=True,
            help='Source framework to read from')
        p_convert.add_argument(
            '--target-framework', required=True,
            help='Target framework to write to')
        p_convert.add_argument(
            '--from-name', default=None,
            help='Source agent name to read (default: auto-select or default)')
        p_convert.add_argument(
            '--target-name', default=None,
            help='Target agent name to write as (default: same as --from-name)')
        p_convert.add_argument(
            '--local-dir', default=None,
            help='Source workspace root to read from')
        p_convert.add_argument(
            '--out', default=None,
            help='Destination directory to write to (default: target framework path)')
        p_convert.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be written without writing')

        # ---- stop ----
        sub.add_parser('stop', help='Stop background watch process')

        parser.set_defaults(_command=AgentCMD)

    def execute(self):
        _init_agent_data_dir()
        action = getattr(self.args, 'agent_action', None)
        if not action:
            print(
                'Usage: modelscope agent '
                '<upload|download|watch|list|status|backups|restore|convert|stop>')
            return

        handler = {
            'upload': self._upload,
            'download': self._download,
            'watch': self._watch,
            'list': self._list,
            'status': self._status,
            'backups': self._backups,
            'restore': self._restore,
            'convert': self._convert,
            'stop': self._stop,
        }.get(action)

        if handler:
            handler()
        else:
            _fail(f'unknown action: {action}')

    # ------------------------------------------------------------------
    # Subcommand implementations
    # ------------------------------------------------------------------

    def _upload(self):
        from ultron.cli.client import ApiError, UltronClient
        from ultron.cli.commands import (
            _build_allowlist,
            _frameworks,
            _resolve_local_name,
        )
        from ultron.services.harness.allowlist import (
            ALLOWLIST_REGISTRY,
            GLOBAL_AGENT_NAME,
        )

        framework = self.args.framework
        if framework not in ALLOWLIST_REGISTRY:
            _fail(
                f"unknown framework '{framework}'. Available: {_frameworks()}")

        # Resolve local agent name.
        local_name, err = _resolve_local_name(self.args.name, framework,
                                              self.args.local_dir)
        if err:
            _fail(err)

        spec = _build_allowlist(framework, local_name, self.args.local_dir)
        resources = spec.collect_bytes()
        if not resources:
            display_name = local_name if local_name != GLOBAL_AGENT_NAME else 'global'
            _fail(
                f'no files found for {framework}/{display_name} under {spec.workspace_root}.')

        total_bytes = sum(len(v) for v in resources.values())
        print(f'Found {len(resources)} file(s) ({total_bytes} bytes):')
        for rel in sorted(resources):
            print(f'  {rel} ({len(resources[rel])} B)')

        if self.args.dry_run:
            print('\n[dry-run] nothing uploaded.')
            return

        config = _get_config(self.args)
        if not config.token:
            _fail("not logged in. Run 'modelscope login' first.")
        username = _get_username(config)
        client = UltronClient(config.endpoint, config.token)

        group, repo = _resolve_repo(self.args.repo, username)

        try:
            file_id = client.upload_file(resources)
            client.create_repo(
                group, repo, framework, system_prompt_files=file_id)
        except ApiError as e:
            _fail(f'upload failed (HTTP {e.status}: {e.detail})')
        except Exception as e:
            _fail(f'upload failed: {e}')

        print(f'\nUploaded {len(resources)} file(s) to {group}/{repo}.')

    def _download(self):
        from ultron.cli.client import ApiError, UltronClient
        from ultron.cli.commands import (
            _build_allowlist,
            _convert,
            _frameworks,
        )
        from ultron.services.harness.allowlist import (
            ALLOWLIST_REGISTRY,
            DEFAULT_AGENT_NAME,
        )

        framework = self.args.framework
        if framework not in ALLOWLIST_REGISTRY:
            _fail(
                f"unknown framework '{framework}'. Available: {_frameworks()}")

        config = _get_config(self.args)

        # Token is optional for download (public repos don't require auth).
        # But if --repo doesn't contain '/', we need username to derive group.
        repo_val = self.args.repo
        if '/' not in repo_val and not config.token:
            _fail(
                f"--repo '{repo_val}' requires login to resolve owner. "
                f"Use 'owner/name' format or run 'modelscope login' first.")
        username = _get_username(config) if config.token else ''
        group, repo = _resolve_repo(repo_val, username)

        client = UltronClient(config.endpoint, config.token or '')

        try:
            info = client.repo_info(group, repo)
            if info is None:
                _fail(f'repository {group}/{repo} not found.')
            paths = client.list_repo_files(group, repo)
            if not paths:
                _fail(f'repository {group}/{repo} has no files.')
            resources = {
                p: client.download_repo_file(group, repo, p)
                for p in paths
            }
        except ApiError as e:
            _fail(f'download failed (HTTP {e.status}: {e.detail})')
        except Exception as e:
            _fail(f'download failed: {e}')

        target_fw = self.args.target_framework or framework
        if target_fw not in ALLOWLIST_REGISTRY:
            _fail(
                f"unknown target framework '{target_fw}'. Available: {_frameworks()}")
        if target_fw != framework:
            resources = _convert(resources, framework, target_fw)
            print(
                f'Converted {framework} -> {target_fw} ({len(resources)} file(s)).')

        # Resolve local agent name for writing.
        local_name = self.args.name or DEFAULT_AGENT_NAME
        spec = _build_allowlist(target_fw, local_name, self.args.local_dir)
        root = spec.workspace_root

        # Filter downloaded resources by allowlist patterns.
        patterns = spec.resolved_patterns()
        filtered = {
            k: v
            for k, v in resources.items() if spec.matches(k, patterns)
        }
        skipped = set(resources.keys()) - set(filtered.keys())
        if skipped:
            print(f'Skipped {len(skipped)} file(s) not matching allowlist:')
            for s in sorted(skipped):
                print(f'  [skip] {s}')

        if not filtered:
            _fail('no downloaded files match the local allowlist patterns.')

        print(
            f'{len(filtered)} file(s) for {group}/{repo} (framework={target_fw}):')
        for rel in sorted(filtered):
            print(f'  {rel} -> {root / rel}')

        if self.args.dry_run:
            print('\n[dry-run] nothing written.')
            return

        written = spec.apply(filtered)
        print(f'\nWrote {len(written)} file(s) under {root}.')

    def _watch(self):
        from ultron.cli.cache import pid_file
        from ultron.cli.client import ApiError, UltronClient
        from ultron.cli.commands import (
            _build_allowlist,
            _frameworks,
            _resolve_local_name,
        )
        from ultron.cli.watcher import daemonize, stop_daemon, watch_loop
        from ultron.services.harness.allowlist import (
            ALLOWLIST_REGISTRY,
            ALL_AGENT_NAME,
            DEFAULT_AGENT_NAME,
            GLOBAL_AGENT_NAME,
        )

        framework = self.args.framework
        if framework not in ALLOWLIST_REGISTRY:
            _fail(
                f"unknown framework '{framework}'. Available: {_frameworks()}")

        # Resolve local agent name: if --name not given, default to ALL mode.
        if self.args.name:
            local_name, err = _resolve_local_name(self.args.name, framework,
                                                  self.args.local_dir)
            if err:
                _fail(err)
        else:
            local_name = ALL_AGENT_NAME

        config = _get_config(self.args)
        if not config.token:
            _fail("not logged in. Run 'modelscope login' first.")
        username = _get_username(config)
        client = UltronClient(config.endpoint, config.token)

        # Clean up stale processes
        pf = pid_file()
        stop_daemon(extra_patterns=['modelscope agent watch'])

        spec = _build_allowlist(framework, local_name, self.args.local_dir)

        # Guard: file-per-agent frameworks with a specific agent name.
        if (not spec.supports_individual_watch
                and local_name not in (GLOBAL_AGENT_NAME, ALL_AGENT_NAME,
                                       DEFAULT_AGENT_NAME)):
            _fail(f"'{framework}' has shared files across sub-agents; "
                  f'watch only supports global/default mode.')

        group, repo = _resolve_repo(self.args.repo, username)

        # Framework mismatch guard
        try:
            info = client.repo_info(group, repo)
            if info:
                remote_fw = info.get('Framework') or info.get(
                    'framework') or ''
                if remote_fw and remote_fw != framework:
                    _fail(
                        f'framework mismatch: local={framework}, remote={remote_fw}. '
                        f"Use 'modelscope agent download --target-framework' for cross-framework sync."
                    )
        except ApiError as e:
            if e.status in (403, 401):
                _fail(f'authentication failed (HTTP {e.status})')
            elif e.status == 404:
                pass  # repo not found — first push will create it
            else:
                _fail(
                    f'failed to get repository info (HTTP {e.status}: {e.detail})')
        except Exception as e:
            _fail(f'failed to get repository info: {e}')

        interval = 120
        push_only = not self.args.pull

        print(f'Starting sync for {group}/{repo} (interval={interval}s)...')
        print(f'  Framework: {framework}')
        print(f'  Root: {spec.workspace_root}')
        if push_only:
            print(
                '  Mode: push-only (local -> remote, will NOT pull remote changes)')
        else:
            print(
                '  Mode: bidirectional (local <-> remote, WILL pull remote changes)')
        print('  Stop: modelscope agent stop')

        daemonize(
            watch_loop,
            spec,
            client,
            username,
            repo,
            framework,
            interval,
            push_only=push_only)
        from ultron.cli.cache import log_file
        print(f'  Watch started (PID file: {pf}).')
        print(f'  Logs: {log_file()}')

    def _list(self):
        """List remote agent repositories."""
        config = _get_config(self.args)
        # list does not require token (public repos are visible without auth)
        from ultron.cli.client import ApiError, UltronClient
        client = UltronClient(
            config.endpoint, config.token or '')

        owner = self.args.owner
        page_number = self.args.page_number
        page_size = self.args.page_size

        try:
            result = client.list_agents(
                owner=owner,
                page_number=page_number,
                page_size=page_size,
            )
        except ApiError as e:
            _fail(f'list failed (HTTP {e.status}: {e.detail})')
        except Exception as e:
            _fail(f'list failed: {e}')

        items = result.get('items', [])
        total = result.get('total_count', len(items))

        if not items:
            print('(no agent repositories found)')
            return

        # Render table aligned with `ms list` style
        headers = ['repo_id', 'framework', 'visibility', 'updated']
        rows = []
        for item in items:
            owner_name = item.get('Path') or item.get('owner') or ''
            repo_name = item.get('Name') or item.get('name') or ''
            repo_id = f'{owner_name}/{repo_name}' if owner_name else repo_name
            fw = item.get('Framework') or item.get('framework') or '-'
            vis = item.get('Visibility') or item.get('visibility') or '-'
            updated = item.get('LastUpdatedDate') or item.get('updated_at') or '-'
            if isinstance(updated, str) and 'T' in updated:
                updated = updated.split('T')[0]
            rows.append((repo_id, fw, vis, updated))

        # Simple table rendering
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))

        fmt = '  '.join(f'{{:<{w}}}' for w in col_widths)
        print(fmt.format(*headers))
        print(fmt.format(*['-' * w for w in col_widths]))
        for row in rows:
            print(fmt.format(*[str(v) for v in row]))

        print(f'\npage {page_number} / total {total} (page_size={page_size})')

    def _status(self):
        """Show local agent status for a framework."""
        from ultron.cli.commands import cmd_list
        from types import SimpleNamespace

        ns = SimpleNamespace(
            framework=self.args.framework,
            local_dir=self.args.local_dir,
        )
        rc = cmd_list(ns)
        if rc:
            sys.exit(rc)

    def _backups(self):
        """List available backups."""
        from ultron.cli.commands import cmd_recover
        from types import SimpleNamespace

        ns = SimpleNamespace(
            target=None,
            framework=self.args.framework,
            name=self.args.name,
            local_dir=self.args.local_dir,
            list=True,
        )
        rc = cmd_recover(ns)
        if rc:
            sys.exit(rc)

    def _restore(self):
        """Restore agent files from a backup."""
        from ultron.cli.commands import cmd_recover
        from types import SimpleNamespace

        ns = SimpleNamespace(
            target=self.args.from_backup,
            framework=self.args.framework,
            name=self.args.name,
            local_dir=self.args.local_dir,
            list=False,
        )
        rc = cmd_recover(ns)
        if rc:
            sys.exit(rc)

    def _convert(self):
        """Convert local agent files between frameworks."""
        from ultron.cli.commands import _build_allowlist, _convert, _frameworks
        from ultron.services.harness.allowlist import (
            ALLOWLIST_REGISTRY,
            DEFAULT_AGENT_NAME,
        )

        source_fw = self.args.from_framework
        target_fw = self.args.target_framework
        for fw, label in ((source_fw, '--from-framework'),
                          (target_fw, '--target-framework')):
            if fw not in ALLOWLIST_REGISTRY:
                _fail(f"unknown framework '{fw}' for {label}. "
                      f"Available: {_frameworks()}")

        # Source agent name
        from_name = self.args.from_name or DEFAULT_AGENT_NAME
        # Target agent name defaults to source name
        target_name = self.args.target_name or from_name

        # Read source files
        src_spec = _build_allowlist(source_fw, from_name, self.args.local_dir)
        src_root = src_spec.workspace_root
        resources = src_spec.collect()
        if not resources:
            _fail(f"no {source_fw} files found for agent '{from_name}' "
                  f"under {src_root}.")

        # Convert
        converted = _convert(resources, source_fw, target_fw)

        # Destination
        out_dir = getattr(self.args, 'out', None)
        dst_spec = _build_allowlist(target_fw, target_name, out_dir)
        dst_root = dst_spec.workspace_root

        print(f'Convert {source_fw}/{from_name} ({src_root}) -> '
              f'{target_fw}/{target_name} ({dst_root}):')
        print(f'  {len(resources)} file(s) in, {len(converted)} file(s) out')
        for rel in sorted(converted):
            print(f'  {rel} -> {dst_root / rel}')

        if self.args.dry_run:
            print('\n[dry-run] nothing written.')
            return

        written = dst_spec.apply(converted)
        print(f'\nWrote {len(written)} file(s) under {dst_root}.')

    def _stop(self):
        from ultron.cli.watcher import stop_daemon

        stopped = stop_daemon(extra_patterns=['modelscope agent watch'])
        if stopped:
            print('Watch process stopped.')
        else:
            print('No watch process running.')
