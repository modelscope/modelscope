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
    return data.get('username', data.get('Username', ''))


def _fail(message: str) -> None:
    print(f'Error: {message}', file=sys.stderr)
    sys.exit(1)


class AgentCMD(CLICommand):
    """Command for managing agent resources (upload/download/watch/restore/stop)."""

    name = 'agent'

    @staticmethod
    def register(subparsers: ArgumentParser) -> None:
        parser = subparsers.add_parser(
            AgentCMD.name,
            help=
            'Manage agent files (upload, download, watch, restore, stop, list).'
        )
        sub = parser.add_subparsers(
            dest='agent_action', help='agent subcommands')

        # ---- upload ----
        p_upload = sub.add_parser(
            'upload', help='Upload local agent files to remote repository')
        p_upload.add_argument(
            '-f', '--framework', required=True, help='Agent framework name')
        p_upload.add_argument(
            '-n',
            '--name',
            default=None,
            help='Local sub-agent name (auto-selects if only one)')
        p_upload.add_argument(
            '-r',
            '--repo',
            default=None,
            help='Remote repo name. Supports group/name format.')
        p_upload.add_argument(
            '--local_dir', default=None, help='Override local workspace root')
        p_upload.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be uploaded without uploading')

        # ---- download ----
        p_download = sub.add_parser(
            'download', help='Download agent files from remote repository')
        p_download.add_argument(
            '-f', '--framework', required=True, help='Agent framework name')
        p_download.add_argument(
            '-r',
            '--repo',
            required=True,
            help='Remote repo name (required). Supports group/name format.')
        p_download.add_argument(
            '-n',
            '--name',
            default=None,
            help='Local sub-agent name to write as (default: default)')
        p_download.add_argument(
            '--local_dir', default=None, help='Override local workspace root')
        p_download.add_argument(
            '--target',
            default=None,
            help='Convert to a different framework on download')
        p_download.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be written without writing')

        # ---- watch ----
        p_watch = sub.add_parser(
            'watch', help='Start background sync for agent files')
        p_watch.add_argument(
            '-f', '--framework', required=True, help='Agent framework name')
        p_watch.add_argument(
            '-n',
            '--name',
            default=None,
            help='Local sub-agent name (default: global/shared only)')
        p_watch.add_argument(
            '-r',
            '--repo',
            default=None,
            help='Remote repo name. Supports group/name format.')
        p_watch.add_argument(
            '--local_dir', default=None, help='Override local workspace root')
        p_watch.add_argument(
            '--pull',
            action='store_true',
            help='Enable bidirectional sync (pull remote changes)')

        # ---- list ----
        p_list = sub.add_parser(
            'list', help='List discoverable sub-agents for a framework')
        p_list.add_argument(
            '-f', '--framework', required=True, help='Agent framework name')
        p_list.add_argument(
            '--local_dir', default=None, help='Override local workspace root')

        # ---- restore ----
        p_restore = sub.add_parser(
            'restore', help='Restore agent files from a backup')
        p_restore.add_argument(
            'target',
            nargs='?',
            default=None,
            help="'last' or a backup filename")
        p_restore.add_argument(
            '-f', '--framework', default=None, help='Agent framework name')
        p_restore.add_argument(
            '-n', '--name', default=None, help='Sub-agent name')
        p_restore.add_argument(
            '--local_dir', default=None, help='Override local workspace root')
        p_restore.add_argument(
            '--list', action='store_true', help='List available backups')

        # ---- stop ----
        sub.add_parser('stop', help='Stop background watch process')

        parser.set_defaults(_command=AgentCMD)

    def execute(self):
        _init_agent_data_dir()
        action = getattr(self.args, 'agent_action', None)
        if not action:
            print(
                'Usage: modelscope agent <upload|download|watch|list|restore|stop>'
            )
            return

        handler = {
            'upload': self._upload,
            'download': self._download,
            'watch': self._watch,
            'list': self._list,
            'restore': self._restore,
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
            _resolve_remote,
        )
        from ultron.services.harness.allowlist import (
            ALLOWLIST_REGISTRY,
            DEFAULT_AGENT_NAME,
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
                f'no files found for {framework}/{display_name} under {spec.workspace_root}.'
            )

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

        # Resolve remote target.
        effective_name = self.args.name if self.args.name else None
        group, repo = _resolve_remote(
            repo=getattr(self.args, 'repo', None),
            name=effective_name,
            framework=framework,
            username=username,
        )

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
            _resolve_remote,
        )
        from ultron.services.harness.allowlist import (
            ALLOWLIST_REGISTRY,
            DEFAULT_AGENT_NAME,
        )

        framework = self.args.framework
        if framework not in ALLOWLIST_REGISTRY:
            _fail(
                f"unknown framework '{framework}'. Available: {_frameworks()}")

        if not getattr(self.args, 'repo', None):
            _fail('--repo is required for download')

        config = _get_config(self.args)
        if not config.token:
            _fail("not logged in. Run 'modelscope login' first.")
        username = _get_username(config)
        client = UltronClient(config.endpoint, config.token)

        # Resolve remote target.
        group, repo = _resolve_remote(
            repo=self.args.repo,
            name=self.args.name,
            framework=framework,
            username=username,
        )

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

        target_fw = self.args.target or framework
        if target_fw not in ALLOWLIST_REGISTRY:
            _fail(
                f"unknown target framework '{target_fw}'. Available: {_frameworks()}"
            )
        if target_fw != framework:
            resources = _convert(resources, framework, target_fw)
            print(
                f'Converted {framework} -> {target_fw} ({len(resources)} file(s)).'
            )

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
            f'{len(filtered)} file(s) for {group}/{repo} (framework={target_fw}):'
        )
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
            _resolve_remote,
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

        # Resolve remote target.
        effective_name = self.args.name if self.args.name else None
        group, repo = _resolve_remote(
            repo=getattr(self.args, 'repo', None),
            name=effective_name,
            framework=framework,
            username=username,
        )

        # Framework mismatch guard
        try:
            info = client.repo_info(group, repo)
            if info:
                remote_fw = info.get('Framework') or info.get(
                    'framework') or ''
                if remote_fw and remote_fw != framework:
                    _fail(
                        f'framework mismatch: local={framework}, remote={remote_fw}. '
                        f"Use 'modelscope agent download --target' for cross-framework sync."
                    )
        except ApiError as e:
            if e.status in (403, 401):
                _fail(f'authentication failed (HTTP {e.status})')
            elif e.status == 404:
                pass  # repo not found — first push will create it
            else:
                _fail(
                    f'failed to get repository info (HTTP {e.status}: {e.detail})'
                )
        except Exception as e:
            _fail(f'failed to get repository info: {e}')

        interval = 120
        push_only = not self.args.pull

        print(f'Starting sync for {group}/{repo} (interval={interval}s)...')
        print(f'  Framework: {framework}')
        print(f'  Root: {spec.workspace_root}')
        if push_only:
            print(
                '  Mode: push-only (local -> remote, will NOT pull remote changes)'
            )
        else:
            print(
                '  Mode: bidirectional (local <-> remote, WILL pull remote changes)'
            )
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
        from ultron.cli.commands import cmd_list
        from types import SimpleNamespace

        ns = SimpleNamespace(
            framework=self.args.framework,
            local_dir=getattr(self.args, 'local_dir', None),
        )
        rc = cmd_list(ns)
        if rc:
            sys.exit(rc)

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

        stopped = stop_daemon(extra_patterns=['modelscope agent watch'])
        if stopped:
            print('Watch process stopped.')
        else:
            print('No watch process running.')
