import os
from pathlib import Path

from modelscope.hub.commit_scheduler import CommitScheduler
from modelscope.utils.repo_utils import CommitInfo


class DummyHubApi:

    def __init__(self):
        self.commits = []

    def create_repo(self,
                    repo_id,
                    token=None,
                    repo_type=None,
                    visibility='public',
                    exist_ok=True,
                    create_default_config=False):
        return f'https://hub/{repo_id}'

    def create_commit(self,
                      repo_id,
                      repo_type,
                      operations,
                      commit_message,
                      revision=None,
                      token=None):
        self.commits.append((repo_id, operations))
        return CommitInfo(
            commit_url='',
            commit_message=commit_message,
            commit_description='',
            oid='0')


def test_commit_scheduler_push_only_changed_files(tmp_path):
    file = tmp_path / 'data.txt'
    file.write_text('start')

    api = DummyHubApi()
    scheduler = CommitScheduler(
        repo_id='owner/repo', folder_path=tmp_path, every=0.001, hf_api=api)

    scheduler.stop()
    scheduler._scheduler_thread.join()
    api.commits.clear()

    file.write_text('updated')
    commit_info = scheduler.push_to_hub()
    assert commit_info is not None
    assert len(api.commits) == 1

    commit_info2 = scheduler.push_to_hub()
    assert commit_info2 is None
    assert len(api.commits) == 1

    scheduler.executor.shutdown(wait=True)
