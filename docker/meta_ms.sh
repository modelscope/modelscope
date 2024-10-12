
pip uninstall ms-swift modelscope

cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone -b $MODELSCOPE_BRANCH  --single-branch https://github.com/modelscope/modelscope.git && cd modelscope && pip install .[all] && cd / && rm -fr /tmp/modelscope && pip cache purge;
cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone -b $SWIFT_BRANCH  --single-branch https://github.com/modelscope/ms-swift.git && cd ms-swift && pip install .[all] && cd / && rm -fr /tmp/ms-swift && pip cache purge;

pip install adaseq

pip install pai-easycv