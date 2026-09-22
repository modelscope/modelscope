#!/bin/bash
set -e
set -o pipefail
# check git is install
git --version >/dev/null 2>&1 || { echo 'git not installed' ; exit 0; }

if [ -z "$MODELSCOPE_USERNAME" ]  || [ -z "$MODELSCOPE_GITLAB_ACCESS_TOKEN" ]; then
    :
else
    git config --global credential.helper store
    echo "http://${MODELSCOPE_USERNAME}:${MODELSCOPE_GITLAB_ACCESS_TOKEN}@www.modelscope.cn">~/.git-credentials
    echo "https://${MODELSCOPE_USERNAME}:${MODELSCOPE_GITLAB_ACCESS_TOKEN}@www.modelscope.cn">>~/.git-credentials
    chmod go-rwx ~/.git-credentials
fi
if [ -z "$MODELSCOPE_USERNAME" ]  || [ -z "$MODELSCOPE_USEREMAIL" ]; then
    :
else
    git config --system user.name ${MODELSCOPE_USERNAME}
    git config --system user.email ${MODELSCOPE_USEREMAIL}
fi
if [ -z "$MODELSCOPE_ENVIRONMENT" ]; then
    :
else
    git config --system --add http.http://www.modelscope.cn.extraHeader "Modelscope_Environment: $MODELSCOPE_ENVIRONMENT"
    git config --system --add http.https://www.modelscope.cn.extraHeader "Modelscope_Environment: $MODELSCOPE_ENVIRONMENT"
fi

if [ -z "$MODELSCOPE_USERNAME" ]; then
    :
else
    git config --system --add http.http://www.modelscope.cn.extraHeader "Modelscope_User: $MODELSCOPE_USERNAME"
    git config --system --add http.https://www.modelscope.cn.extraHeader "Modelscope_User: $MODELSCOPE_USERNAME"
fi

if [ -z "$MODELSCOPE_USERID" ]; then
    :
else
    git config --system --add http.http://www.modelscope.cn.extraHeader "Modelscope_Userid: $MODELSCOPE_USERID"
    git config --system --add http.https://www.modelscope.cn.extraHeader "Modelscope_Userid: $MODELSCOPE_USERID"
fi

if [ -z "$MODELSCOPE_HAVANAID" ]; then
    :
else
    git config --system --add http.http://www.modelscope.cn.extraHeader "Modelscope_Havanaid: $MODELSCOPE_HAVANAID"
    git config --system --add http.https://www.modelscope.cn.extraHeader "Modelscope_Havanaid: $MODELSCOPE_HAVANAID"
fi

pip config set global.index-url https://mirrors.cloud.aliyuncs.com/pypi/simple
pip config set install.trusted-host mirrors.cloud.aliyuncs.com
