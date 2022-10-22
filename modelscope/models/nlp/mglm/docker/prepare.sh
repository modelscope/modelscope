#!/bin/bash

config=/root/.jupyter/jupyter_notebook_config.py

if [ ! -f $config ]; then

  cat > $config <<EOF
c.NotebookApp.allow_password_change = True
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.notebook_dir = '/workspace'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.base_url = '/'
EOF

  default_pwd=$JUPYTER_DEFAULT_PWD

  if [ ! $default_pwd ]; then
    default_pwd=''
    echo "doesn't need password"

    echo "c.NotebookApp.token =''" >> $config
  else
    default_pwd=`python -c "from notebook.auth import passwd; pwd=passwd('${default_pwd}'); print(pwd);"`
    echo "sha1 password: $default_pwd"
    echo "default password: $default_pwd"

    echo "c.NotebookApp.password ='${default_pwd}'" >> $config
  fi

fi
