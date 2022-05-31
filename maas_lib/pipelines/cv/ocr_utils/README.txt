# compile
# step 1. compile tf ops
python ./manage.py clean_op
modify cpp/CMakeLists.txt line22 path to result of "import tensorflow as tf; print(tf.sysconfig.get_lib())"
python ./manage.py build_op

# step 2. compile lanms
cd lanms
make