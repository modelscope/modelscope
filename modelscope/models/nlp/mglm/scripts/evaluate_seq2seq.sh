export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
mkdir tmp
cat $1 | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > tmp/test.hypo.tokenized
cat $2 | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > tmp/test.hypo.target
files2rouge tmp/test.hypo.tokenized tmp/test.hypo.target
