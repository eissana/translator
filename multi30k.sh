#!/bash/bin

mkdir -p data/multi30k/

wget https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz
tar -xzvf training.tar.gz -C data/multi30k/
rm -f training.tar.gz

wget https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz
tar -xzvf validation.tar.gz -C data/multi30k/
rm -f validation.tar.gz

wget https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz
tar -xzvf mmt16_task1_test.tar.gz -C data/multi30k/
rm -f mmt16_task1_test.tar.gz

mv data/multi30k/test.de data/multi30k/test2016.de
mv data/multi30k/test.en data/multi30k/test2016.en
mv data/multi30k/test.fr data/multi30k/test2016.fr
