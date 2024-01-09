for D in ./*/;
do
    cd "${D}";
    python ../../../python/clean_index_error.py
    cd ..;
done