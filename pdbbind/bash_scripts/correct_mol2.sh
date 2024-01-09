for D in '3qlb' '4oem' '4oel' '2fov' '2fou' '2foy'
do
    cd "${D}";
    echo "${D}";
    FNAME=$(basename ${D})
    FPATH=$(dirname ${D})
    DNAME=$(basename ${FNAME})
    echo $DNAME

    mv ${DNAME}_ligand.mol2 ${DNAME}_ligand.mol2.record;
    babel ${DNAME}_ligand.sdf -omol2 ${DNAME}_ligand.mol2;
    cd ..
done