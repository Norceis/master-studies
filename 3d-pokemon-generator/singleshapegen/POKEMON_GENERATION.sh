#cd voxelization
#python voxelize_for_interp.py -s '..\POKEMON\charizard\pm0006_00_00.obj' --res 100 --n_scales 5 -o '../POKEMON/charizard/charizard.h5'
#python voxelize_for_interp.py -s '../POKEMON/charmander/pm0004_00_00.obj' --res 100 --n_scales 5 -o '../POKEMON/charmander/charmander.h5'
#python voxelize_for_interp.py -s '../POKEMON/charmeleon/pm0005_00_00.obj' --res 100 --n_scales 5 -o '../POKEMON/charmeleon/charmeleon.h5'
#
#cd ..
python main_for_interp_2_heads.py train --tag 'gyarados-charizard-2-heads' -s 'POKEMON/gyarados/gyarados.h5' -ss 'POKEMON/charizard/charizard.h5' -g 0 --n_iters 10000 --G_struct 'triplane_2_heads'

python main_for_interp_with_sliced_samples.py train --tag 'charmeleon-charizard-spliced-samples-10' -s 'POKEMON/charmeleon/charmeleon.h5' -ss 'POKEMON/charizard/charizard.h5' -g 0 --n_iters 8000 --iter_cut 10
python main_for_interp_with_sliced_samples.py train --tag 'gyarados-charizard-spliced-samples-10' -s 'POKEMON/gyarados/gyarados.h5' -ss 'POKEMON/charizard/charizard.h5' -g 0 --n_iters 8000 --iter_cut 10
python main_for_interp_with_sliced_samples.py train --tag 'gyarados-charizard-spliced-samples-50' -s 'POKEMON/gyarados/gyarados.h5' -ss 'POKEMON/charizard/charizard.h5' -g 0 --n_iters 8000 --iter_cut 50

#python main_for_interp.py train --tag 'charmander-charizard-bigger-network' -s 'POKEMON/charmander/charmander.h5' -ss 'POKEMON/charizard/charizard.h5' -g 0 --n_iters 10000 