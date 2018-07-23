import numpy as np
from DSPFP import DSPFP_faster_K, greedy_assignment
from bundle_segmentation import graph_matching
import nibabel as nib
from dipy.tracking.utils import length
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.distances import bundles_distances_mam
import pickle


def resample_tractogram(tractogram, step_size):
    """Resample the tractogram with the given step size.
    """
    lengths=list(length(tractogram))
    tractogram_res = []
    for i, f in enumerate(tractogram):
		nb_res_points = np.int(np.floor(lengths[i]/step_size))
		tmp = set_number_of_points(f, nb_res_points)
		tractogram_res.append(tmp)
    return tractogram_res


def compute_superset(true_tract, kdt, prototypes, k=1000, distance_func=bundles_distances_mam):
    """Compute a superset of the true target tract with k-NN.
    """
    true_tract = np.array(true_tract, dtype=np.object)
    dm_true_tract = distance_func(true_tract, prototypes)
    D, I = kdt.query(dm_true_tract, k=k)
    superset_idx = np.unique(I.flat)
    return superset_idx    


def compute_dsc(estimated_tract, true_tract):

    aff=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0, 0, 1.25, -72],[0, 0, 0, 1]])
    voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=aff).keys()
    voxel_list_true_tract = streamline_mapping(true_tract, affine=aff).keys()
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)
    return DSC


if __name__ == '__main__':

	sub_list = ['910241']#, '910443', '911849', '991267', '983773', '917558', '919966', '990366', '927359', '929464']
	example_list = ['500222']#, '506234', '510225', '510326', '512835']
	#tract_name_list_wma = ['Left_pArc', 'Left_TPC', 'Left_MdLF-SPL', 'Left_MdLF-Ang']
	tract_name_list_wma = ['Right_pArc', 'Right_TPC', 'Right_MdLF-SPL', 'Right_MdLF-Ang']
	trk_tracts_dir_wma = 'derivatives/ens_prob_wma'
	tractogram_dir = 'derivatives/brain-life.app-ensembletracking'
	kdt_and_prototypes_dir = 'results/kdt_and_prototypes'
	k = 500
	lam = 0.9

	DSC_results = np.zeros((len(sub_list), len(tract_name_list_wma), len(example_list)))

	for ss, sub in enumerate(sub_list):

		print("Loading tractogram of subject %s..." %sub)
		static_tractogram_filename = '%s/sub-%s/sub-%s_track.tck' %(tractogram_dir, sub, sub)
		static_tractogram = nib.streamlines.load(static_tractogram_filename)
		static_tractogram = static_tractogram.streamlines
		print("Resampling tractogram with step size = 0.625 mm")
		static_tractogram_res = resample_tractogram(static_tractogram, step_size=0.625)
		static_tractogram = np.array(static_tractogram_res, dtype=np.object)

		kdt_filename = '%s/%s_kdt' %(kdt_and_prototypes_dir, sub)
		prototypes_filename = '%s/%s_prototypes.npy' %(kdt_and_prototypes_dir, sub)
		kdt = pickle.load(open(kdt_filename))
		prototypes = np.load(prototypes_filename)

		for ee, example in enumerate(example_list):

			for tt, tract_name in enumerate(tract_name_list_wma):

				example_tract_filename = '%s/sub-%s/sub-%s_%s_tract.trk' %(trk_tracts_dir_wma, example, example, tract_name)
				example_tract = nib.streamlines.load(example_tract_filename)
				example_tract = example_tract.streamlines
				example_tract_res = resample_tractogram(example_tract, step_size=0.625)
				example_tract = example_tract_res

				print("Computing superset with k = %s" %k)
				superset_idx = compute_superset(example_tract, kdt, prototypes, k=k)

				print("Segmentation with graph matching...")
				from functools import partial
				from distances import euclidean_distance, parallel_distance_computation
				distance = partial(parallel_distance_computation, distance=bundles_distances_mam)
				estimated_bundle_idx = graph_matching(example_tract, static_tractogram[superset_idx], distance)
				estimated_tract = static_tractogram[estimated_bundle_idx]
				print("Lenght estimated tract: %s" %len(estimated_tract))
				true_tract_filename = '%s/sub-%s/sub-%s_%s_tract.trk' %(trk_tracts_dir_wma, sub, sub, tract_name)
				true_tract = nib.streamlines.load(true_tract_filename)
				true_tract = true_tract.streamlines
				true_tract_res = resample_tractogram(true_tract, step_size=0.625)
				true_tract = true_tract_res

				print("Computing the DSC value.")
				DSC = compute_dsc(estimated_tract, true_tract)	
				print("The DSC value is %s" %DSC)
				DSC_results[ss,tt,ee] = DSC

	np.save('DSC_gm_wmaR', DSC_results)
