#!/usr/bin/env python

# @Authors: Lukas Burget, Mireia Diez, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, mireia@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# The recipe consists in doing Agglomerative Hierachical Clustering on
# x-vectors in a first step. Then, Variational Bayes HMM over x-vectors
# is applied using the AHC output as args.initialization.
#
# A detailed analysis of this approach is presented in
# M. Diez, L. Burget, F. Landini, S. Wang, J. \v{C}ernock\'{y}
# Optimizing Bayesian HMM based x-vector clustering for the second DIHARD speech
# diarization challenge, ICASSP 2020
# A more thorough description and study of the VB-HMM with eigen-voice priors
# approach for diarization is presented in
# M. Diez, L. Burget, F. Landini, J. \v{C}ernock\'{y}
# Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors,
# IEEE Transactions on Audio, Speech and Language Processing, 2019
# 
# TODO: Add new paper

import argparse
import logging
import os
import itertools

import h5py
import kaldi_io
import numpy as np
from scipy.special import softmax
from scipy.linalg import eigh

from diarization_lib import read_xvector_timing_dict, l2_norm, cos_similarity, twoGMMcalib_lin, AHC, \
    merge_adjacent_labels, mkdir_p
from kaldi_utils import read_plda
from VB_diarization import VB_diarization
from helpers.x_vec import extract_xvec_normalized
from helpers.scoring import kaldi_ivector_plda_scoring_dense_NvsM
from io_operations.io_writers import arr_to_ctm
from io_operations.io_loaders import normalize_tuple_s, load_diar_rttm_unchanged
import scipy.io.wavfile as wav
from io_operations.custom import create_dir_if_not_exist
from extract_xvec_from_audio_raw import xvec_extract
import sys
from scipy.spatial import distance
from helpers.logging import *

def write_output(fp, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
                 f'<NA> <NA> {label + 1} <NA> <NA>{os.linesep}')


def find_speakers(sim):
    most_segments = np.argsort(sim[:, 1], axis=0)[::-1]
    client = (0, 0, -1)
    therapist = (0, 0, -1)

    def enough_segments(candidate, original):
        return sim[candidate][1] > original[1] / 4

    if np.max(sim[:, 0]) <= 0:
        therapist = np.argmax(sim[:, 0])
        for index in most_segments:
            if index == therapist:
                continue
            if sim[index][0] < client[0] and enough_segments(index, client):
                client = (sim[index][0], sim[index][1], index)
        return therapist, client[2]
    elif np.min(sim[:, 0]) >= 0:
        client = np.argmin(sim[:, 0])
        for index in most_segments:
            if index == client:
                continue
            if sim[index][0] > therapist[0] and enough_segments(index, therapist):
                therapist = (sim[index][0], sim[index][1], index)
        return therapist[2], client
    for index in most_segments:
        if sim[index][0] < client[0] and enough_segments(index, client):
            client = (sim[index][0], sim[index][1], index)
        if sim[index][0] > therapist[0] and enough_segments(index, therapist):
            therapist = (sim[index][0], sim[index][1], index)
    return therapist[2], client[2]


def reject_outliers(data, m=2):
    if len(data) < 50:
        return data
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def extract_speaker_embeddings(root_path, labels, filename, segs_dict):
    logger.info('Extracting embeddings for each cluster...')
    speakers_path = os.path.join(root_path, 'speakers')
    create_dir_if_not_exist(speakers_path)
    wav_path = os.path.join(root_path, 'audios', f'{filename}.wav')
    sampling_rate, audio = wav.read(wav_path)

    start, end = segs_dict[file_name][1].T
    starts, ends, out_labels = merge_adjacent_labels(start, end, labels)
    speakers = np.unique(out_labels)
    for index, speaker in enumerate(speakers):
        speaker_segments = out_labels == speaker
        s_starts, s_ends = starts[speaker_segments], ends[speaker_segments]
        logger.debug(
            f'Cluster {index}, number of segments: {np.sum(speaker_segments)}')
        speaker_audio = np.hstack(
            [audio[int(s * sampling_rate): int(e * sampling_rate)] for (s, e) in zip(s_starts, s_ends)])
        speaker_wav_path = os.path.join(speakers_path, f'{speaker}.wav')
        wav.write(speaker_wav_path, sampling_rate, speaker_audio)
        xvec_extract(speaker_wav_path, os.path.dirname(sys.argv[0]), os.path.join(speakers_path, f'{speaker}'),
                     seg_len=144, seg_jump=24)
        logger.info(f'Embeddings for cluster {index} extracted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', required=True, type=str, choices=['AHC', 'AHC+VB', 'random_5'],
                        help='AHC for using only AHC or AHC+VB for VB-HMM after AHC initilization or random_5 for running 5 random initializations for VBx and picking the best per-ELBO', )
    parser.add_argument('--out-rttm-dir', required=True, type=str, help='Directory to store output rttm files')
    parser.add_argument('--xvec-ark-file', required=True, type=str,
                        help='Kaldi ark file with x-vectors from one or more input recordings. '
                             'Attention: all x-vectors from one recording must be in one ark file')
    parser.add_argument('--segments-file', required=True, type=str,
                        help='File with x-vector timing info (see diarization_lib.read_xvector_timing_dict)')
    parser.add_argument('--xvec-transform', required=True, type=str,
                        help='path to x-vector transformation h5 file')
    parser.add_argument('--plda-file', required=True, type=str,
                        help='File with PLDA model in Kaldi format used for AHC and VB-HMM x-vector clustering')
    parser.add_argument('--threshold', required=True, type=float, help='args.threshold (bias) used for AHC')
    parser.add_argument('--lda-dim', required=True, type=int,
                        help='For VB-HMM, x-vectors are reduced to this dimensionality using LDA')
    parser.add_argument('--Fa', required=True, type=float,
                        help='Parameter of VB-HMM (see VB_diarization.VB_diarization)')
    parser.add_argument('--Fb', required=True, type=float,
                        help='Parameter of VB-HMM (see VB_diarization.VB_diarization)')
    parser.add_argument('--loopP', required=True, type=float,
                        help='Parameter of VB-HMM (see VB_diarization.VB_diarization)')
    parser.add_argument('--init-smoothing', required=False, type=float, default=5.0,
                        help='AHC produces hard assignments of x-vetors to speakers. These are "smoothed" to soft '
                             'assignments as the args.initialization for VB-HMM. This parameter controls the amount of'
                             ' smoothing. Not so important, high value (e.g. 10) is OK  => keeping hard assigment')
    parser.add_argument('--output-2nd', required=False, type=bool, default=False,
                        help='Output also second most likely speaker of VB-HMM')
    parser.add_argument('--therapist_template', required=False, type=str,
                        help='Therapist x-vec template file')
    parser.add_argument('--sid_threshold', required=False, type=float,
                        help='SID threshold')
    parser.add_argument('--out_ctm', required=False, type=str,
                        help='Output ctm path')

    args = parser.parse_args()
    assert 0 <= args.loopP <= 1, f'Expecting loopP between 0 and 1, got {args.loopP} instead.'

    # segments file with x-vector timing information
    segs_dict = read_xvector_timing_dict(args.segments_file)

    kaldi_plda = read_plda(args.plda_file)
    plda_mu, plda_tr, plda_psi = kaldi_plda
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    # Open ark file with x-vectors and in each iteration of the following for-loop
    # read a batch of x-vectors corresponding to one recording
    arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file)
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0])  # group xvectors in ark by recording name
    for file_name, segs in recit:
        logger.info(f'Clustering segments...')
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        with h5py.File(args.xvec_transform, 'r') as f:
            mean1 = np.array(f['mean1'])
            mean2 = np.array(f['mean2'])
            lda = np.array(f['lda'])
            x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

        if args.init == 'AHC' or args.init.endswith('VB') or args.init.startswith('random_'):
            if args.init.startswith('AHC'):
                # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise
                # similarities between all x-vectors)
                scr_mx = cos_similarity(x)
                # Figure out utterance specific args.threshold for AHC.
                thr, junk = twoGMMcalib_lin(scr_mx.ravel())
                # output "labels" is an integer vector of speaker (cluster) ids
                labels1st = AHC(scr_mx, thr + args.threshold)
            if args.init.endswith('VB'):
                # Smooth the hard labels obtained from AHC to soft assignments
                # of x-vectors to speakers
                qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
                qinit[range(len(labels1st)), labels1st] = 1.0
                qinit = softmax(qinit * args.init_smoothing, axis=1)
                fea = (x - plda_mu).dot(plda_tr.T)[:, :args.lda_dim]
                # Use VB-HMM for x-vector clustering. Instead of i-vector extractor model, we use PLDA
                # => GMM with only 1 component, V derived accross-class covariance,
                # and iE is inverse within-class covariance (i.e. identity)
                sm = np.zeros(args.lda_dim)
                siE = np.ones(args.lda_dim)
                sV = np.sqrt(plda_psi[:args.lda_dim])
                q, sp, L = VB_diarization(
                    fea, sm, np.diag(siE), np.diag(sV),
                    pi=None, gamma=qinit, maxSpeakers=qinit.shape[1],
                    maxIters=40, epsilon=1e-6,
                    loopProb=args.loopP, Fa=args.Fa, Fb=args.Fb)

                labels1st = np.argsort(-q, axis=1)[:, 0]
                if q.shape[1] > 1:
                    labels2nd = np.argsort(-q, axis=1)[:, 1]
            if args.init.startswith("random_"):
                MAX_SPKS = 10
                prev_L = -float('inf')
                random_iterations = int(args.init.split('_')[1])
                np.random.seed(3)  # for reproducibility
                for _ in range(random_iterations):
                    q_init = np.random.normal(size=(x.shape[0], MAX_SPKS), loc=0.5, scale=0.01)
                    q_init = softmax(q_init * args.init_smoothing, axis=1)
                    fea = (x - plda_mu).dot(plda_tr.T)[:, :args.lda_dim]
                    sm = np.zeros(args.lda_dim)
                    siE = np.ones(args.lda_dim)
                    sV = np.sqrt(plda_psi[:args.lda_dim])
                    q_tmp, sp, L = VB_diarization(
                        fea, sm, np.diag(siE), np.diag(sV),
                        pi=None, gamma=q_init, maxSpeakers=q_init.shape[1],
                        maxIters=40, epsilon=1e-6,
                        loopProb=args.loopP, Fa=args.Fa, Fb=args.Fb)
                    if L[-1][0] > prev_L:
                        prev_L = L[-1][0]
                        q = q_tmp
                labels1st = np.argsort(-q, axis=1)[:, 0]
                if q.shape[1] > 1:
                    labels2nd = np.argsort(-q, axis=1)[:, 1]
        else:
            raise ValueError('Wrong option for args.initialization.')

        uniq_speakers = np.sort(np.unique(labels1st))
        logger.info(f'VBx number of speakers initially detected: {len(uniq_speakers)}')
        if len(uniq_speakers) < 2:
            raise ValueError('Detected only single speaker!')

        # extract_speaker_embeddings(os.path.dirname(args.out_rttm_dir), labels1st, file_name, segs_dict)
        logger.info('Merging speaker clusters...')
        if len(uniq_speakers) >= 2:
            if os.path.exists(args.therapist_template):
                # Load x_vec segment representations

                template_vecs = np.vstack(
                    [extract_xvec_normalized(os.path.join(args.therapist_template, file), [mean1, mean2, lda])[0]
                     for file in os.listdir(args.therapist_template) if file.endswith('.ark')])
                speakers_vec = [
                    (x[np.where(labels1st == speaker), :][0, :], np.argwhere(labels1st == speaker).squeeze())
                    for speaker in uniq_speakers]

                # speakers_vec = [
                #     (extract_xvec_normalized(
                #         os.path.join(os.path.join(os.path.dirname(args.out_rttm_dir)),
                #                      'speakers', f'{speaker}.ark'), [mean1, mean2, lda])[0],
                #      np.argwhere(labels1st == speaker).squeeze())
                #     for speaker in uniq_speakers]

                logger.debug('Cluster similarities:')
                for pair in itertools.combinations(np.arange(0, len(uniq_speakers)), 2):
                    cos_sim = np.mean(1 - distance.cdist(speakers_vec[pair[0]][0], speakers_vec[pair[1]][0]))
                    x_vecs = np.append(speakers_vec[pair[0]][0], speakers_vec[pair[1]][0], axis=0)
                    plda_sim = kaldi_ivector_plda_scoring_dense_NvsM(kaldi_plda, x_vecs,
                                                                     template_range=speakers_vec[pair[0]][0].shape[0],
                                                                     pca_dim=x_vecs.shape[
                                                                         1]).flatten()
                    logger.debug(
                        f'Pair: {pair}, cos sim: {cos_sim}, mean plda_score: {np.mean(plda_sim)}, plda outliers rejected: {np.mean(reject_outliers(plda_sim))}')

                speakers_count = len(speakers_vec)
                if speakers_count > 2:
                    speakers = np.zeros((speakers_count, 2))
                    # Score calculation
                    logger.info('Recognizing therapist and client...')
                    for speaker in range(speakers_count):
                        x_vectors = np.append(template_vecs, speakers_vec[speaker][0], axis=0)
                        speaker_score = reject_outliers(kaldi_ivector_plda_scoring_dense_NvsM(kaldi_plda, x_vectors,
                                                                                              template_range=
                                                                                              template_vecs.shape[0],
                                                                                              pca_dim=x_vectors.shape[
                                                                                                  1]).flatten())
                        logger.debug(
                            f'Speaker {speaker} cosine sim to template: {np.mean(1 - distance.cdist(template_vecs, speakers_vec[speaker][0]))}')
                        speakers[speaker] = [np.mean(speaker_score), speakers_vec[speaker][0].shape[0]]

                    # Therapist probably not found
                    if np.max(speakers[:, 0]) < args.sid_threshold:
                        logger.error('Therapist not matched correctly')
                    # extract x-vectors and calculate means
                    therapist, client = find_speakers(speakers)
                    logger.debug(
                        f'Similarities to therapist template: {speakers[:, 0]}, number of segments: {speakers[:, 1]}')
                    logger.info(f'Therapist {therapist}, client: {client}')
                    # therapist, client = np.argmax(speakers[:, 0]), np.argmin(speakers[:, 0])
                    therapist_vec, client_vec = speakers_vec[therapist], speakers_vec[client]
                    # Remove elements from array
                    if therapist > client:
                        del speakers_vec[therapist]
                        del speakers_vec[client]
                    else:
                        del speakers_vec[client]
                        del speakers_vec[therapist]
                    speakers_count -= 2
                    # Iterate over until 2 speakers left
                    while speakers_count > 0:
                        # Calculate score with other speakers
                        t_scores = [
                            np.mean(reject_outliers(kaldi_ivector_plda_scoring_dense_NvsM(kaldi_plda,
                                                                                          np.append(therapist_vec[0],
                                                                                                    spk[0], axis=0),
                                                                                          template_range=
                                                                                          therapist_vec[0].shape[0],
                                                                                          pca_dim=
                                                                                          therapist_vec[0].shape[
                                                                                              1]).flatten())) for spk
                            in
                            speakers_vec]
                        c_scores = [
                            np.mean(reject_outliers(kaldi_ivector_plda_scoring_dense_NvsM(kaldi_plda,
                                                                                          np.append(client_vec[0],
                                                                                                    spk[0], axis=0),
                                                                                          template_range=
                                                                                          client_vec[0].shape[0],
                                                                                          pca_dim=client_vec[0].shape[
                                                                                              1]).flatten())) for spk in
                            speakers_vec]

                        # Merge two closest segments
                        t_max, c_max = (np.max(t_scores), np.argmax(t_scores)), (np.max(c_scores), np.argmax(c_scores))
                        logger.debug(f'Similarities to therapist cluster: {t_scores}, client cluster: {c_scores}')
                        if t_max[0] > c_max[0]:
                            logger.debug(f'Merging therapist with cluster {t_max[1]}')
                            therapist_vec = np.append(therapist_vec[0], speakers_vec[t_max[1]][0], axis=0), np.append(
                                therapist_vec[1], speakers_vec[t_max[1]][1])
                            del speakers_vec[t_max[1]]
                        else:
                            logger.debug(f'Merging client with cluster {c_max[1]}')
                            client_vec = np.append(client_vec[0], speakers_vec[c_max[1]][0], axis=0), np.append(
                                client_vec[1],
                                speakers_vec[
                                    c_max[1]][
                                    1])
                            del speakers_vec[c_max[1]]
                        speakers_count = speakers_count - 1
                    speakers_vec.append(therapist_vec)
                    speakers_vec.append(client_vec)

                speakers_count = 2
                speakers = np.zeros((speakers_count, 2))
                # Score calculation

                for speaker in range(speakers_count):
                    x_vectors = np.append(template_vecs, speakers_vec[speaker][0], axis=0)
                    speaker_score = reject_outliers(kaldi_ivector_plda_scoring_dense_NvsM(kaldi_plda, x_vectors,
                                                                                          template_range=
                                                                                          template_vecs.shape[0],
                                                                                          pca_dim=x_vectors.shape[
                                                                                              1]).flatten())
                    speakers[speaker] = [np.mean(speaker_score), speakers_vec[speaker][0].shape[0]]
                logger.debug(f'Similarities of resulting clusters to therapist template: {speakers[:, 0]}')
                labels1st[speakers_vec[0][1]] = np.argmax(speakers[:, 0])
                labels1st[speakers_vec[1][1]] = np.argmin(speakers[:, 0])
            else:
                raise ValueError("Did not found therapist template")
        else:
            raise ValueError("Did not detect second speaker")

        assert (np.all(segs_dict[file_name][0] == np.array(seg_names)))
        start, end = segs_dict[file_name][1].T

        starts, ends, out_labels = merge_adjacent_labels(start, end, labels1st)

        mkdir_p(args.out_rttm_dir)
        rttm_path = os.path.join(args.out_rttm_dir, f'{file_name}.rttm')
        with open(rttm_path, 'w') as fp:
            write_output(fp, out_labels, starts, ends)

        segments, _ = load_diar_rttm_unchanged(rttm_path)
        speaker_tags = ['terapeut', 'klient']
        segments = [normalize_tuple_s([segment[1], segment[2] - segment[1], f'{speaker_tags[segment[0] - 1]}'])
                    for segment in segments]
        arr_to_ctm(args.out_ctm, segments)

    if args.output_2nd and args.init.endswith('VB') and q.shape[1] > 1:
        starts, ends, out_labels2 = merge_adjacent_labels(start, end, labels2nd)
        output_rttm_dir = f'{args.out_rttm_dir}2nd'
        mkdir_p(output_rttm_dir)
        with open(os.path.join(output_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
            write_output(fp, out_labels2, starts, ends)
