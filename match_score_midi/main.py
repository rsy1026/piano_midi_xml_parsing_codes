import os
import sys
sys.setrecursionlimit(100000)
import numpy as np
from glob import glob
from fractions import Fraction
import pretty_midi
import csv
import time
import shutil
from decimal import Decimal

# import matplotlib
# matplotlib.use("agg")
# import matplotlib.pyplot as plt
# import seaborn as sns

# import soundfile as sf

from musicxml_parser import MusicXMLDocument
from parse_utils import *
from nakamura_match import *


'''
* The current code includes several steps:
    - should be prepared with perform WAV, score XML, score MIDI, and perform MIDI

    1-1. perform WAV - perform MIDI --> midi file("*_aligned.mid")
        - temporally align performance audio with performance MIDI
        - FUNCTIONS:
            - align_wav_midi()
            - pretty_midi/align_midi.py 
                (https://github.com/craffel/pretty-midi/blob/master/examples/align_midi.py)

    1-2. score MIDI - perform MIDI --> corresp file("*_corresp.text") 
        - match score-performance MIDI with Nakamura algorithm 
        - https://midialignment.github.io/demo.html (Nakamura et al., 2017)
        - FUNCTIONS:
            - save_corresp_files() 

    2. score XML - score MIDI --> var "xml_score_pairs" **IMPERFECT**
        - rule-based note-by-note matching between score XML and score MIDI 
        - FUNCTIONS:
            - match_XML_to_scoreMIDI()
    
    3. score XML- score MIDI - perform MIDI --> var "xml_score_perform_pairs" **IMPERFECT**
        - rule-based note-by-note matching with "xml_score_pairs" and corresp file("*_corresp.txt")
        - FUNCTIONS: 
            - match_score_to_performMIDI()

    4. SPLIT perform WAV by measures --> splitted wav file("*.measure*.wav")
        - split wav according to xml_score_perform_pairs 
        - can split all measures at once & split particular range of measures 
        - FUNCTIONS:
            - split_wav_by_measure()

'''

class XML_SCORE_PERFORM_MATCH(object):
    def __init__(self, 
                 current_dir=None,
                 save_dir=None,
                 program_dir=None):

        self.current_dir = current_dir
        self.save_dir = save_dir
        self.program_dir = program_dir

    def align_wav_midi(self, wav, pmid, name=None):
        if name is None:
            name = self.perform_name
        elif name is not None:
            name = name
        # for wav in wavs:
        save_name = os.path.join(
            self.save_dir, "{}.aligned.mid".format(name))
        # align
        if not os.path.exists(save_name):
            subprocess.call(
                ['python3', 'align_midi.py', wav, pmid, save_name])
        return save_name

    def __call__(self, smid, pmid, xml, wav=None, name=None):

        ### PERFORM WAV - PERFORM MIDI ### 
        self.perform_name = name

        if wav is not None:
            pmid2 = self.align_wav_midi(wav, pmid)
            print("** aligned wav! **                          ")
        else:
            pmid2 = pmid

        ### SCORE MIDI - PERFORM MIDI ### 
        corresp = save_corresp_file(
            pmid2, smid, self.program_dir, self.save_dir) 
        os.chdir(self.current_dir)
        print("** aligned score midi-perform midi! **          ")
        assert os.path.exists(corresp)

        ### SCORE XML - SCORE MIDI ### 
        # load xml object 
        XMLDocument = MusicXMLDocument(xml)
        # extract all xml/midi score notes 
        xml_parsed = extract_xml_notes(XMLDocument)
        score_parsed, _ = extract_midi_notes(smid, clean=True) 
        num_score = len(score_parsed)
        # match score xml to score midi
        xml_score_pairs = \
            match_XML_to_scoreMIDI_plain(xml_parsed, score_parsed)
        print("** aligned score xml-score midi! **             ")
        
        check_alignment_with_1d_plot(
          xml_parsed, score_parsed, xml_score_pairs, self.perform_name)

        # match score pairs with perform midi
        perform_parsed, _ = extract_midi_notes(pmid2, clean=False, no_pedal=True)
        num_perform = len(perform_parsed)
        corresp_parsed = extract_corresp(corresp, num_score, num_perform)
        xml_score_perform_pairs = match_score_to_performMIDI(
            xml_score_pairs, corresp_parsed, perform_parsed, score_parsed, xml_parsed)   
        print("** aligned score xml-score midi-perform midi! **")    

        return xml_score_perform_pairs, pmid2


def split_wav_by_measure(
    wav_path, mid_path, pair, save_path, split_by_measure=None, split_by_note=None,
    fade_in=1e-4, fade_out=1e-3, dtype="float32", subtype="PCM_32"):

    if wav_path is not None:
        wav, sr = sf.read(wav_path, dtype=dtype) # stereo
    else:
        wav = None

    mid, ccs = extract_midi_notes(mid_path, clean=False, no_pedal=False)
    # sort pair based on xml notes' order 
    pair = [p for p in pair if p["xml_note"] is not None]
    pair_ = sorted(pair, key=lambda x: x["xml_note"][0])
    pair_score = sorted(pair, key=lambda x: x["score_midi"][0])
    marker = get_measure_marker(pair_)
    start_measure, end_measure = None, None
    start_note, end_note = None, None

    f_name = '.'.join(os.path.basename(mid_path).split('.')[:1])
    f_name2 = '.'.join(os.path.basename(mid_path).split('.')[1:-1])
    max_key = np.max(list(marker.keys()))

    # split all measures:
    if split_by_measure is None and split_by_note is None: 
        print("splitting all measures...")

        for measure in marker:
            same_measure_midi = list()
            same_measure_wav = list()
            notes = marker[measure]
            all_onsets = [n["perform_midi"][1].start \
                for n in notes if n["perform_midi"] is not None]
            first_onset = np.min(all_onsets)
            try:
                all_next_onsets = list()
                add = 0
                while len(all_next_onsets) == 0:
                    next_notes = marker[measure+1+add]
                    all_next_onsets = [n["perform_midi"][1].start \
                        for n in next_notes if n["perform_midi"] is not None]
                    add += 1
                next_measure_onset = np.min(all_next_onsets)

            except KeyError:
                if measure == max_key:
                    next_measure_onset = len(wav) / sr
                else:
                    raise AssertionError 

            # split midi 
            for note in mid:
              if note.start >= first_onset and note.start < next_measure_onset:
                  same_measure_midi.append(note)
              else:
                  continue
            measure_midi = make_midi_start_zero(same_measure_midi)
            # save splitted midi
            save_new_midi(measure_midi, new_midi_path=os.path.join(
                save_path, "{}.measure{}.mid".format(f_name, measure+1)))

            if wav is not None:
                first_sample = int(np.round(first_onset / (1/sr)))
                next_sample = int(np.round(next_measure_onset / (1/sr)))
                same_measure_wav = wav[first_sample:next_sample]

                # fade-in and fade-out
                if start_measure == 1:
                    fade_in_len = None
                else:
                    fade_in_len = int(sr * fade_in)
                fade_out_len = int(sr * fade_out)
                fade_wav = fade_in_out(same_measure_wav, fade_in_len=fade_in_len, fade_out_len=fade_out_len)
                # save splitted audio 
                sf.write(os.path.join(
                    save_path, '{}.measure{}.wav'.format(f_name, measure)), 
                    fade_wav, sr, subtype=subtype)

    elif split_by_measure is not None or split_by_note is not None:
        if split_by_measure is not None:
            start_measure, end_measure = [int(m) for m in split_by_measure]
            notes = marker[start_measure]
            all_onsets = [n["perform_midi"][1].start \
                for n in notes if n["perform_midi"] is not None]
            first_onset = np.min(all_onsets)
            try:
                all_next_onsets = list()
                add = 0
                while len(all_next_onsets) == 0:
                    next_notes = marker[end_measure+1+add]
                    all_next_onsets = [n["perform_midi"][1].start \
                        for n in next_notes if n["perform_midi"] is not None]
                    add += 1
                next_measure_onset = np.min(all_next_onsets) 

            except KeyError:
                if measure == max_key:
                    next_measure_onset = len(wav) / sr
                else:
                    raise AssertionError 

            print("splitting measures from {} to {}...".format(
                start_measure, end_measure))

        if split_by_note is not None:
            start_note, end_note = [int(m) for m in split_by_note]
            all_onsets = [n["perform_midi"][1].start \
                for n in pair_score if n["score_midi"][1].start==pair_score[start_note]["score_midi"][1].start \
                and n["perform_midi"] is not None]
            first_onset = np.min(all_onsets)
            all_next_onsets = list()
            add = 0
            while len(all_next_onsets) == 0:
                next_note = pair_score[end_note+1+add]
                assert next_note["score_midi"][1].start > pair_score[end_note]["score_midi"][1].start
                all_next_onsets = [n["perform_midi"][1].start \
                    for n in pair_score if n["score_midi"][1].start==next_note["score_midi"][1].start]
                add += 1
            next_measure_onset = np.min(all_next_onsets)

            print("splitting notes from {} to {}...".format(
                    start_note, end_note))

        same_measure_midi = list()
        same_measure_wav = list()
        same_measure_cc = list()

        # split midi 
        first_onset_ = quantize(float(Decimal(str(first_onset))), unit=0.00005)
        next_measure_onset_ = quantize(float(Decimal(str(next_measure_onset))), unit=0.00005)
        # first_onset_ = first_onset 
        # next_measure_onset_ = next_measure_onset
        for note in mid:
          if note.start >= first_onset_ and note.start < next_measure_onset_:
              same_measure_midi.append(note)
          else:
              continue
        if len(ccs) > 0:
            for cc in ccs:
              if cc.time >= first_onset_ and cc.time < next_measure_onset_:
                  same_measure_cc.append(cc)
              else:
                  continue   
        else: 
            same_measure_cc = None         

        if start_measure == 1 or start_note == 1:
            measure_midi = same_measure_midi
        else: 
            measure_midi = make_midi_start_zero(same_measure_midi)

        # save splitted midi
        # new_midi_path = os.path.join(
        #     save_dir, "{}.measure{}-{}.mid".format(f_name, start_measure, end_measure))
        new_midi_path = os.path.join(
            save_dir, "{}.short.{}.mid".format(f_name, f_name2))
        save_new_midi(measure_midi, ccs=same_measure_cc, 
            new_midi_path=new_midi_path, start_zero=False)
        print("saved splitted mid for {}".format(f_name))

        if wav is not None:
            first_sample = int(np.round(first_onset / (1/sr)))
            next_sample = int(np.round(next_measure_onset / (1/sr)))
            same_measure_wav = wav[first_sample:next_sample]
            # print(same_measure_midi)
            print("RESULT: splitting from {} to {}...".format(start_measure, end_measure+add-1))
            print("first_onset: {:.3f} / next_measure_onset: {:.3f}".format(first_onset, next_measure_onset))
            print("samples length: {} / non-zero: {}".format(len(same_measure_wav), np.sum(np.abs(same_measure_wav))>0))
            print(save_dir)

            # fade-in and fade-out
            if start_measure == 1:
                fade_in_len = None
            else:
                fade_in_len = int(sr * fade_in)
            fade_out_len = int(sr * fade_out)
            fade_wav = fade_in_out(same_measure_wav, fade_in_len=fade_in_len, fade_out_len=fade_out_len)
            # save splitted audio 
            sf.write(os.path.join(
                save_dir, '{}.measure{}-{}.wav'.format(f_name, start_measure, end_measure)), 
                fade_wav, sr, subtype=subtype)
            print("saved splitted audio for {}".format(f_name))


def main(wav_paths=None, 
         score_paths=None, 
         perform_paths=None, 
         xml_paths=None,
         save_dir=None, 
         program_dir=None,
         target_measure=None,
         target_note=None):

    if wav_paths is None:
        wav_paths = [None] * len(perform_paths)
    elif wav_paths is not None:
        wav_paths = sorted(wav_paths)
    score_paths = sorted(score_paths)
    perform_paths = sorted(perform_paths)
    xml_paths = sorted(xml_paths)

    match = XML_SCORE_PERFORM_MATCH(
        save_dir=save_dir, program_dir=program_dir)

    for wav, score, perform, xml in zip(
        wav_paths, score_paths, perform_paths, xml_paths):

        # wav, score, perform, xml = wav_paths[0], score_paths[0], perform_paths[0], xml_paths[0]

        # make sure right files to match
        s_name = '.'.join(os.path.basename(score).split('.')[:-2])
        p_name = '.'.join(os.path.basename(perform).split('.')[:-2])
        x_name = '.'.join(os.path.basename(xml).split('.')[:-2])
        if wav is None:
            assert s_name == p_name == x_name
        elif wav is not None:
            w_name = '.'.join(os.path.basename(wav).split('.')[:-2])
            assert w_name == s_name == p_name == x_name

        pair_name = os.path.join(os.path.dirname(perform), 
            "{}.xml_score_perform_pairs.npy".format(p_name))

        if target_measure is None and target_note is None:
            if not os.path.exists(pair_name):
                pair, perform2 = match(smid=score, pmid=perform, xml=xml, wav=wav, name=p_name)
                np.save(pair_name, pair)

        elif target_measure is not None or target_note is not None:
            if not os.path.exists(pair_name):
                pair, perform2 = match(smid=score, pmid=perform, xml=xml, wav=wav, name=p_name)
            else:
                pair = np.load(pair_name, allow_pickle=True).tolist()
                perform2 = os.path.join(os.path.dirname(perform), 
                    "{}.perform.aligned.cleaned.mid".format(p_name))
                wav2 = os.path.join(os.path.dirname(perform), 
                    "{}.perform.wav".format(p_name))
                if not os.path.exists(perform2) and not os.path.exists(wav2):
                    perform2 = os.path.join(os.path.dirname(perform), 
                        "{}.perform.cleaned.mid".format(p_name))

            ### split wav by measure ###
            '''
            * To determine dtype & subtype: 
            --> find out original type of audio
                --> terminal command: ffmpeg -i ***.wav
                --> find line containing a term forming like "pcm_f32le" 
                --> above term indicates "PCM 32-bit floating-point little-endian"
                --> dtype(when load wav): "float32" / subtype(when save wav): "PCM_32"

            * Fade_in, fade_out parameter are in millisec
            --> default: fade_in = 0.1ms / fade_out = 1ms 
            --> this is for avoiding tick sounds (due to discontinuity of splitted waveform)
            '''

            if target_measure == "all":
                split_by_measure = None
            else:
                split_by_measure = target_measure

            split_by_note = target_note

            split_wav_by_measure(
                wav_path=wav, mid_path=perform2, pair=pair, save_path=save_dir, 
                split_by_measure=split_by_measure, split_by_note=split_by_note,
                fade_in=1e-4, fade_out=1e-3, dtype="float32", subtype="PCM_32")
        


if __name__ == "__main__":
    '''
    * DEFAULT: split into all measures
    
        python3 main.py /home/user/data /home/user/savedir

    * To split particular measures: 

        python3 main.py /home/user/data /home/user/savedir start_measure end_measure
    
    * measure number starts with 1 
    * unfinished measure(very first measure with shorter length) is counted
    '''
    # data_dir = sys.argv[1]
    # save_dir = sys.argv[2]
    # target_measure = "all"

    # if len(sys.argv) > 3:
        # target_measure = [sys.argv[3], sys.argv[4]]

    # make new directory for saving data
    # if not os.path.exists(save_dir):
        # os.makedirs(save_dir)
    
    # for debugging
    # data_dir = "/home/rsy/Dropbox/RSY/Piano/git_piano_parsing/wav_split_by_score/test"
    # save_dir = "/home/rsy/Dropbox/RSY/Piano/git_piano_parsing/wav_split_by_score/test_result"
    # target_measure = [1, 4]

    
    #---------------------------------------------------------------------------------------------#

    data_dir = "/home/seungyeon/Piano/sarah/recording_data"
    # groups = sorted(glob(os.path.join(data_dir, "*_all/")))
    groups = sorted(glob(os.path.join(data_dir, "*_add/")))
    # group = groups[0]
    for group in groups:
        g_name = group.split('/')[-2]
        
        perform_paths = sorted(glob(os.path.join(group, "scale_*.perform.mid")))
        score_paths = sorted(glob(os.path.join(group, "scale_*.plain.mid")))
        xml_paths = sorted(glob(os.path.join(group, "scale_*.plain.xml"))) 
        # wav_paths = sorted(glob(os.path.join(group, "*.perform.wav")))
        wav_paths = None 
        # save_dir = group
        save_dir = os.path.join(data_dir, "{}_scale".format(g_name))

        if wav_paths is None:
            wav_paths_ = perform_paths
        else:
            wav_paths_ = wav_paths
        assert len(score_paths) == len(xml_paths) == len(perform_paths) == len(wav_paths_)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        program_dir = os.getcwd()

        # main func
        main(wav_paths=wav_paths, 
             score_paths=score_paths, 
             perform_paths=perform_paths,
             xml_paths=xml_paths, 
             save_dir=save_dir, 
             program_dir=program_dir,
             target_note=[1,113])


    # measures = os.path.join(data_dir, "measures.txt")
    # with open(measures, "r") as txt_file:
    #     lines = txt_file.readlines()
    #     pieces = list()
    #     for line in lines:
    #         pieces.append(line.split("\n")[0].split(" "))

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #     for piece in pieces[1:]:
    #         p_name = piece[0][:-1]
    #         # start_measure = int(piece[1][:-1])
    #         # end_measure = int(piece[2])
    #         score_paths = sorted(glob(os.path.join(data_dir, "score", "{}*.mid".format(p_name))))
    #         xml_paths = sorted(glob(os.path.join(data_dir, "musicxml", "{}*.musicxml".format(p_name))))

    #         for score, xml in zip(score_paths, xml_paths):
    #             m_name = '.'.join(os.path.basename(score).split(".")[:-1])
    #             perform_paths = sorted(glob(os.path.join(data_dir, "perform", "{}.*.mid".format(m_name))))
    #             wav_paths = sorted(glob(os.path.join(data_dir, "wav", "{}.*.wav".format(m_name))))
    #             for perform, wav in zip(perform_paths, wav_paths):
    #                 pl_name = '.'.join(os.path.basename(perform).split(".")[:-1])
    #                 new_perform = os.path.join(save_dir, "{}.perform.mid".format(pl_name))
    #                 new_score = os.path.join(save_dir, "{}.score.mid".format(pl_name))
    #                 new_xml = os.path.join(save_dir, "{}.musicxml".format(pl_name))
    #                 new_wav = os.path.join(save_dir, os.path.basename(wav))
    #                 shutil.copy(perform, new_perform)
    #                 shutil.copy(score, new_score)
    #                 shutil.copy(xml, new_xml)
    #                 shutil.copy(wav, new_wav)

    # for piece in pieces[10:11]:
    #     p_name = piece[0][:-1]
    #     start_measure = int(piece[1][:-1])
    #     end_measure = int(piece[2])
    #     score_paths = sorted(glob(os.path.join(data_dir, "parsing_space", "{}*.score.mid".format(p_name))))
    #     xml_paths = sorted(glob(os.path.join(data_dir, "parsing_space", "{}*.musicxml".format(p_name))))
    #     perform_paths = sorted(glob(os.path.join(data_dir, "parsing_space", "{}*.perform.mid".format(p_name))))
    #     wav_paths = sorted(glob(os.path.join(data_dir, "parsing_space", "{}*.wav".format(p_name))))

    #     assert len(score_paths) == len(xml_paths) == len(perform_paths) == len(wav_paths)

    #     program_dir = os.getcwd()

    #     # main func
    #     main(wav_paths=wav_paths, 
    #          score_paths=score_paths, 
    #          perform_paths=perform_paths,
    #          xml_paths=xml_paths, 
    #          save_dir=save_dir, 
    #          program_dir=program_dir,
    #          target_measure=[start_measure, end_measure])





