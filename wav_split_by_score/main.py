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

# import matplotlib
# matplotlib.use("agg")
# import matplotlib.pyplot as plt
# import seaborn as sns

import soundfile as sf

from musicxml_parser import MusicXMLDocument
from parse_utils import *
from nakamura_match_by_file import *


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
                 save_dir=None,
                 program_dir=None):

        self.save_dir = save_dir
        self.program_dir = program_dir

    def align_wav_midi(self, wav, pmid):
        # for wav in wavs:
        perform_name = os.path.basename(pmid).split('.')[0]
        save_name = os.path.join(
            self.save_dir, "{}.aligned.mid".format(perform_name))
        # align
        if not os.path.exists(save_name):
            subprocess.call(
                ['sudo', 'python3', 'align_midi.py', 
                wav, pmid, save_name])
        return save_name

    def __call__(self, smid, pmid, xml, wav=None):

        ### PERFORM WAV - PERFORM MIDI ### 
        if wav is not None:
            pmid2 = self.align_wav_midi(wav, pmid)
            print("** aligned wav! **                     ")
        else:
            pmid2 = pmid

        ### SCORE MIDI - PERFORM MIDI ### 
        corresp = save_corresp_file(
            pmid2, smid, self.program_dir, self.save_dir) 
        os.chdir(self.program_dir)
        print("** aligned score midi-perform midi! **     ")

        ### SCORE XML - SCORE MIDI ### 
        # load xml object 
        XMLDocument = MusicXMLDocument(xml)
        # extract all xml/midi score notes 
        xml_parsed = extract_xml_notes(XMLDocument)
        score_parsed = extract_midi_notes(smid) 
        num_score = len(score_parsed)
        # match score xml to score midi
        self.xml_score_pairs = \
            match_XML_to_scoreMIDI(xml_parsed, score_parsed)
        print("** aligned score xml-score midi! **        ")
        
        # check_alignment_with_1d_plot(
        #   xml_parsed, score_parsed, self.xml_score_pairs, self.score_name)

        # match score pairs with perform midi
        perform_parsed = extract_midi_notes(pmid2)
        num_perform = len(perform_parsed)
        corresp_parsed = extract_corresp(corresp, num_score, num_perform)
        self.xml_score_perform_pairs = match_score_to_performMIDI(
            self.xml_score_pairs, corresp_parsed, perform_parsed,
            score_parsed, xml_parsed)   
        print("** aligned score xml-score midi-perform midi! **")    

        return self.xml_score_perform_pairs, pmid2

def get_measure_marker(pair):
    
    first_measure_num = pair[0]['xml_note'][1].measure_number +1
    prev_measure_num = first_measure_num
    marker = dict()

    marker[first_measure_num] = [pair[0]]
    for each_note in pair[1:]:
        xml = each_note['xml_note'][1]
        measure_num = xml.measure_number + 1

        if prev_measure_num == measure_num: # if in same measure
            marker[prev_measure_num].append(each_note)                  
        elif prev_measure_num < measure_num: # if next measure
            marker[measure_num] = [each_note]

        prev_measure_num = measure_num

    return marker

def split_wav_by_measure(
    wav_path, mid_path, marker, save_path, split_part_only=None,
    fade_in=1e-4, fade_out=1e-3, dtype="float32", subtype="PCM_32"):

    wav, sr = sf.read(wav_path, dtype=dtype) # stereo
    mid = extract_midi_notes(mid_path)

    f_name = os.path.basename(wav_path).split('.')[0]
    max_key = np.max(list(marker.keys()))

    # split all measures:
    if split_part_only is None: 
        print("splitting all measures...")

        for measure in marker:
            same_measure_midi = list()
            same_measure_wav = list()
            notes = marker[measure]
            all_onsets = [n["perform_midi"][1].start \
                for n in notes if n["perform_midi"] is not None]
            first_onset = np.min(all_onsets)
            try:
                next_notes = marker[measure+1]
                all_next_onsets = [n["perform_midi"][1].start \
                    for n in next_notes if n["perform_midi"] is not None]
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

            first_sample = int(np.round(first_onset / (1/sr)))
            next_sample = int(np.round(next_measure_onset / (1/sr)))
            same_measure_wav = wav[first_sample:next_sample]

            # fade-in and fade-out
            fade_in_len = int(sr * fade_in)
            fade_out_len = int(sr * fade_out)
            fade_wav = fade_in_out(same_measure_wav, fade_in_len=fade_in_len, fade_out_len=fade_out_len)
            # save splitted audio 
            sf.write(os.path.join(
                save_path, '{}.measure{}.wav'.format(f_name, measure)), 
                fade_wav, sr, subtype=subtype)

    elif split_part_only is not None:
        start_measure, end_measure = [int(m) for m in split_part_only]
        print("splitting measures from {} to {}...".format(
            start_measure, end_measure))

        same_measure_midi = list()
        same_measure_wav = list()
        notes = marker[start_measure]
        all_onsets = [n["perform_midi"][1].start \
            for n in notes if n["perform_midi"] is not None]
        first_onset = np.min(all_onsets)
        try:
            next_notes = marker[end_measure+1]
            all_next_onsets = [n["perform_midi"][1].start \
                for n in next_notes if n["perform_midi"] is not None]
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
            save_path, "{}.measure{}-{}.mid".format(f_name, start_measure, end_measure)))

        first_sample = int(np.round(first_onset / (1/sr)))
        next_sample = int(np.round(next_measure_onset / (1/sr)))
        same_measure_wav = wav[first_sample:next_sample]

        # fade-in and fade-out
        fade_in_len = int(sr * fade_in)
        fade_out_len = int(sr * fade_out)
        fade_wav = fade_in_out(same_measure_wav, fade_in_len=fade_in_len, fade_out_len=fade_out_len)
        # save splitted audio 
        sf.write(os.path.join(
            save_path, '{}.measure{}-{}.wav'.format(f_name, start_measure, end_measure)), 
            fade_wav, sr, subtype=subtype)

def fade_in_out(
    wav, fade_in_len=None, fade_out_len=None):
    # wav is stereo
    new_wav = np.copy(wav)
    factor = 0
    # fade in
    for ind, sample in enumerate(new_wav):
        if ind <= fade_in_len:
            left = sample[0] * factor
            right = sample[1] * factor
            factor = (np.exp(ind*1e-3)-1)/(np.exp(fade_in_len*1e-3)-1)
            new_wav[ind,:] = [left, right]
        else:
            break
    # fade out
    factor = 0
    for ind, sample in enumerate(reversed(new_wav)):
        if ind <= fade_out_len:
            left = sample[0] * factor
            right = sample[1] * factor
            factor = (np.exp(ind*1e-3)-1)/(np.exp(fade_out_len*1e-3)-1)
            new_wav[-(ind+1),:] = [left, right]
        else:
            break
    return new_wav

def main(wav_paths=None, 
         score_paths=None, 
         perform_paths=None, 
         xml_paths=None,
         save_dir=None, 
         program_dir=None,
         target_measure="all"):

    wav_paths = sorted(wav_paths)
    score_paths = sorted(score_paths)
    perform_paths = sorted(perform_paths)
    xml_paths = sorted(xml_paths)

    match = XML_SCORE_PERFORM_MATCH(
        save_dir=save_dir, program_dir=program_dir)

    for wav, score, perform, xml in zip(
        wav_paths, score_paths, perform_paths, xml_paths):

        # make sure right files to match
        w_name = os.path.basename(wav).split('.')[0]
        s_name = os.path.basename(score).split('.')[0]
        p_name = os.path.basename(perform).split('.')[0]
        x_name = os.path.basename(xml).split('.')[0]
        assert w_name == s_name == p_name == x_name

        ### MATCH XML-SCORE-PERFORM(WAV) ###
        pair, perform2 = match(smid=score, pmid=perform, xml=xml, wav=wav)
        
        # sort pair based on xml notes' order 
        pair = [p for p in pair if p["xml_note"] is not None]
        pair_ = sorted(pair, key=lambda x: x["xml_note"][0])
        marker = get_measure_marker(pair_)

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
            split_part_only = None
        else:
            split_part_only = target_measure

        split_wav_by_measure(
            wav_path=wav, mid_path=perform2, marker=marker, save_path=save_dir, 
            split_part_only=split_part_only,
            fade_in=1e-4, fade_out=1e-3, dtype="float32", subtype="PCM_32")
        


if __name__ == "__main__":
    '''
    * DEFAULT: split into all measures
    
        python3 main.py /home/user/data /home/user/savedir

    * To split particular measures: 

        python3 main.py /home/user/data /home/user/savedir start_measure end_measure
    
    '''
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    target_measure = "all"

    if len(sys.argv) > 3:
        target_measure = [sys.argv[3], sys.argv[4]]

    # make new directory for saving data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # for debugging
    # data_dir = "/home/rsy/Dropbox/RSY/Piano/git_piano_parsing/wav_split_by_score/test"
    # save_dir = "/home/rsy/Dropbox/RSY/Piano/git_piano_parsing/wav_split_by_score/test_result"
    # target_measure = [1, 4]
    
    wav_paths = sorted(glob(os.path.join(data_dir, "*.wav")))
    score_paths = sorted(glob(os.path.join(data_dir, "*.score.mid")))
    perform_paths = sorted(glob(os.path.join(data_dir, "*.perform.mid")))
    xml_paths = sorted(glob(os.path.join(data_dir, "*.musicxml")))    
    program_dir = os.getcwd()

    # main func
    main(wav_paths=wav_paths, 
         score_paths=score_paths, 
         perform_paths=perform_paths,
         xml_paths=xml_paths, 
         save_dir=save_dir, 
         program_dir=program_dir,
         target_measure=target_measure)




