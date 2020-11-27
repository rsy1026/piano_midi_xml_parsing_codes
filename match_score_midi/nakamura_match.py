from __future__ import division
import csv
import math
import subprocess
from glob import glob
import os 
import shutil
import pretty_midi
import numpy as np
from parse_utils import get_cleaned_midi

def xml_score_match(xml, score):
    nakamura_c = './MusicXMLToMIDIAlign.sh'
    subprocess.call([nakamura_c, xml, score])

def make_corresp(score, perform, align_c):
    subprocess.call([align_c, score, perform])

def copy_alignment_tools(tool_path, save_path):
    align_c = os.path.join(tool_path, 'MIDIToMIDIAlign.sh')
    code_c = os.path.join(tool_path, 'Code')
    programs_c = os.path.join(tool_path, 'Programs')
    align_copy = os.path.join(save_path, 'MIDIToMIDIAlign.sh')
    programs_copy = os.path.join(save_path, 'Programs')
    code_copy = os.path.join(save_path, 'Code')
    # copy files for every group
    if not os.path.exists(align_copy):
        shutil.copy(align_c, align_copy)
    if not os.path.exists(programs_copy):
        shutil.copytree(programs_c, programs_copy)
    if not os.path.exists(code_copy):
        shutil.copytree(code_c, code_copy)

def copy_scores_for_perform(perform_file, score_files, xml_files, p_name):
    xml_file = [xml_ for xml_ in xml_files if p_name in xml_][0]
    score_file = [score_ for score_ in score_files if p_name in score_][0]
    parent_path = os.path.dirname(perform_file)

    shutil.copy(xml_file, os.path.join(
        parent_path, "{}.plain.xml".format(p_name)))
    shutil.copy(score_file, os.path.join(
        parent_path, "{}.plain.mid".format(p_name)))
    print("saved xml/score file for {}".format(p_name))

def save_corresp_file(perform, score, tool_path, save_path, remove_cleaned=False):
    # get filenames 
    _perform = '.'.join(os.path.basename(perform).split('.')[:-1])
    _score = '.'.join(os.path.basename(score).split('.')[:-1])
    p_name = os.path.basename(perform).split('.')[0]
    s_name = os.path.basename(score).split('.')[0]
    # assert p_name == s_name # make sure same filenames
    corresp_path = os.path.join(save_path, '{}.cleaned_corresp.txt'.format(_perform))

    if not os.path.exists(corresp_path):
        # copy if cannot access to alignment tools
        copy_alignment_tools(tool_path, save_path)
        perform_savename = os.path.join(save_path, "{}.cleaned.mid".format(_perform))
        score_savename = os.path.join(save_path, "{}.cleaned.mid".format(_score))
        # temporally save cleaned midi
        get_cleaned_midi(perform, perform_savename, no_vel=False, no_pedal=True)
        get_cleaned_midi(score, score_savename, no_vel=True, no_pedal=True)
        # save corresp file
        os.chdir(os.path.dirname(perform))
        make_corresp(_score+".cleaned", _perform+".cleaned", './MIDIToMIDIAlign.sh')
        # erase all resulted files but corresp file
        else_txt = glob('./*[!_corresp].txt'.format(_perform))
        for file_ in else_txt:
            os.remove(file_)
        os.remove('./{}.cleaned_spr.txt'.format(_perform))
        os.remove('./{}.cleaned_spr.txt'.format(_score))
        if remove_cleaned is True:
            os.remove('./{}.cleaned.mid'.format(_perform))
            os.remove('./{}.cleaned.mid'.format(_score))
        print('saved corresp file for {}'.format(_perform))
        print()
    
    return corresp_path






