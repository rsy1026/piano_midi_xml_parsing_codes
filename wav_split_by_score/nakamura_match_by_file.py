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

def save_corresp_file(perform, score, tool_path, save_path):
    # copy if cannot access to alignment tools
    copy_alignment_tools(tool_path, save_path)
    # get filenames 
    _perform = '.'.join(os.path.basename(perform).split('.')[:-1])
    _score = '.'.join(os.path.basename(score).split('.')[:-1])
    perform_savename = os.path.join(save_path, "{}.cleaned.mid".format(_perform))
    score_savename = os.path.join(save_path, "{}.cleaned.mid".format(_score))
    p_name = os.path.basename(perform).split('.')[0]
    s_name = os.path.basename(score).split('.')[0]
    assert p_name == s_name # make sure same filenames
    # temporally save cleaned midi
    get_cleaned_midi(perform, perform_savename, no_vel=False, no_pedal=False)
    get_cleaned_midi(score, score_savename, no_vel=True, no_pedal=True)
    # save corresp file
    corresp_path = os.path.join(save_path, '{}.cleaned_corresp.txt'.format(_perform))
    if not os.path.exists(corresp_path):
        os.chdir(os.path.dirname(perform))
        make_corresp(_score+".cleaned", _perform+".cleaned", './MIDIToMIDIAlign.sh')
        # erase all resulted files but corresp file
        else_txt = glob('./*[!_corresp].txt'.format(_perform))
        for file_ in else_txt:
            os.remove(file_)
        os.remove('./{}.cleaned_spr.txt'.format(_perform))
        os.remove('./{}.cleaned_spr.txt'.format(_score))
        os.remove('./{}.cleaned.mid'.format(_perform))
        os.remove('./{}.cleaned.mid'.format(_score))
        print('saved corresp file for {}'.format(_perform))
        print()
    
    return corresp_path






