from __future__ import division
import csv
import math
import subprocess
from glob import glob
import os 
import shutil
import pretty_midi
import numpy as np

def xml_score_match(xml, score):
    nakamura_c = './MusicXMLToMIDIAlign.sh'
    subprocess.call([nakamura_c, xml, score])

def make_corresp(score, perform, align_c):
    subprocess.call([align_c, score, perform])

def save_cleaned_midi(
    filepath, savename=None, no_vel=None, no_pedal=None, save=True):
    filename = os.path.basename(filepath).split('.')[0]
    midi = pretty_midi.PrettyMIDI(filepath)
    midi_new = pretty_midi.PrettyMIDI(resolution=1000, initial_tempo=120) # new midi object
    inst_new = pretty_midi.Instrument(0) # new instrument object
    min_pitch, max_pitch = 21, 108
    orig_note_num = 0
    for inst in midi.instruments: # existing object from perform midi
        for note in inst.notes:
            if note.pitch >= min_pitch and note.pitch <= max_pitch:
                inst_new.notes.append(note)
        for cc in inst.control_changes:
            inst_new.control_changes.append(cc)
        orig_note_num += len(inst.notes)
    new_note_num = len(inst_new.notes)
    # append new instrument
    midi_new.instruments.append(inst_new)
    midi_new.remove_invalid_notes()
    # in case of removing velocity/pedals
    for track in midi_new.instruments:
        if no_vel == True:
            for i in range(len(track.notes)):
                track.notes[i].velocity = 64
        if no_pedal == True:
            track.control_changes = list()
    if save == True:
        midi_new.write(savename)
        return None
    elif save == False:
        return midi_new

    print("{}: {}/{} notes saved --> plain vel: {}".format(
        filename, new_note_num, orig_note_num, no_vel))

def copy_scores_for_players(parent_path, groups):
    xml_files = np.asarray(sorted(glob(os.path.join(parent_path, '*.plain.xml'))))
    score_files = np.asarray(sorted(glob(os.path.join(parent_path, '*.plain.mid'))))

    for group in groups:
        g_name = group.split("/")[-2].split("_")[0]
        perform_files = sorted(glob(os.path.join(group, "*.mid")))
        for file in perform_files:
            pl_name = '.'.join(os.path.basename(file).split(".")[:-3])
            # p_name = pl_name.split("_")[0]
            xml_file = [xml_ for xml_ in xml_files 
                if '.'.join(os.path.basename(xml_).split('.')[:-2]) in pl_name][0]
            score_file = [score_ for score_ in score_files 
                if '.'.join(os.path.basename(score_).split('.')[:-2]) in pl_name][0]

            # print(os.path.basename(file))
            # print(xml_file)
            # print(score_file)
            # print()

            shutil.copy(xml_file, os.path.join(
                group, "{}.plain.xml".format(pl_name)))
            shutil.copy(score_file, os.path.join(
                group, "{}.plain.mid".format(pl_name)))
            print("saved xml/score file for {}/{}".format(g_name, pl_name))
            
def copy_files_for_players(groups):
    aligntool = '/home/seungyeon/Piano/gen_task/parse_xml_midi/AlignmentTool_v2/'
    align_c = os.path.join(aligntool, 'MIDIToMIDIAlign.sh')
    code_c = os.path.join(aligntool, 'Code')
    programs_c = os.path.join(aligntool, 'Programs')

    for g in groups: # amateur / professional
        align_copy = os.path.join(g, 'MIDIToMIDIAlign.sh')
        programs_copy = os.path.join(g, 'Programs')
        code_copy = os.path.join(g, 'Code')
        # copy files for every group
        if not os.path.exists(align_copy):
            shutil.copy(align_c, align_copy)
        if not os.path.exists(programs_copy):
            shutil.copytree(programs_c, programs_copy)
        if not os.path.exists(code_copy):
            shutil.copytree(code_c, code_copy)
        perform_mids = sorted(glob(os.path.join(g, '*.perform.aligned.mid')))
        score_mids = sorted(glob(os.path.join(g, '*.plain.mid')))
        score_xmls = sorted(glob(os.path.join(g, '*.plain.xml')))
        for pm, sm, sx in zip(perform_mids, score_mids, score_xmls):
            pm_name = os.path.basename(pm).split('.')[0]
            sm_name = os.path.basename(sm).split('.')[0]
            sx_name = os.path.basename(sx).split('.')[0]
            assert pm_name == sm_name == sx_name
            # save cleaned midi 
            score_savename = sm
            perform_savename = os.path.join(os.path.dirname(pm), "{}.perform.aligned.cleaned.mid".format(pm_name))
            if not os.path.exists(perform_savename):
                save_cleaned_midi(sm, score_savename, no_vel=True, no_pedal=False)
                save_cleaned_midi(pm, perform_savename, no_vel=False, no_pedal=False)

                names = pm.split('/')
                print('copied files for {}: {}'.format(names[-2], pm_name))    
                print()

def copy_alignment_tools(tool_path, savepath):
    # aligntool = '/workspace/Piano/gen_task/parse_xml_midi/AlignmentTool_v2/'
    align_c = os.path.join(tool_path, 'MIDIToMIDIAlign.sh')
    code_c = os.path.join(tool_path, 'Code')
    programs_c = os.path.join(tool_path, 'Programs')

    align_copy = os.path.join(savepath, 'MIDIToMIDIAlign.sh')
    programs_copy = os.path.join(savepath, 'Programs')
    code_copy = os.path.join(savepath, 'Code')
    # copy files for every group
    if not os.path.exists(align_copy):
        shutil.copy(align_c, align_copy)
    if not os.path.exists(programs_copy):
        shutil.copytree(programs_c, programs_copy)
    if not os.path.exists(code_copy):
        shutil.copytree(code_c, code_copy)

def score_perform_matches(groups):
    for g in groups:
        perform_mids = sorted(glob(os.path.join(g, '*.perform.aligned.cleaned.mid')))
        score_mids = sorted(glob(os.path.join(g, '*.plain.mid')))  
        align_c = os.path.join(g, 'MIDIToMIDIAlign.sh')
        os.chdir(g)        
        for pm, sm in zip(perform_mids, score_mids):
            _perform = os.path.basename(pm).split('.')[0]
            _score = os.path.basename(sm).split('.')[0]
            assert _perform == _score
            if not os.path.exists(os.path.join(g, _perform+'.perform.cleaned_corresp.txt')):
                make_corresp(_score+'.plain', _perform+'.perform.aligned.cleaned', './MIDIToMIDIAlign.sh')
                names = pm.split('/')
                print('saved corresp file for {}:{}: player {}'.format(names[-4],names[-3],names[-2]))
                print()

def score_perform_match(perform_path, score_path):
    savepath = os.path.dirname(perform_path)
    _perform = os.path.basename(perform_path).split('.')[0]
    _score = os.path.basename(score_path).split('.')[0]
    score_savename = os.path.join(savepath, "{}.score.cleaned.mid".format(_score))
    perform_savename = os.path.join(savepath, "{}.perform.cleaned.mid".format(_perform))
    save_cleaned_midi(score_path, score_savename, no_vel=True, no_pedal=False)
    save_cleaned_midi(perform_path, perform_savename, no_vel=False, no_pedal=False)
    assert _perform == _score
    if not os.path.exists(os.path.join(savepath, _perform+'.perform.cleaned_corresp.txt')):
        make_corresp(_score+'.score.cleaned', _perform+'.perform.cleaned', './MIDIToMIDIAlign.sh')
        # erase all resulted files but corresp file
        else_txt = glob(os.path.join(savepath, '*[!_corresp].txt'.format(_perform)))
        for file_ in else_txt:
            os.remove(file_)
        os.remove(os.path.join(savepath, _perform+'.perform.cleaned_spr.txt'))
        os.remove(os.path.join(savepath, _score+'.score.cleaned_spr.txt'))
        os.remove(_score+'.score.cleaned.mid')
        os.remove(_perform+'.perform.cleaned.mid')
        print('saved corresp file for {}'.format(_perform))
        print()


if __name__ == "__main__":
    # copy_files_for_players()
    score_perform_matches()





