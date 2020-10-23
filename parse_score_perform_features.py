from musicxml_parser import MusicXMLDocument
from parse_utils import *
from nakamura_match import *

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

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, mode
import pandas as pd

from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation

dc = getcontext()
dc.prec = 6
dc.rounding = ROUND_HALF_UP

'''
* About This code: 
	- originally from Jaejun Lee(SNU, jjlee0721@gmail.com)
	- modified by Seungyeon Rhyu(SNU, rsy1026@gmail.com)

** version 2: (2019.05~2019.11)

** version 3: (2019.12~)
	- modified for new experiments for score midi-to-performance midi

-----------------------------------------------------------------------------------------

* The current code includes several steps:
	- should be prepared with score XML, score MIDI, and performance MIDI
	1. score-performance matching: --> corresp file
		- score-performance MIDI matching with Nakamura algorithm (Nakamura et al., 2017) 
	2. XML-score matching: --> xml-score pairs
		- rule-based note-by-note matching  
	3. XML-score-performance matching: --> xml-score-perform pairs
		- rule-based note-by-note matching with corresp file 
	4. parse score features: --> score condition input
		- note-by-note parsing according to paired musicXML and score MIDI
	5. parse performance features: --> performance feature output
		- note-by-note parsing according to paired performance	

'''


def search(dirname):
	""" 
	* Function to search 4 kinds of files in 'dirname'
		- xml(score): 'xml_list'
		- midi(score): 'score_midi_list'
		- corresp file(score-perform): 'corresp_list'
		- midi(perform): 'perform_midi_list'
	"""
	# initialize lists 
	xml_list = dict()
	score_midi_list = dict()
	corresp_list = dict()
	perform_midi_list = dict()

	# collect directories 
	categs = sorted(glob(os.path.join(dirname, "*/"))) # category ex) Ballade
	for c in categs:
		c_name = c.split('/')[-2] # get category name
		xml_list[c_name] = dict()
		score_midi_list[c_name] = dict()
		corresp_list[c_name] = dict()
		perform_midi_list[c_name] = dict()
		pieces = sorted(glob(os.path.join(c, "*/"))) # piece ex) (Ballade) No.1
		for p in pieces: 
			p_name = p.split('/')[-2] # get piece name
			players = sorted(glob(os.path.join(p, "*/"))) # player 3x) (Ballade No.1) player1
			# get each path of xml, score, performance files
			xml_path = os.path.join(p, "musicxml_cleaned_plain.musicxml")
			# assign paths to corresponding piece category
			xml_list[c_name][p_name] = xml_path
			score_midi_list[c_name][p_name] = dict()
			corresp_list[c_name][p_name] = dict()
			perform_midi_list[c_name][p_name] = dict()
			for pl in players:
				pl_name = pl.split('/')[-2]
				score_path = glob(os.path.join(pl, "score_plain.cleaned.mid"))[0]
				corresp_path = glob(os.path.join(pl, '*.cleaned_corresp.txt'))[0]
				perform_path = glob(os.path.join(pl, '[!score_plain]*.cleaned.mid'))[0]
				score_midi_list[c_name][p_name][pl_name] = score_path
				corresp_list[c_name][p_name][pl_name] = corresp_path
				perform_midi_list[c_name][p_name][pl_name] = perform_path
	return xml_list, score_midi_list, corresp_list, perform_midi_list

def save_matched_files():
	# PARENT DIRECTORY
	dirname = '/home/rsy/Dropbox/RSY/Piano/data/chopin_maestro/original'
	# get directory lists
	xml_list, score_midi_list, \
	corresp_list, perform_midi_list = search(dirname)
	
	# ########### for debugging #############
	# categ = sorted(xml_list)[0]
	# piece = sorted(xml_list[categ])[0] # 57

	# categ = sorted(xml_list)[1]
	# piece = sorted(xml_list[categ])[-4] # 25-5

	# categ = sorted(xml_list)[-2]
	# piece = sorted(xml_list[categ])[-1] # 31-2

	# categ = sorted(xml_list)[1]
	# piece = sorted(xml_list[categ])[4] # 10-3	

	# categ = sorted(xml_list)[-1]
	# piece = sorted(xml_list[categ])[0] # 58-1	
	# #######################################
	
	# start matching
	for categ in sorted(corresp_list): 
		for piece in sorted(corresp_list[categ]):
			# load xml and score for each piece 
			xml = xml_list[categ][piece]
			score = score_midi_list[categ][piece]['1'] # since all same scores

			if not os.path.exists(os.path.join(
				os.path.dirname(xml), "1/xml_score_perform_pairs.npy", "---")):
				# load xml object 
				XMLDocument = MusicXMLDocument(xml)
				# extract all xml/midi score notes 
				xml_parsed = extract_xml_notes(XMLDocument)
				score_parsed = extract_midi_notes(score)	
				num_score = len(score_parsed)

				if len(xml_parsed) != len(score_parsed):
					print(categ, piece)
					print(len(xml_parsed), len(score_parsed))
					print()
				
				# match xml to score midi
				xml_score_pairs = match_xml_to_scoreMIDI_plain(xml_parsed, score_parsed)
				
				check_alignment_with_1d_plot(
					xml_parsed, score_parsed, xml_score_pairs, categ, piece)

				for player in sorted(corresp_list[categ][piece]):
					perform = perform_midi_list[categ][piece][player]
					corresp = corresp_list[categ][piece][player]
					player_path = os.path.dirname(perform) 
					parsed_path = os.path.join(player_path, "xml_score_perform_pairs.npy")
					if not os.path.exists(parsed_path+"---"):
						perform_parsed = extract_midi_notes(perform)
						num_perform = len(perform_parsed)
						corresp_parsed = extract_corresp(corresp, num_score, num_perform)
						xml_score_perform_pairs = match_score_to_performMIDI(
							xml_score_pairs, corresp_parsed, perform_parsed,
							score_parsed, xml_parsed)		

						# save parsed final pairs
						np.save(parsed_path, xml_score_perform_pairs)

						print("saved parsed xml objects for {}:{}: player_{}".format(categ, piece, player))

def make_corresp_file(parent_dir, perform, score):
	tool_path = "/workspace/Piano/gen_task/parse_xml_midi/AlignmentTool_v2"
	if not os.path.exists(os.path.join(parent_dir, 'MIDIToMIDIAlign.sh')):
		copy_alignment_tools(tool_path, parent_dir)
	else:
		pass
	# match score and performance
	score_perform_match(perform, score)

def match_file(xml, score, corresp, perform):

	midi_name = perform.split(".")[0]
	# load xml object 
	XMLDocument = MusicXMLDocument(xml)
	# extract all xml/midi score notes 
	xml_parsed = extract_xml_notes(XMLDocument)
	score_parsed = extract_midi_notes(score)	
	num_score = len(score_parsed)

	if len(xml_parsed) != len(score_parsed):
		print(len(xml_parsed), len(score_parsed))
		raise AssertionError

	# match xml to score midi
	xml_score_pairs = \
		match_XML_to_scoreMIDI_plain(xml_parsed, score_parsed)
	
	# check_alignment_with_1d_plot(
	# 	xml_parsed, score_parsed, xml_score_pairs, categ, piece)

	player_path = os.path.dirname(perform)
	perform_parsed = extract_midi_notes(perform, clean=False, no_pedal=True)
	num_perform = len(perform_parsed)
	corresp_parsed = extract_corresp(corresp, num_score, num_perform)
	xml_score_perform_pairs = match_score_to_performMIDI(
		xml_score_pairs, corresp_parsed, perform_parsed,
		score_parsed, xml_parsed)		

	return xml_score_perform_pairs


def save_features_xml():
	dirname = '/data/chopin_maestro/original'
	categs = sorted(glob(os.path.join(dirname, "*/")))
	
	pc = ['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B']

	# ################## for debugging ###################
	# # etude 10-4
	# c_name = categs[1].split('/')[-2]
	# pieces = sorted(glob(os.path.join(categs[1], "*/")))
	# piece = pieces[5]
	# # scherzo 31-2
	# c_name = categs[-2].split('/')[-2]
	# pieces = sorted(glob(os.path.join(categs[-2], "*/")))
	# piece = pieces[-1]
	# ####################################################

	for categ in categs:
		c_name = categ.split('/')[-2]
		pieces = sorted(glob(os.path.join(categ, "*/")))
		for piece in pieces:
			p_name = piece.split('/')[-2]
			pair_path = os.path.join(piece, 
				'1', "xml_score_perform_pairs.npy")
			pairs = np.load(pair_path).tolist()
			pairs_xml = [p for p in pairs if p['xml_note'] is not None]
			pairs_xml = sorted(pairs_xml, key=lambda x: x['xml_note'][0])

			# PARSE FEATURES
			cond_list = list()
			note_list = list()
			csv_list = list()
			prev_xml_note = None
			prev_xml_measure = None
			for i in range(len(pairs_xml)):
				xml_note = pairs_xml[i]['xml_note'][1]
				xml_measure = pairs_xml[i]['xml_measure']	
				midi_ind = pairs_xml[i]['score_midi'][0]

				# parse features for each note
				parsed_note = XMLFeatures(note=xml_note,
										  measure=xml_measure,
										  prev_note=prev_xml_note,
										  prev_measure=prev_xml_measure,
										  note_ind=i)

				_input = parsed_note._input

				# add parsed feature informations to csv files 
				csv_list.append([])
				csv_list.append(['<{}th note: measure {}>'.format(i, xml_note.measure_number)])
				csv_list.append(['> tempo: {}'.format(parsed_note.tempo)])
				csv_list.append(['> onset: {:4f}'.format(parsed_note.time_position)])
				csv_list.append(['> pitch: {} (pc: {} / octave: {})'.format(
					parsed_note.pitch_name, parsed_note.pitch_class, parsed_note.octave)])
				# csv_list.append(['> pitch/normalized_pitch: {}({})/{} (key: {} {})'.format(
					# parsed_note.pitch_name, parsed_note.pitch, parsed_note.pitch_norm, pc[int(parsed_note.key_final)], parsed_note.mode)])
				csv_list.append(['> type: {}'.format(parsed_note._type)])
				csv_list.append(['> is_dot: {}'.format(parsed_note.is_dot)])
				csv_list.append(['> voice: {}'.format(parsed_note.voice)])
				csv_list.append(["> current_directions: {}".format(
					[d.type['content'] for d in parsed_note.current_directions])])
				csv_list.append(["> dynamics: {}".format(parsed_note.dynamics)])
				csv_list.append(["> is_new_dynamic: {}".format(parsed_note.is_new_dynamics)])
				csv_list.append(["> next_dynamic: {}".format(parsed_note.next_dynamics)])
				csv_list.append(["> same_onset: {}".format(parsed_note.same_onset)])
				csv_list.append(["> is_downbeat: {}".format(parsed_note.is_downbeat)])
				csv_list.append(["> is_grace_note: {}".format(parsed_note.is_grace_note)])
				# csv_list.append(["> ornament: {}".format(parsed_note.ornament)])
				# csv_list.append(["> tuplet: {}".format(parsed_note.tuplet)])
				csv_list.append(["> is_tied: {}".format(parsed_note.is_tied)])	

				csv_list.append(["---------------------------------------------------------------------------"])
				
				csv_list.append(["--> type input: {}".format(_input[1:7])]) # 6 
				csv_list.append(["--> dot input: {}".format(_input[7:9])]) # 2
				csv_list.append(["--> staff input: {}".format(_input[9:11])]) # 2
				csv_list.append(["--> grace note input: {}".format(_input[11:13])]) # 2
				csv_list.append(["--> voice input: {}".format(_input[13:17])]) # 4
				csv_list.append(["--> dynamic input: {}".format(_input[17:23])]) # 6
				csv_list.append(["--> pitch result: {}".format(_input[23:111])]) # 88
				csv_list.append(["--> pitch result 1: pc input: {}".format(_input[111:123])]) # 12
				csv_list.append(["--> pitch result 2: octave input: {}".format(_input[123:131])]) # 8
				# csv_list.append(["--> ornament input: {}".format(_input[42:46])])
				# csv_list.append(["--> tuplet input: {}".format(_input[46:49])])
				# csv_list.append(["--> tied input: {}".format(_input[42:44])])				
				csv_list.append(["--> same onset input: {}".format(_input[131:133])]) # 2
				# csv_list.append(["--> downbeat input: {}".format(_input[46:])])
				csv_list.append([])
				csv_list.append([])
				
				# append input list
				cond_list.append([midi_ind, _input])

				# update previous measure number and onset time 
				prev_xml_note = parsed_note # InputFeatures object
				prev_xml_measure = parsed_note.measure # xml object

			# save csv file for checking 
			writer = csv.writer(open("./features_check_{}_{}.csv".format(c_name, p_name), 'w'))
			for row in csv_list:
				writer.writerow(row)
			cond_list = np.asarray(cond_list)

			np.save(os.path.join(piece, "cond.npy"), cond_list)
			print()
			print("parsed {}/{} condition input".format(c_name, p_name))
			print()


def save_features_midi():
	# dirname = '/home/rsy/Dropbox/RSY/Piano/data/chopin_cleaned/original'
	# dirname = '/home/rsy/Dropbox/RSY/Piano/data/chopin_maestro/original'
	# dirname = '/home/seungyeon/DATA/seungyeon_files/Piano/chopin_cleaned/original'
	dirname = '/data/chopin_maestro/original'
	categs = sorted(glob(os.path.join(dirname, "*/")))
	all_inputs = list()
	all_outputs = list()

	for categ in categs:
		c_name = categ.split('/')[-2]
		pieces = sorted(glob(os.path.join(categ, "*/")))
		for piece in pieces:
			# p = dirname_
			p_name = piece.split('/')[-2]
			players = sorted(glob(os.path.join(piece, '*/')))
			for player in players:
				pl_name = player.split('/')[-2]
				pair_path = os.path.join(player, "xml_score_perform_pairs.npy")
				if not os.path.exists(pair_path):
					continue
				pairs = np.load(pair_path).tolist()
				pairs_score = [p for p in pairs if p['score_midi'] is not None]
				pairs_score = sorted(pairs_score, key=lambda x: x['score_midi'][0])

				prev_onset = 0.
				first_onset_group = list()
				for note in pairs_score:
					score_onset = note['score_midi'][1].start
					if score_onset == prev_onset:
						if note['perform_midi'] is not None:
							first_onset_group.append(note['perform_midi'][1].start)
					elif score_onset > prev_onset:
						break
					prev_onset = score_onset
				if len(first_onset_group) > 0:
					first_onset = np.min(first_onset_group)
				elif len(first_onset_group) == 0:
					first_onset = 0.
					# raise AssertionError

				midi_path = os.path.join(player, 'score_plain.cleaned.mid')
				midi_notes = extract_midi_notes(midi_path)
				input_list = list()
				output_list = list()
				onset_list = list()
				feature_list = list()

				prev_note = None
				for i in range(len(pairs_score)):
					# assign each pair
					note = pairs_score[i]

					# scale first onset/offset to 0
					if note['perform_midi'] is not None:
						note['perform_midi'][1].start -= first_onset
						note['perform_midi'][1].end -= first_onset

					# parse features for each note
					parsed_note = MIDIFeatures(note=note,
											   prev_note=prev_note,
											   note_ind=i)

					if prev_note is None:
						prev_mean_onset = -parsed_note.dur_16th
						prev_mean_onset2 = -parsed_note.dur_16th

					if parsed_note.is_same_onset is False:

						same_onset = [parsed_note.perform_onset]
						same_onset_ind = [i]
						prev_note_ = note						
						# gather same onset groups
						for j in range(i+1, len(pairs_score)):
							next_note = pairs_score[j]
							if next_note['perform_midi'] is not None:
								next_perform_onset = next_note['perform_midi'][1].start - first_onset
							elif next_note['perform_midi'] is None:
								next_perform_onset = None
							if next_note['score_midi'][1].start == prev_note_['score_midi'][1].start:
								same_onset.append(next_perform_onset)
								same_onset_ind.append(j)
								prev_note_ = next_note
							else:
								break
						
						same_onset_values = [float(v) for v in same_onset if v is not None]
						if len(same_onset_values) == 0:
							mean_onset = None
							mean_onset2 = None
						elif len(same_onset_values) > 0:
							mean_onset = np.min(same_onset_values)
							mean_onset2 = np.mean(same_onset_values)

						# update onset standard for computing ioi
						onset_for_ioi = prev_mean_onset
						onset_for_ioi2 = prev_mean_onset2

						# round onset
						if onset_for_ioi is not None:
							onset_for_ioi = onset_for_ioi
							onset_for_ioi2 = onset_for_ioi2
						
					elif parsed_note.is_same_onset is True: # repeat
						mean_onset = mean_onset
						mean_onset2 = mean_onset2
						same_onset_values = same_onset_values
						onset_for_ioi = onset_for_ioi
						onset_for_ioi2 = onset_for_ioi2		

					onset_list.append(onset_for_ioi)					

					# print(onset_for_ioi2, mean_onset2)

					_input = parsed_note.get_input_features(onset_for_ioi2)
					_output = parsed_note.get_output_features(onset_for_ioi2, mean_onset2)
					
					# print(_output, parsed_note.is_same_onset)

					input_list.append([i, _input])
					output_list.append([i, _output])
					feature_list.append(parsed_note.ioi_units)
					
					# update previous attributes
					prev_note = parsed_note # MIDIFeatures object
					prev_mean_onset = mean_onset
					prev_mean_onset2 = mean_onset2
					prev_same_onset_values = same_onset_values

				assert len(midi_notes) == len(output_list) == len(input_list)
				input_list = np.array(input_list)
				output_list = np.array(output_list)
				all_inputs.append(input_list)
				all_outputs.append(output_list)
				
				# save outputs
				np.save(os.path.join(player, 'inp.npy'), input_list)
				np.save(os.path.join(player, 'oup.npy'), output_list)
				print("parsed {}/{} output: player {}".format(c_name, p_name, pl_name))

	# np.save("chopin_maestro_all_input.npy", all_inputs)
	# np.save("chopin_maestro_all_output.npy", all_outputs)

def trim_length(midi_notes, sec=None):
	onset_group = list()
	same_onset = [midi_notes[0]]
	prev = -1
	for note in midi_notes[1:]:
		if note.start > prev:
			onset_group.append(same_onset)
			same_onset = [note]
		elif note.start == prev:
			same_onset.append(note)
		prev = note.start
	onset_group.append(same_onset)
	# trim to given seconds long
	sub_notes = list()
	for onset in onset_group:
		if onset[0].start < sec:
			for note in onset:
				sub_notes.append(note)
		else:
			break
	return sub_notes

def trim_length_pairs(pairs, sec=None):
	onset_pairs = group_by_onset(pairs)
	for onset in onset_pairs:
		all_offsets = [n["score_midi"][1].end for n in onset]
		max_offset = np.max(all_offsets)
		if max_offset > sec:
			break 
	min_ind = np.min([n["score_midi"][0] for n in onset])
	return min_ind


def parse_test_cond(pair=None, pair_path=None):
	if pair is not None:
		pairs = pair 
	elif pair is None and pair_path is not None:
		pairs = np.load(pair_path).tolist()

	pairs_xml = [p for p in pairs if p['xml_note'] is not None]
	pairs_xml = sorted(pairs_xml, key=lambda x: x['xml_note'][0])

	# PARSE FEATURES
	cond_list = list()
	note_list = list()
	csv_list = list()
	prev_xml_note = None
	prev_xml_measure = None
	for i in range(len(pairs_xml)):
		xml_note = pairs_xml[i]['xml_note'][1]
		xml_measure = pairs_xml[i]['xml_measure']	
		midi_ind = pairs_xml[i]['score_midi'][0]

		# parse features for each note
		parsed_note = XMLFeatures(note=xml_note,
								  measure=xml_measure,
								  prev_note=prev_xml_note,
								  prev_measure=prev_xml_measure,
								  note_ind=i)

		_input = parsed_note._input
		
		# append input list
		cond_list.append([midi_ind, _input])

		# update previous measure number and onset time 
		prev_xml_note = parsed_note # InputFeatures object
		prev_xml_measure = parsed_note.measure # xml object

	cond_list = np.asarray(cond_list)
	cond_list = sorted(cond_list, key=lambda x: x[0])
	cond = np.asarray([c[1] for c in cond_list])

	return cond


def parse_test_x_features(midi_path, quarter=None, sec=None):

	midi_notes = extract_midi_notes(midi_path)
	
	if quarter is None:
		dur_dict = dict()
		prev_note = None
		for note in midi_notes:
			if prev_note is None:
				prev_note = note
				continue
			onset = Decimal(str(note.start))
			prev_onset = Decimal(str(prev_note.start))
			dur = round(onset - prev_onset, 3)
			try:
				dur_dict[dur] += 1
			except KeyError:
				dur_dict[dur] = 1
			prev_note = note
		quarter_cand = sorted(dur_dict.items(), key=lambda x: x[1], reverse=True)
		for q, _ in quarter_cand:
			if float(q) >= 0.3 and float(q) <= 2.5:
				break
		quarter = float(q)
	
	dur_16th = Decimal(str(quarter)) / 4
	min_dur = dur_16th / 4

	# set start to 0
	first_onset = midi_notes[0].start
	for note in midi_notes:
		note.start -= first_onset
		note.end -= first_onset
		note.start = quantize(note.start, unit=float(min_dur))
		note.end = quantize(note.end, unit=float(min_dur))
	
	# group into onset
	if sec is not None:
		sub_notes = trim_length(midi_notes, sec=sec)
	else:
		sub_notes = midi_notes

	input_list = list()
	prev_note = None
	for i in range(len(sub_notes)):
		# assign each pair
		note = midi_notes[i]
		# parse features for each note
		parsed_note = MIDIFeatures_test(note=note,
								        prev_note=prev_note,
								        note_ind=i,
								        dur_16th=dur_16th)

		if prev_note is None:
			prev_mean_onset = -parsed_note.dur_16th

		if parsed_note.is_same_onset is False:
			mean_onset = parsed_note.score_onset
			# update onset standard for computing ioi
			onset_for_ioi = prev_mean_onset
			
		elif parsed_note.is_same_onset is True: # repeat
			mean_onset = mean_onset
			onset_for_ioi = onset_for_ioi
					
		# print(onset_for_ioi, mean_onset, parsed_note.is_same_onset)

		_input = parsed_note.get_input_features(onset_for_ioi)
		input_list.append(_input)
		
		# update previous attributes
		prev_note = parsed_note # MIDIFeatures object
		prev_mean_onset = mean_onset

	assert len(sub_notes) == len(input_list)
	input_list = np.array(input_list)

	return input_list, sub_notes, quarter

def parse_test_y_features(xml, score, perform, corresp=None):

	# get xml_score_perform_pairs
	parent_dir = os.path.dirname(perform)
	if corresp is None:
		make_corresp_file(parent_dir, perform, score)

	pairs = match_file(xml, score, corresp, perform)

	pairs_score = [p for p in pairs if p['score_midi'] is not None]
	pairs_score = sorted(pairs_score, key=lambda x: x['score_midi'][0])

	prev_onset = 0.
	first_onset_group = list()
	for note in pairs_score:
		score_onset = note['score_midi'][1].start
		if score_onset == prev_onset:
			if note['perform_midi'] is not None:
				first_onset_group.append(note['perform_midi'][1].start)
		elif score_onset > prev_onset:
			break
		prev_onset = score_onset
	if len(first_onset_group) > 0:
		first_onset = np.min(first_onset_group)
	elif len(first_onset_group) == 0:
		first_onset = 0.
		# raise AssertionError

	input_list = list()
	output_list = list()
	onset_list = list()
	feature_list = list()

	prev_note = None
	for i in range(len(pairs_score)):
		# assign each pair
		note = pairs_score[i]

		# scale first onset/offset to 0
		if note['perform_midi'] is not None:
			note['perform_midi'][1].start -= first_onset
			note['perform_midi'][1].end -= first_onset

		# parse features for each note
		parsed_note = MIDIFeatures(note=note,
								   prev_note=prev_note,
								   note_ind=i)

		if prev_note is None:
			prev_mean_onset = -parsed_note.dur_16th
			prev_mean_onset2 = -parsed_note.dur_16th

		if parsed_note.is_same_onset is False:

			same_onset = [parsed_note.perform_onset]
			same_onset_ind = [i]
			prev_note_ = note						
			# gather same onset groups
			for j in range(i+1, len(pairs_score)):
				next_note = pairs_score[j]
				if next_note['perform_midi'] is not None:
					next_perform_onset = next_note['perform_midi'][1].start - first_onset
				elif next_note['perform_midi'] is None:
					next_perform_onset = None
				if next_note['score_midi'][1].start == prev_note_['score_midi'][1].start:
					same_onset.append(next_perform_onset)
					same_onset_ind.append(j)
					prev_note_ = next_note
				else:
					break
			
			same_onset_values = [float(v) for v in same_onset if v is not None]
			if len(same_onset_values) == 0:
				mean_onset = None
				mean_onset2 = None
			elif len(same_onset_values) > 0:
				mean_onset = np.min(same_onset_values)
				mean_onset2 = np.mean(same_onset_values)

			# update onset standard for computing ioi
			onset_for_ioi = prev_mean_onset
			onset_for_ioi2 = prev_mean_onset2

			# round onset
			if onset_for_ioi is not None:
				onset_for_ioi = onset_for_ioi
				onset_for_ioi2 = onset_for_ioi2
			
		elif parsed_note.is_same_onset is True: # repeat
			mean_onset = mean_onset
			mean_onset2 = mean_onset2
			same_onset_values = same_onset_values
			onset_for_ioi = onset_for_ioi
			onset_for_ioi2 = onset_for_ioi2		

		onset_list.append(onset_for_ioi)					

		_input = parsed_note.get_input_features(onset_for_ioi2)
		_output = parsed_note.get_output_features(onset_for_ioi2, mean_onset2)
		
		output_list.append(_output)
		
		# update previous attributes
		prev_note = parsed_note # MIDIFeatures object
		prev_mean_onset = mean_onset
		prev_mean_onset2 = mean_onset2
		prev_same_onset_values = same_onset_values

	assert len(pairs_score) == len(output_list)
	output_list = np.array(output_list)
	
	# clip
	vel = np.clip(output_list[:,0], 24, 104)
	loc = np.clip(output_list[:,1], -0.05, 0.05)
	dur = np.clip(output_list[:,2], 0.125, 8)
	ioi = np.clip(output_list[:,3], 0.25, 4)
	output_list[:,0] = vel
	output_list[:,1] = loc
	output_list[:,2] = dur
	output_list[:,3] = ioi

	return output_list

def feature_distribution():
	dirname = '/home/rsy/Dropbox/RSY/Piano/data/chopin_maestro/original'
	# dirname = '/home/seungyeon/DATA/seungyeon_files/Piano/chopin_cleaned/original'
	categs = sorted(glob(os.path.join(dirname, "*/")))

	vel_list = list()
	ioi_list = list()
	dur_list = list()
	tempo_group_list = list()
	dynamic_group_list = list()

	for categ in categs:
		pieces = sorted(glob(os.path.join(categ, "*/")))
		for piece in pieces:
			players = sorted(glob(os.path.join(piece, "*/")))
			if not os.path.exists(os.path.join(piece, "cond.npy")):
				continue
			cond = np.load(os.path.join(piece, "cond.npy"))
			for player in players:
				inp = np.load(os.path.join(player, "inp.npy"))
				oup = np.load(os.path.join(player, "oup.npy"))
				pairs = np.load(os.path.join(player, "xml_score_perform_pairs.npy"))
				pairs = [p for p in pairs if p['score_midi'] is not None]
				pairs = sorted(pairs, key=lambda x: x['score_midi'][0])
				for i, features in enumerate(oup):
					cond_ind = pairs[i]['xml_note'][0]
					score_ind = pairs[i]['score_midi'][0]
					assert i == score_ind
					each_cond = cond[cond_ind]
					tempo_group = str(np.where(each_cond[1][0] >= 160, 'fast', 'slow'))
					dynamic_group = str(np.where(np.argmax(each_cond[1][1:]) >= 3, 'forte', 'piano'))
					vel = features[1][0]
					dur = np.log(features[1][1])
					ioi = np.log(features[1][2])
					vel_list.append(vel)
					dur_list.append(dur)
					ioi_list.append(ioi)
					tempo_group_list.append(tempo_group)
					dynamic_group_list.append(dynamic_group)

	# dataframe
	d = {'tempo_group': tempo_group_list,
		 'dynamic_group': dynamic_group_list,
		 'perform_vel': vel_list, 
		 'perform_dur': dur_list, 
		 'perform_ioi': ioi_list}
	df = pd.DataFrame(data=d)

	# heatmap
	plt.figure(figsize=(10,10))
	sns.pairplot(df, hue="tempo_group")
	plt.title("Pair plot of perform features")
	plt.savefig("pairplot_features_by_tempo.png")

	# pearson correlation scatter plot
	p = pearsonr(vel_list, dur_list_out)
	plt.rc('font', size=10)
	plt.figure(figsize=(8,8))
	sns.regplot(x=vel_list, y=dur_list_out, fit_reg=True, color=sns.xkcd_rgb["faded green"], line_kws={'color': 'black'})
	# colors: windows blue, amber, faded green
	plt.xlabel('perform velocity')
	plt.ylabel('perform dur_ratio')
	plt.title("Scatter plot for perform velocity and perform dur ratio\n(r = {})".format(p[0]))
	plt.savefig("scatter_vel_dur.png")		







#----------------------------------------------------------------------------------#

# Class for parsing MIDI features(input/output)
class MIDIFeatures(object):
	def __init__(self, 
				 note=None, 
				 prev_note=None, 
				 note_ind=None,
				 tempo_=None):

		# Inputs
		self.note = note
		self.score_note = note['score_midi'][1]
		self.directions = note['xml_measure'].directions
		if note['perform_midi'] is not None:
			self.perform_note = note['perform_midi'][1] 
		else:
			self.perform_note = None
		self.prev_note = prev_note
		self.note_ind = note_ind

		# Features to parse
		self.prev_velocity = None
		self.prev_score_note = None
		self.is_same_onset = None
		self.score_onset = None
		self.score_offset = None
		self.score_dur = None
		self.score_ioi = None
		self.ioi_units = None
		self.dur_units = None
		self.perform_onset = None
		self.perform_offset = None
		self.perform_dur = None
		self.perform_ioi = None
		self.tempo = tempo_
		self.null_tempo = 160
		self.dur_16th = None
		self.ioi_class = None
		self.dur_class = None
		self.velocity = 0
		self.dur_ratio = 0
		self.ioi_ratio = 0
		self._input = None
		self._output = None


		# Functions		
		if self.prev_note is not None:
			self.get_prev_attributes()

		if self.tempo is None:
			self.get_tempo()
		elif self.tempo is not None:
			pass

		self.get_score_attributes() # onset/offset/dur
		self.get_perform_attributes() # onset/offset/dur

	def get_prev_attributes(self):
		pass

	def get_input_features(self, onset_for_ioi):
		self.get_score_ioi(onset_for_ioi)
		self.get_score_duration()
		self.get_ioi_class()
		self.get_dur_class()
		self.get_pitch()
		self.input_to_vector()
		return self._input	

	def get_output_features(self, onset_for_ioi, mean_onset):
		
		self.get_velocity() # velocity value
		self.get_ioi_ratio2(onset_for_ioi, mean_onset) # ioi ratio
		self.get_local_dev(mean_onset)
		self.get_duration_ratio() # duration ratio 
		self.output_to_vector()
		return self._output

	def input_to_vector(self):
		_score_ioi = np.zeros([11,])
		_score_ioi[self.ioi_class] = 1
		_score_dur = np.zeros([11,])
		_score_dur[self.dur_class] = 1
		_pitch = np.zeros([88,])
		_pitch[self.pitch] = 1
		_same_onset = np.zeros([2,])
		if self.is_same_onset == False:
			_same_onset[0] = 1
		elif self.is_same_onset == True:
			_same_onset[1] = 1
		assert np.sum(_score_ioi) == 1
		assert np.sum(_score_dur) == 1
		assert np.sum(_pitch) == 1
		assert np.sum(_same_onset) == 1

		self._input = np.concatenate([_score_ioi, _score_dur, _pitch, _same_onset], axis=-1)

	def output_to_vector(self):
		_velocity = np.reshape(self.velocity, (1,))
		_local_dev = np.reshape(float(self.local_dev), (1,))
		_dur_ratio = np.reshape(float(self.dur_ratio), (1,))
		_ioi_ratio = np.reshape(float(self.ioi_ratio), (1,))
		
		self._output = np.concatenate([_velocity, _local_dev, _dur_ratio, _ioi_ratio], axis=-1)

	def get_tempo(self):
		if len(self.directions) > 0:
			for direction in self.directions:
				if direction.tempo is not None:
					self.tempo = direction.tempo
				elif direction.tempo is None:
					continue
		elif len(self.directions) == 0:
			pass

		if self.tempo is None:
			if self.prev_note is not None:
				self.tempo = self.prev_note.tempo
			elif self.prev_note is None:
				raise AssertionError
		elif self.tempo is not None:
			pass

		self.tempo = Decimal(self.tempo)
		dur_quarter = Decimal(60) / self.tempo # BPM
		self.dur_16th = dur_quarter / Decimal(4)
		self.dur_16th = round(self.dur_16th, 3)
		self.tempo_ratio = self.tempo / Decimal(self.null_tempo)

	def get_score_attributes(self):
		self.score_onset = Decimal(str(self.score_note.start))
		self.score_offset = Decimal(str(self.score_note.end))
		self.score_dur = self.score_offset - self.score_onset
		self.score_dur = round(self.score_dur, 3)
		assert self.score_dur > 0

		if self.prev_note is None:
			self.is_same_onset = False
		elif self.prev_note is not None:
			if self.score_onset > self.prev_note.score_onset: 
				self.is_same_onset = False
			elif self.score_onset == self.prev_note.score_onset:
				self.is_same_onset = True
	
	def get_perform_attributes(self):
		if self.perform_note is None:
			self.perform_onset = None
			self.perform_offset = None
			self.perform_dur = None

		elif self.perform_note is not None:
			self.perform_onset = Decimal(str(self.perform_note.start))
			self.perform_offset = Decimal(str(self.perform_note.end))
			self.perform_dur = self.perform_offset - self.perform_onset
			self.perform_dur = round(self.perform_dur, 3)
			assert self.perform_dur > 0	

	def get_ioi_class(self):
		'''
		score midi duration based on unit of 16th note
		'''

		self.ioi_units = self.score_ioi / self.dur_16th
		self.ioi_units = float(round(Decimal(str(self.ioi_units)), 1))
		# assign to certain class
		if self.ioi_units < 1.0:
			self.ioi_class = 0 # shorter
		elif self.ioi_units == 1.0:
			self.ioi_class = 1 # 16th
		elif self.ioi_units > 1.0 and self.ioi_units < 2.0:
			self.ioi_class = 2 # 16th+a
		elif self.ioi_units == 2.0:
			self.ioi_class = 3 # 8th
		elif self.ioi_units > 2.0 and self.ioi_units < 4.0:
			self.ioi_class = 4 # 8th+a
		elif self.ioi_units == 4.0:
			self.ioi_class = 5 # quarter
		elif self.ioi_units > 4.0 and self.ioi_units < 8.0:
			self.ioi_class = 6 # quarter+a
		elif self.ioi_units == 8.0:
			self.ioi_class = 7 # half
		elif self.ioi_units > 8.0 and self.ioi_units < 16.0:
			self.ioi_class = 8 # half+a
		elif self.ioi_units == 16.0:
			self.ioi_class = 9 # whole
		elif self.ioi_units > 16.0:
			self.ioi_class = 10 # longer

	def get_dur_class(self):
		'''
		score midi duration based on unit of 16th note
		'''

		self.dur_units = self.score_dur / self.dur_16th
		self.dur_units = float(round(Decimal(str(self.dur_units)), 1))
		# assign to certain class
		if self.dur_units < 1.0:
			self.dur_class = 0 # shorter
		elif self.dur_units == 1.0:
			self.dur_class = 1 # 16th
		elif self.dur_units > 1.0 and self.dur_units < 2.0:
			self.dur_class = 2 # 16th+a
		elif self.dur_units == 2.0:
			self.dur_class = 3 # 8th
		elif self.dur_units > 2.0 and self.dur_units < 4.0:
			self.dur_class = 4 # 8th+a
		elif self.dur_units == 4.0:
			self.dur_class = 5 # quarter
		elif self.dur_units > 4.0 and self.dur_units < 8.0:
			self.dur_class = 6 # quarter+a
		elif self.dur_units == 8.0:
			self.dur_class = 7 # half
		elif self.dur_units > 8.0 and self.dur_units < 16.0:
			self.dur_class = 8 # half+a
		elif self.dur_units == 16.0:
			self.dur_class = 9 # whole
		elif self.dur_units > 16.0:
			self.dur_class = 10 # longer

	def get_score_ioi(self, onset_for_ioi):
		'''
		score midi ioi based on null tempo
		''' 
		if onset_for_ioi is not None:
			self.onset_for_ioi = Decimal(str(onset_for_ioi))
		elif onset_for_ioi is None:
			self.onset_for_ioi = None
		if self.is_same_onset == False:
			if self.prev_note is None:
				self.score_ioi = self.score_onset - self.onset_for_ioi # same for score and perform for only the first note
			elif self.prev_note is not None:
				self.score_ioi = self.score_onset - self.prev_note.score_onset
		elif self.is_same_onset == True:
			self.score_ioi = self.prev_note.score_ioi
		assert self.score_ioi > 0.

		self.score_ioi_norm = round(self.score_ioi * self.tempo_ratio, 3)

	def get_score_duration(self):
		'''
		score midi duration based on null tempo
		'''
		self.score_dur_norm = round(self.score_dur * self.tempo_ratio, 3)

	def get_pitch(self):
		'''
		pitch of score midi
		'''
		midi_num = self.score_note.pitch
		self.pitch = midi_num - 21
		self.pitch_class = np.mod(midi_num, 12) # pitch class
		self.octave = int(midi_num / 12) - 1 # octave

	def get_velocity(self):
		'''
		velocity value: 0~127 / 127
		0: off / 1-16: ppp / 17-32: pp / 33-48: p / 49-64: mp 
		65-80: mf / 81-96: f / 97-112: ff / 113-127: fff
		'''				
		if self.perform_note is not None:
			self.velocity = self.perform_note.velocity

		elif self.perform_note is None:
			if self.prev_note is None: # first note
				self.velocity = 64
			elif self.prev_note is not None:
				self.velocity = self.prev_note.velocity

		# self.q_vel = int(self.velocity // 4) # quantize to 32 classes

	def get_ioi_ratio2(self, onset_for_ioi, mean_onset):
		'''
		ratio of perform midi ioi / score midi(norm) ioi 
		'''
		# get perform ioi and compute ioi ratio
		if onset_for_ioi is not None:
			self.onset_for_ioi = Decimal(str(onset_for_ioi))
		elif onset_for_ioi is None:
			self.onset_for_ioi = None

		if mean_onset is not None:
			self.mean_onset = Decimal(str(mean_onset))
		elif mean_onset is None:
			self.mean_onset = None		

		if self.onset_for_ioi is not None:
			if self.mean_onset is not None:
				self.perform_ioi = self.mean_onset - self.onset_for_ioi
				# ioi value is not 0 
				if self.perform_ioi <= 0.:
					self.perform_ioi = Decimal(str(1e-3))
				# compute ioi ratio
				self.ioi_ratio = self.perform_ioi / self.score_ioi_norm

			elif self.mean_onset is None:
				if self.prev_note is None:
					self.ioi_ratio = 1 / self.tempo_ratio
				elif self.prev_note is not None:
					self.ioi_ratio = self.prev_note.ioi_ratio

		elif self.onset_for_ioi is None:
			if self.prev_note is None:
				self.ioi_ratio = 1 / self.tempo_ratio
			elif self.prev_note is not None:
				self.ioi_ratio = self.prev_note.ioi_ratio 

		try:
			self.ioi_ratio = round(self.ioi_ratio, 4)
		except InvalidOperation:
			print("decimal exception!")
			self.ioi_ratio = self.ioi_ratio
		assert self.ioi_ratio > 0

	def get_local_dev(self, mean_onset):
		'''
		ratio of perform midi ioi / score midi(norm) ioi 
		'''
		# get perform ioi and compute ioi ratio
		if mean_onset is not None:
			self.mean_onset = Decimal(str(mean_onset))
		elif mean_onset is None:
			self.mean_onset = None

		if self.mean_onset is not None:
			if self.perform_note is not None:
				self.local_dev = self.perform_onset - self.mean_onset

			elif self.perform_note is None:
				if self.prev_note is None:
					self.local_dev = Decimal(str(0.))
				elif self.prev_note is not None:
					self.local_dev = self.prev_note.local_dev

		elif self.mean_onset is None:
			if self.prev_note is None:
				self.local_dev = Decimal(str(0.))
			elif self.prev_note is not None:
				self.local_dev = self.prev_note.local_dev 

		try:
			self.local_dev = round(self.local_dev, 4)
		except InvalidOperation:
			print("decimal exception!")
			self.local_dev = self.local_dev

	# def get_ioi_ratio(self, onset_for_ioi):
	# 	'''
	# 	ratio of perform midi ioi / score midi(norm) ioi 
	# 	'''
	# 	# get perform ioi and compute ioi ratio
	# 	if onset_for_ioi is not None:
	# 		self.onset_for_ioi = Decimal(str(onset_for_ioi))
	# 	elif onset_for_ioi is None:
	# 		self.onset_for_ioi = None
	# 	if self.onset_for_ioi is not None:
	# 		if self.perform_note is not None:
	# 			self.perform_ioi = self.perform_onset - self.onset_for_ioi
	# 			# ioi value is not 0 
	# 			if self.perform_ioi <= 0.:
	# 				self.perform_ioi = Decimal(str(1e-3))
	# 			# compute ioi ratio
	# 			self.ioi_ratio = self.perform_ioi / self.score_ioi_norm

	# 		elif self.perform_note is None:
	# 			if self.prev_note is None:
	# 				self.ioi_ratio = 1 / self.tempo_ratio
	# 			elif self.prev_note is not None:
	# 				self.ioi_ratio = self.prev_note.ioi_ratio

	# 	elif self.onset_for_ioi is None:
	# 		if self.prev_note is None:
	# 			self.ioi_ratio = 1 / self.tempo_ratio
	# 		elif self.prev_note is not None:
	# 			self.ioi_ratio = self.prev_note.ioi_ratio 

	# 	self.ioi_ratio = round(self.ioi_ratio, 4)
	# 	assert self.ioi_ratio > 0

	def get_duration_ratio(self):
		'''
		ratio of perform midi duration / score midi duration 
		'''
		if self.perform_dur is not None:
			self.dur_ratio = self.perform_dur / self.score_dur_norm
		
		elif self.perform_dur is None:
			if self.prev_note is None:
				self.dur_ratio = 1 / self.tempo_ratio
			elif self.prev_note is not None:
				self.dur_ratio = self.prev_note.dur_ratio

		try:
			self.dur_ratio = round(self.dur_ratio, 4)
		except InvalidOperation:
			print("decimal exception!")
			self.dur_ratio = self.dur_ratio
		assert self.dur_ratio > 0


class MIDIFeatures_test(object):
	def __init__(self, 
				 note=None, 
				 prev_note=None, 
				 note_ind=None,
				 dur_16th=None):

		# Inputs
		self.note = note
		self.score_note = note
		self.prev_note = prev_note
		self.note_ind = note_ind
		self.dur_16th = round(Decimal(str(dur_16th)), 3)

		# Features to parse
		self.prev_score_note = None
		self.is_same_onset = None
		self.score_onset = None
		self.score_offset = None
		self.score_dur = None
		self.score_ioi = None
		self.ioi_units = None
		self.dur_units = None
		self.ioi_class = None
		self.dur_class = None
		self._input = None

		# Functions		
		if self.prev_note is not None:
			self.get_prev_attributes()

		self.get_score_attributes() # onset/offset/dur

	def get_prev_attributes(self):
		pass

	def get_input_features(self, onset_for_ioi):
		self.get_score_ioi(onset_for_ioi)
		self.get_ioi_class()
		self.get_dur_class()
		self.get_pitch()
		self.input_to_vector()
		return self._input	

	def input_to_vector(self):
		_score_ioi = np.zeros([11,])
		_score_ioi[self.ioi_class] = 1
		_score_dur = np.zeros([11,])
		_score_dur[self.dur_class] = 1
		_pitch = np.zeros([88,])
		_pitch[self.pitch] = 1
		_same_onset = np.zeros([2,])
		if self.is_same_onset == False:
			_same_onset[0] = 1
		elif self.is_same_onset == True:
			_same_onset[1] = 1
		assert np.sum(_score_ioi) == 1
		assert np.sum(_score_dur) == 1
		assert np.sum(_pitch) == 1
		assert np.sum(_same_onset) == 1

		self._input = np.concatenate([_score_ioi, _score_dur, _pitch, _same_onset], axis=-1)

	def get_score_attributes(self):
		self.score_onset = Decimal(str(self.score_note.start))
		self.score_offset = Decimal(str(self.score_note.end))
		self.score_dur = self.score_offset - self.score_onset
		self.score_dur = round(self.score_dur, 3)
		assert self.score_dur > 0

		if self.prev_note is None:
			self.is_same_onset = False
		elif self.prev_note is not None:
			if self.score_onset > self.prev_note.score_onset: 
				self.is_same_onset = False
			elif self.score_onset == self.prev_note.score_onset:
				self.is_same_onset = True

	def get_ioi_class(self):
		'''
		score midi duration based on unit of 16th note
		'''

		self.ioi_units = self.score_ioi / self.dur_16th
		self.ioi_units = float(round(Decimal(str(self.ioi_units)), 1))
		# assign to certain class
		if self.ioi_units < 1.0:
			self.ioi_class = 0 # shorter
		elif self.ioi_units == 1.0:
			self.ioi_class = 1 # 16th
		elif self.ioi_units > 1.0 and self.ioi_units < 2.0:
			self.ioi_class = 2 # 16th+a
		elif self.ioi_units == 2.0:
			self.ioi_class = 3 # 8th
		elif self.ioi_units > 2.0 and self.ioi_units < 4.0:
			self.ioi_class = 4 # 8th+a
		elif self.ioi_units == 4.0:
			self.ioi_class = 5 # quarter
		elif self.ioi_units > 4.0 and self.ioi_units < 8.0:
			self.ioi_class = 6 # quarter+a
		elif self.ioi_units == 8.0:
			self.ioi_class = 7 # half
		elif self.ioi_units > 8.0 and self.ioi_units < 16.0:
			self.ioi_class = 8 # half+a
		elif self.ioi_units == 16.0:
			self.ioi_class = 9 # whole
		elif self.ioi_units > 16.0:
			self.ioi_class = 10 # longer

	def get_dur_class(self):
		'''
		score midi duration based on unit of 16th note
		'''

		self.dur_units = self.score_dur / self.dur_16th
		self.dur_units = float(round(Decimal(str(self.dur_units)), 1))
		# assign to certain class
		if self.dur_units < 1.0:
			self.dur_class = 0 # shorter
		elif self.dur_units == 1.0:
			self.dur_class = 1 # 16th
		elif self.dur_units > 1.0 and self.dur_units < 2.0:
			self.dur_class = 2 # 16th+a
		elif self.dur_units == 2.0:
			self.dur_class = 3 # 8th
		elif self.dur_units > 2.0 and self.dur_units < 4.0:
			self.dur_class = 4 # 8th+a
		elif self.dur_units == 4.0:
			self.dur_class = 5 # quarter
		elif self.dur_units > 4.0 and self.dur_units < 8.0:
			self.dur_class = 6 # quarter+a
		elif self.dur_units == 8.0:
			self.dur_class = 7 # half
		elif self.dur_units > 8.0 and self.dur_units < 16.0:
			self.dur_class = 8 # half+a
		elif self.dur_units == 16.0:
			self.dur_class = 9 # whole
		elif self.dur_units > 16.0:
			self.dur_class = 10 # longer

	def get_score_ioi(self, onset_for_ioi):
		'''
		score midi ioi based on null tempo
		''' 
		if onset_for_ioi is not None:
			self.onset_for_ioi = Decimal(str(onset_for_ioi))
		elif onset_for_ioi is None:
			self.onset_for_ioi = None

		if self.is_same_onset == False:
			if self.prev_note is None:
				self.score_ioi = self.score_onset - self.onset_for_ioi # same for score and perform for only the first note
			elif self.prev_note is not None:
				self.score_ioi = self.score_onset - self.prev_note.score_onset
		elif self.is_same_onset == True:
			self.score_ioi = self.prev_note.score_ioi
		assert self.score_ioi > 0.

	def get_pitch(self):
		'''
		pitch of score midi
		'''
		midi_num = self.score_note.pitch
		self.pitch = midi_num - 21
		self.pitch_class = np.mod(midi_num, 12) # pitch class
		self.octave = int(midi_num / 12) - 1 # octave



class XMLFeatures(object):
	def __init__(self, 
				 note=None, 
				 measure=None,
				 prev_note=None,
				 prev_measure=None,
				 note_ind=None):

		# Inputs
		self.note = note # xml note
		self.measure = measure # xml measure
		self.prev_note = prev_note
		self.prev_measure = prev_measure
		self.note_ind = note_ind

		# Initialize features to parse
		self.tempo = None
		self._type = {'shorter': 0, '16th': 0, 'eighth': 0,
					  'quarter': 0, 'half': 0, 'longer': 0}
		self.is_dot = 0
		self.voice = 0
		self.dynamics = {'pp': 0, 'p': 0, 'mp': 0,
						 'ff': 0, 'f': 0, 'mf': 0}
		# self.global_wedge = {'none': 0, 'cresc': 0, 'dim': 0} 
		# self.local_wedge = {'none': 0, 'cresc': 0, 'dim': 0} 
		self.is_downbeat = 0
		self.is_grace_note = 0
		# self.accent = {'none': 0, 'accent': 0, 'strong_accent': 0}
		# self.staccato = {'none': 0, 'staccato': 0, 'strong_staccato': 0}
		# self.is_arpeggiate = 0
		# self.is_trill = 0
		# self.tempo = {'rit': 0, 'accel': 0, 'a_tempo': 0}
		self.same_onset = {'start': 0, 'cont': 0}
		# self.ornament = {'none': 0, 'trill': 0, 'mordent': 0, 'wavy': 0}
		self.tuplet = {'none': 0, 'start': 0, 'stop': 0} 
		self.is_tied = 0
		self.pitch_name = None
		self.pitch = None
		self.pitch_norm = None
		self.mode = None
		self.key_final = None
		self.pitch_class = None
		self.octave = None
		self._input = None

		# initialize previous note attributes
		self.prev_measure_number = None
		self.prev_time_position = None
		self.prev_dynamics = None
		self.prev_next_dynamics = None
		self.prev_downbeat = None
		self.prev_stem = None
		self.prev_staff = None
		self.prev_is_global = None
		self.prev_is_local = None

		# initialize current note attributes
		self.time_position = None
		self.measure_number = None
		self.x_position = None
		self.y_position = None
		self.stem = None
		self.staff = None
		self.current_directions = list()
		self.is_new_dynamic = None
		self.next_dynamics = None
		# self.is_cresc_global = False
		# self.is_cresc_global_word = False
		# self.is_dim_global = False
		# self.is_dim_global_word = False
		# self.is_cresc_local_1 = False
		# self.is_cresc_local_word_1 = False
		# self.is_dim_local_1 = False
		# self.is_dim_local_word_1 = False
		# self.is_cresc_local_2 = False
		# self.is_cresc_local_word_2 = False
		# self.is_dim_local_2 = False
		# self.is_dim_local_word_2 = False

		# get attributes 
		if self.prev_note is not None:
			self.get_prev_attributes()
		self.get_current_attributes()
		# get input features
		self.get_features()
		# wrap up inputs
		self.features_to_onehot()

		print("parsed {}th note(measure: {})".format(
			self.note_ind, self.measure_number))


	def get_prev_attributes(self):
		self.prev_measure_number = self.prev_note.measure_number
		self.prev_time_position = self.prev_note.time_position
		self.prev_x_position = self.prev_note.x_position
		self.prev_y_position = self.prev_note.y_position
		self.prev_dynamics = self.prev_note.dynamics
		self.prev_next_dynamics = self.prev_note.next_dynamics
		self.prev_stem = self.prev_note.stem
		self.prev_staff = self.prev_note.staff
		self.prev_is_downbeat = self.prev_note.is_downbeat
		self.prev_key_final = self.prev_note.key_final
 
	def get_current_attributes(self):
		self.time_position = self.note.note_duration.time_position
		self.measure_number = self.note.measure_number
		self.x_position = self.note.x_position
		self.y_position = self.note.y_position
		self.stem = self.note.stem
		self.staff = self.note.staff
		# if current measure contains any directions
		if len(self.measure.directions) > 0:
			for direction in self.measure.directions:
				_time_position = direction.time_position
				if _time_position == self.time_position:
					self.current_directions.append(direction)		

	def get_features(self):
		self.get_tempo() # written tempo
		self.get_type() # type of duration 
		self.get_grace_note() # whether grace note
		self.get_voice() # voice number for each note
		self.get_dynamics() # global dynamics (no wedge)
		# self.get_wedge() # global/local wedges (cresc, dim, none)
		self.get_downbeat() # whether first beat of a measure
		# self.get_accent() # whether accent 
		# self.get_staccato() # whether staccato 
		# self.get_arpeggiate() # whether arpeggiate 
		# self.get_fermata() # whether fermata
		# self.get_tempo_change() # whether tempo changes
		self.get_same_onset() # whether in same onset group: start, cont 
		self.get_pitch() # pitch class and octave
		# self.get_ornament() # whether ornaments
		# self.get_tuplet() # whether tuplet
		self.get_tied() # whether tied

	def features_to_onehot(self):
		'''
		Make features into binary vectors
		'''
		# tempo
		_tempo = np.zeros([1,])
		_tempo[0] = self.tempo
		# type
		_type = np.zeros([6,])
		for i, key in enumerate(sorted(self._type)):
			if self._type[key] == 1:
				_type[i] = 1
				break
		# dot
		_dot = np.zeros([2,])
		_dot[self.is_dot] = 1
		# staff
		_staff = np.zeros([2,])
		_staff[self.staff-1] = 1
		# grace note
		_grace = np.zeros([2,])
		_grace[self.is_grace_note] = 1	
		# voice
		_voice = np.zeros([4,])
		_voice[self.voice-1] = 1	
		# dynamics
		_dynamics = np.zeros([6,])
		if self.dynamics['pp'] == 1:
			_dynamics[0] = 1
		elif self.dynamics['p'] == 1:
			_dynamics[1] = 1
		elif self.dynamics['mp'] == 1:
			_dynamics[2] = 1
		elif self.dynamics['mf'] == 1:
			_dynamics[3] = 1
		elif self.dynamics['f'] == 1:
			_dynamics[4] = 1
		elif self.dynamics['ff'] == 1:
			_dynamics[5] = 1
		# downbeat 
		_downbeat = np.zeros([2,])
		_downbeat[self.is_downbeat] = 1		
		# onset
		_same_onset = np.zeros([2,])	
		if self.same_onset['start'] == 1:
			_same_onset[0] = 1
		elif self.same_onset['cont'] == 1:
			_same_onset[1] = 1		
		# pitch class and octave 
		_pitch_class = np.zeros([12,])
		_octave = np.zeros([8,])
		_pitch_class[self.pitch_class] = 1
		_octave[self.octave] = 1
		_pitch2 = np.concatenate([_pitch_class, _octave], axis=-1)
		_pitch = np.zeros([88,])
		_pitch[int(self.pitch-21)] = 1
		# # ornament
		# _ornament = np.zeros([4,])
		# for i, key in enumerate(sorted(self.ornament)):
		# 	if self.ornament[key] == 1:
		# 		_ornament[i] = 1
		# 		break
		# tuplet
		# _tuplet = np.zeros([3,])
		# for i, key in enumerate(sorted(self.tuplet)):
		# 	if self.tuplet[key] == 1:
		# 		_tuplet[i] = 1
		# 		break
		# tied
		_tied = np.zeros([2,])
		_tied[self.is_tied] = 1

		# check if onehot for each feature 
		# assert np.sum(_tempo) == 1
		assert np.sum(_type) == 1
		assert np.sum(_staff) == 1
		assert np.sum(_dot) == 1
		assert np.sum(_grace) == 1
		assert np.sum(_voice) == 1
		if np.sum(_dynamics) != 1:
			print(self.dynamics, _dynamics, self.note)
			raise AssertionError
		assert np.sum(_downbeat) == 1
		assert np.sum(_same_onset) == 1
		assert np.sum(_pitch_class) == 1
		assert np.sum(_octave) == 1
		assert np.sum(_pitch) == 1
		# assert np.sum(_ornament) == 1
		# assert np.sum(_tuplet) == 1
		assert np.sum(_tied) == 1

		# concatenate all features into one vector 
		self._input = np.concatenate(
			[_tempo, _type, _dot, _staff, _grace, _voice, _dynamics, 
			_pitch, _pitch2, _same_onset], axis=-1)
		# self._input = np.concatenate([_tempo, _dynamics], axis=-1)

	def get_tempo(self):
		if len(self.current_directions) > 0:
			for direction in self.current_directions:
				if direction.tempo is not None:
					self.tempo = direction.tempo

		if self.tempo is None:
			if self.prev_note is not None:
				self.tempo = self.prev_note.tempo
			elif self.prev_note is None:
				raise AssertionError

		self.tempo = float(self.tempo)

	def get_type(self):
		'''
		get note type from note.note_duration.type 
		'''
		note_type = self.note.note_duration.type
		shorter_group = ['32nd','64th','128th','256th','512th','1024th']
		longer_group = ['whole','breve','long','maxima']
		if len(note_type) > 0:
			if note_type in shorter_group:
				self._type['shorter'] = 1
			elif note_type == '16th':
				self._type['16th']= 1
			elif note_type == 'eighth':
				self._type['eighth']= 1
			elif note_type == 'quarter':
				self._type['quarter'] = 1
			elif note_type == 'half':
				self._type['half'] = 1
			elif note_type in longer_group:
				self._type['longer'] = 1
		# if note is dotted
		if self.note.note_duration.dots > 0:
			self.is_dot = 1

	def get_grace_note(self):
		'''
		get whether a note is grace note from note.is_grace_note
		'''
		if self.note.is_grace_note is True:
			self.is_grace_note = 1
		elif self.note.is_grace_note is False:
			self.is_grace_note = 0

	def get_voice(self):
		'''
		get voice number for a note from note.voice
		'''
		if self.note.voice <= 4:
			self.voice = self.note.voice
		elif self.note.voice > 4:
			self.voice = self.note.voice - 4

	def get_dynamics(self):
		'''
		get dynamics from measure.directions.type
		* only consider global dynamics (most of dynamics are not local) 
		'''	
		dynamic_list = list(self.dynamics.keys())
		dynamic_candidates = list()

		# parse dynamics within current directions 
		if len(self.current_directions) > 0:
			for direction in self.current_directions:
				'''
						ff (staff 1-above)
				staff 1 ==================
						ff (staff 1-below / staff 2-above)
				staff 2 ==================
						ff (staff 2-below)
				'''
				_staff = int(direction.staff) # staff position of the direction
				_place = direction.placement # whether above/below note.staff
				_dynamic = None
				_next_dynamic = None # in case of "fp" kinds

				if (_staff == 1 and _place == 'below') or \
					(_staff == 2 and _place == 'above'):
				
					content = direction.type["content"]
					# parse dynamics with "dynamic" type
					if direction.type["type"] == "dynamic":
						if content in dynamic_list:
							_dynamic = content 
						else: # other dynamics other than basic ones
							if content == "fp":
								_dynamic = "f" 
								# _dynamic = "p" 
								_next_dynamic = "p"
							elif content == "ffp":
								_dynamic = "ff" 
								# _dynamic = "p" 
								_next_dynamic = "p"
							elif content == "ppp":
								_dynamic = "pp"
							elif content == "fff":
								_dynamic = "ff"
							else:
								# print("** dynamics at {}th note: {}".format(note_ind, content))
								continue

					# parse dynamics with "word" type (ex: f con fuoco)
					elif direction.type["type"] == "words":
						if 'pp ' in str(content) or ' pp' in str(content):
							_dynamic = 'pp'
						elif 'p ' in str(content) or ' p' in str(content):
							_dynamic = 'p'
						elif 'mp ' in str(content) or ' mp' in str(content):
							_dynamic = 'mp'
						elif 'mf ' in str(content) or ' mf' in str(content):
							_dynamic = 'mf'
						elif 'f ' in str(content) or ' f' in str(content):
							_dynamic = 'f'
						elif 'ff ' in str(content) or ' ff' in str(content):
							_dynamic = 'ff'

					if _dynamic is not None:
						dynamic_candidates.append(
							{'dynamic': _dynamic, 'next_dynamic': _next_dynamic})
		# print(dynamic_candidates)

		if len(dynamic_candidates) == 1:
			self.is_new_dynamics = True
			self.dynamics[dynamic_candidates[0]['dynamic']] = 1
			self.next_dynamics = dynamic_candidates[0]['next_dynamic']

		elif len(dynamic_candidates) > 1:
			print("** Global dynamic is more than one:")
			print("- measure number {}".format(self.measure_number))
			print(dynamic_candidates)
			raise AssertionError	

		# if no dynamic is assigned
		elif len(dynamic_candidates) == 0: 
			self.is_new_dynamics = False
			if self.prev_dynamics is not None: 
				if self.prev_next_dynamics is None:
					self.dynamics = self.prev_dynamics
				# in case if "fp" kinds
				elif self.prev_next_dynamics is not None:
					self.dynamics[self.prev_next_dynamics] = 1
			# This case can happen when no dynamics at start
			elif self.prev_dynamics is None: 
				self.dynamics['mp'] = 1

	def get_same_onset(self):
		'''
		see xml and find whether onsets are the same with previous, next notes
		'''
		if self.prev_note is None: # first note:
			self.same_onset['start'] = 1
		elif self.prev_note is not None:
			if self.is_grace_note == 0:
				if self.time_position > self.prev_time_position:
					self.same_onset['start'] = 1
				elif self.time_position == self.prev_time_position:
					self.same_onset['cont'] = 1
			elif self.is_grace_note == 1:
				if self.x_position != self.prev_x_position:
					self.same_onset['start'] = 1
				elif self.x_position == self.prev_x_position:
					self.same_onset['cont'] = 1

	def get_pitch(self):
		'''
		get pitch from note.pitch
		'''
		midi_num = self.note.pitch[1]
		self.pitch = midi_num 
		self.pitch_name = self.note.pitch[0]
		'''
		measure.key_signature.key --> based on fifths
		- -1(F), 0(C), 1(G), D(2), ...
		abs(fifths * 7) % 12 --> tonic
		'''
		if self.measure.key_signature is not None:
			fifths_in_measure = self.measure.key_signature.key
			if fifths_in_measure < 0:
				key = ((fifths_in_measure * 7) % -12) + 12 # if Ab major, "-4"
			elif fifths_in_measure >= 0:
				key = (fifths_in_measure * 7) % 12

			self.mode = self.measure.key_signature.mode # 'major' / 'minor'
			if self.mode == "minor":
				self.key_final = (key - 3 + 12) % 12 # minor 3 below
			elif self.mode == "major":
				self.key_final = key
			
			self.pitch_norm = midi_num - self.key_final # normalize to C major/minor
		
		elif self.measure.key_signature is None:
			self.key_final = self.prev_key_final
			self.pitch_norm = midi_num - self.key_final

		self.pitch_class = np.mod(midi_num, 12) # pitch class
		self.octave = int(midi_num / 12) - 1 # octave
		assert self.pitch_class != None
		assert self.octave != None

	def get_downbeat(self):
		'''
		get measure number of each note and see transition point 
		- notes in same onset group are considered as one event 
		'''
		if self.is_grace_note == 0:
			if self.prev_note is None:
				self.is_downbeat = 1
			elif self.prev_note is not None:
				# if in different onset group
				if self.prev_time_position != self.time_position:
					# new measure
					if self.measure_number != self.prev_measure_number: 
						self.is_downbeat = 1
					# same measure
					elif self.measure_number == self.prev_measure_number: 
						self.is_downbeat = 0
				# if in same onset group
				elif self.prev_time_position == self.time_position:
					self.is_downbeat = self.prev_is_downbeat
		elif self.is_grace_note == 1:
			self.is_downbeat = 0 # if grace note, no downbeat

	def get_ornament(self):
		'''
		get ornaments from note.note_notations.*:
			- is_trill
			- is_mordent
			- wavy_line
		'''
		if self.note.note_notations.is_trill is True:
			self.ornament['trill'] = 1
		elif self.note.note_notations.is_mordent is True:
			self.ornament['mordent'] = 1
		elif self.note.note_notations.wavy_line is not None:
			self.ornament['wavy'] = 1
		else:
			self.ornament['none'] = 1

	def get_tuplet(self):
		'''
		get tuplets from note.note_notations.tuplet_*:
			- start
			- stop
		'''
		if self.note.note_notations.tuplet_start is True:
			self.tuplet['start'] = 1
		elif self.note.note_notations.tuplet_stop is True:
			self.tuplet['stop'] = 1
		else:
			self.tuplet['none'] = 1

	def get_tied(self):
		'''
		get tied from note.note_notations.tied_start
		* tied_stop notes are applied as longer duration
		'''
		if self.note.note_notations.tied_start is True:
			self.is_tied = 1



