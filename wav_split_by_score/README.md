
Prepare requirements: 

	python3 install -r requirements.txt

Default setting is to split into all measures
    
	python3 main.py /home/user/data /home/user/savedir

To split particular measures: 

	python3 main.py /home/user/data /home/user/savedir start_measure end_measure


The current code includes several steps:
    - perform WAV, score XML, score MIDI, and perform MIDI should be prepared

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


** Other things
- 못갖춘마디도 마디수에 포함 

