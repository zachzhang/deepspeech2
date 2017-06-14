import os
import pandas as pd

label_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
label_map = dict([(label_map[i], i) for i in range(len(label_map))])


def parse_transcript(transcript_path, label_map):
    with open(transcript_path, 'r') as transcript_file:
        transcript = transcript_file.readlines()

    # y = [t.split(' ',1)[1] for t in transcript]
    y = []
    for trans in transcript:
        y.append(list(filter(None, [label_map.get(x) for x in trans])))

    ids = [t.split(' ')[0] for t in transcript]

    return y, ids


base_dir = '../LibriSpeech/dev-clean/'

audio_df = []
label_df = []

for sub_dir in os.listdir(base_dir):
    for sub_dir2 in os.listdir(base_dir + sub_dir):
        files = os.listdir(base_dir + sub_dir + '/' + sub_dir2)
        audio_files = [file for file in files if file[-4:] == 'flac']
        trans_file = sub_dir + '-' + sub_dir2 + '.trans.txt'

        audio_path = [base_dir + sub_dir + '/' + sub_dir2 + '/' + file for file in audio_files]
        audio_ids = [a.split('.')[0] for a in files]

        label, label_ids = parse_transcript(base_dir + sub_dir + '/' + sub_dir2 + '/' + trans_file, label_map)

        audio_df += list(zip(audio_ids, audio_path))
        label_df += list(zip(label_ids, label))

audio_df = pd.DataFrame(audio_df)
label_df = pd.DataFrame(label_df)

df = pd.merge(audio_df,label_df,left_on=0,right_on=0)
df = df.drop('0_x',axis=1)

df.to_csv('training_manifest.csv')