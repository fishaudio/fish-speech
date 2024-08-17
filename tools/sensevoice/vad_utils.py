import torch
from torch.nn.utils.rnn import pad_sequence


def slice_padding_fbank(speech, speech_lengths, vad_segments):
    speech_list = []
    speech_lengths_list = []
    for i, segment in enumerate(vad_segments):

        bed_idx = int(segment[0][0] * 16)
        end_idx = min(int(segment[0][1] * 16), speech_lengths[0])
        speech_i = speech[0, bed_idx:end_idx]
        speech_lengths_i = end_idx - bed_idx
        speech_list.append(speech_i)
        speech_lengths_list.append(speech_lengths_i)
    feats_pad = pad_sequence(speech_list, batch_first=True, padding_value=0.0)
    speech_lengths_pad = torch.Tensor(speech_lengths_list).int()
    return feats_pad, speech_lengths_pad


def slice_padding_audio_samples(speech, speech_lengths, vad_segments):
    speech_list = []
    speech_lengths_list = []
    intervals = []
    for i, segment in enumerate(vad_segments):
        bed_idx = int(segment[0][0] * 16)
        end_idx = min(int(segment[0][1] * 16), speech_lengths)
        speech_i = speech[bed_idx:end_idx]
        speech_lengths_i = end_idx - bed_idx
        speech_list.append(speech_i)
        speech_lengths_list.append(speech_lengths_i)
        intervals.append([bed_idx // 16, end_idx // 16])

    return speech_list, speech_lengths_list, intervals


def merge_vad(vad_result, max_length=15000, min_length=0):
    new_result = []
    if len(vad_result) <= 1:
        return vad_result
    time_step = [t[0] for t in vad_result] + [t[1] for t in vad_result]
    time_step = sorted(list(set(time_step)))
    if len(time_step) == 0:
        return []
    bg = 0
    for i in range(len(time_step) - 1):
        time = time_step[i]
        if time_step[i + 1] - bg < max_length:
            continue
        if time - bg > min_length:
            new_result.append([bg, time])
        # if time - bg < max_length * 1.5:
        #     new_result.append([bg, time])
        # else:
        #     split_num = int(time - bg) // max_length + 1
        #     spl_l = int(time - bg) // split_num
        #     for j in range(split_num):
        #         new_result.append([bg + j * spl_l, bg + (j + 1) * spl_l])
        bg = time
    new_result.append([bg, time_step[-1]])
    return new_result
