import torch
import torch.nn as nn
import math


class VideoInTurn(nn.Module):
    def __init__(self, num_group: int, device):
        super().__init__()
        self._num_group = num_group
        self._num_sa = int(math.log2(num_group))
        self._group_binary_list = self._get_group_binary_list()
        self._steps = 0
        self._device = device

    def _get_group_binary_list(self) -> list:
        group_binary_list = []
        for i in range(self._num_group):
            group_binary_list.append(format(i, "0" + str(self._num_sa) + "b"))
        return group_binary_list

    def _shuffle_tensor(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor,
        group_others: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randperm(video.shape[0])
        return (
            video[indices],
            audio[indices],
            text[indices],
            target[indices],
            group[indices],
            group_others[indices],
        )

    def _video_in_turn(
        self,
        batch_video: torch.Tensor,
        batch_audio: torch.Tensor,
        batch_text: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        iter = self._steps % self._num_sa
        self._steps += 1

        group_video_0 = []
        group_audio_0 = []
        group_text_0 = []
        group_target_0 = []
        group_others_0 = []

        group_video_1 = []
        group_audio_1 = []
        group_text_1 = []
        group_target_1 = []
        group_others_1 = []
        for group_binary in self._group_binary_list:
            if group_binary[iter] == "0":
                group_idx = int(group_binary, 2)
                group_video = batch_video[torch.where(batch_group == group_idx)[0]]
                group_audio = batch_audio[torch.where(batch_group == group_idx)[0]]
                group_text = batch_text[torch.where(batch_group == group_idx)[0]]
                group_target = batch_target[torch.where(batch_group == group_idx)[0]]

                group_binary_others = group_binary[:iter] + group_binary[iter + 1 :]
                group_binary_others_idx = int(group_binary_others, 2)
                for i in range(group_video.shape[0]):
                    group_others_0.append([group_binary_others_idx])

                group_video_0.append(group_video)
                group_audio_0.append(group_audio)
                group_text_0.append(group_text)
                group_target_0.append(group_target)
            elif group_binary[iter] == "1":
                group_idx = int(group_binary, 2)
                group_video = batch_video[torch.where(batch_group == group_idx)[0]]
                group_audio = batch_audio[torch.where(batch_group == group_idx)[0]]
                group_text = batch_text[torch.where(batch_group == group_idx)[0]]
                group_target = batch_target[torch.where(batch_group == group_idx)[0]]

                group_binary_others = group_binary[:iter] + group_binary[iter + 1 :]
                group_binary_others_idx = int(group_binary_others, 2)
                for i in range(group_video.shape[0]):
                    group_others_1.append([group_binary_others_idx])

                group_video_1.append(group_video)
                group_audio_1.append(group_audio)
                group_text_1.append(group_text)
                group_target_1.append(group_target)

        group_video_0 = torch.cat(group_video_0, dim=0)
        group_audio_0 = torch.cat(group_audio_0, dim=0)
        group_text_0 = torch.cat(group_text_0, dim=0)
        group_target_0 = torch.cat(group_target_0, dim=0)
        group_label_0 = torch.zeros(group_video_0.shape[0], 1)
        group_others_0 = torch.tensor(group_others_0)

        group_video_1 = torch.cat(group_video_1, dim=0)
        group_audio_1 = torch.cat(group_audio_1, dim=0)
        group_text_1 = torch.cat(group_text_1, dim=0)
        group_target_1 = torch.cat(group_target_1, dim=0)
        group_label_1 = torch.ones(group_video_1.shape[0], 1)
        group_others_1 = torch.tensor(group_others_1)

        video = torch.cat([group_video_0, group_video_1], dim=0)
        audio = torch.cat([group_audio_0, group_audio_1], dim=0)
        text = torch.cat([group_text_0, group_text_1], dim=0)
        target = torch.cat([group_target_0, group_target_1], dim=0)
        group = torch.cat([group_label_0, group_label_1], dim=0)
        group_others = torch.cat([group_others_0, group_others_1], dim=0)

        return video, audio, text, target, group, group_others

    def forward(
        self,
        batch_video: torch.Tensor,
        batch_audio: torch.Tensor,
        batch_text: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video, audio, text, target, group, group_others = self._video_in_turn(
            batch_video,
            batch_audio,
            batch_text,
            batch_target,
            batch_group,
        )
        video, audio, text, target, group, group_others = self._shuffle_tensor(
            video,
            audio,
            text,
            target,
            group,
            group_others,
        )

        video = video.to(self._device)
        audio = audio.to(self._device)
        text = text.to(self._device)
        target = target.to(self._device)
        group = group.to(self._device)
        group_others = group_others.to(self._device)
        return video, audio, text, target, group, group_others
