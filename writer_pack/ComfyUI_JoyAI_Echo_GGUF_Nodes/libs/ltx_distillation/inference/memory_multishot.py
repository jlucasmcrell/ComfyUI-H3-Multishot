"""Standalone paired audio-video memory helpers for multi-shot inference."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from ltx_core.model.audio_vae import AudioProcessor
from ltx_core.types import Audio
from ltx_distillation.audio_memory import (
    latent_window_size_to_pixel_window_size,
    mel_window_bounds_to_seconds,
    select_audio_window_with_bounds,
    select_video_frame_indices_from_time_range,
)

def prompt_payload_to_text(payload: Any, prompt_max_chars: Optional[int] = None) -> str:
    if isinstance(payload, str):
        text = payload.strip()
    else:
        raise TypeError(
            f"Unsupported prompt payload type: {type(payload).__name__}. "
            "Each shot must be a single concatenated prompt string."
        )

    if prompt_max_chars and len(text) > prompt_max_chars:
        text = text[:prompt_max_chars]
    return text


def json_to_prompts(data: dict[str, Any], prompt_max_chars: Optional[int] = None) -> list[str]:
    if isinstance(data.get("prompts"), list):
        prompts = [prompt_payload_to_text(item, prompt_max_chars) for item in data["prompts"]]
        return [prompt for prompt in prompts if prompt]

    shots = data.get("shots", [])
    if not isinstance(shots, list):
        return []
    prompts = [prompt_payload_to_text(item, prompt_max_chars) for item in shots]
    return [prompt for prompt in prompts if prompt]


def load_multishot_prompts(prompts_file: str | Path, prompt_max_chars: Optional[int] = None) -> list[str]:
    prompts_path = Path(prompts_file)
    with prompts_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    prompts = json_to_prompts(payload, prompt_max_chars=prompt_max_chars)
    if not prompts:
        raise ValueError(f"No prompts found in multishot prompts file: {prompts_path}")
    return prompts


def normalize_audio_waveform_for_media(audio_waveform: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if audio_waveform is None:
        return None

    waveform = getattr(audio_waveform, "waveform", audio_waveform)
    waveform = torch.as_tensor(waveform).detach().cpu().float()

    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for decoded audio, got shape={tuple(waveform.shape)}")
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] not in {1, 2} and waveform.shape[1] in {1, 2}:
        waveform = waveform.transpose(0, 1)
    elif waveform.ndim != 2:
        raise ValueError(f"Expected decoded audio with 1, 2, or 3 dims, got shape={tuple(waveform.shape)}")

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform.contiguous()


def audio_waveform_stats(audio_waveform: Optional[torch.Tensor]) -> dict[str, Any]:
    waveform = normalize_audio_waveform_for_media(audio_waveform)
    if waveform is None:
        return {
            "present": False,
            "shape": None,
            "num_samples": 0,
            "rms": 0.0,
            "peak": 0.0,
            "mean_abs": 0.0,
        }
    waveform_f = waveform.float()
    return {
        "present": True,
        "shape": list(waveform.shape),
        "num_samples": int(waveform.shape[-1]),
        "rms": float(torch.sqrt(torch.mean(waveform_f.square())).item()),
        "peak": float(torch.max(torch.abs(waveform_f)).item()),
        "mean_abs": float(torch.mean(torch.abs(waveform_f)).item()),
    }


def build_paired_audio_memory_kwargs(
    memory_bank: "PairedAudioVideoMemoryBank",
    *,
    enable_audio_memory: bool,
    v2a_grad_scale: float = 1.0,
    memory_position_mode: str = "reference",
    speaker: Optional[str] = None,
) -> dict[str, Any]:
    if not enable_audio_memory:
        return {}

    memory_audio = memory_bank.get_memory_audio(speaker=speaker)
    if memory_audio is None:
        raise RuntimeError("audio memory was requested but the memory bank contains entries without audio latents")

    kwargs: dict[str, Any] = {
        "memory_audio": memory_audio,
        "memory_audio_timestep": torch.zeros(memory_audio.shape[:2], dtype=torch.float32),
        "memory_audio_segment_lengths": memory_bank.get_memory_audio_segment_lengths(),
        "v2a_grad_scale": float(v2a_grad_scale),
        "memory_position_mode": str(memory_position_mode),
        "paired_audio_memory": True,
    }
    return kwargs


@dataclass
class MemoryEntry:
    frame: Image.Image | list[Image.Image]
    audio_latent: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PairedAudioVideoMemoryBank:
    def __init__(self, max_size: int, save_mode: str, num_fix_frames: int = 0) -> None:
        self.max_size = int(max_size)
        self.save_mode = str(save_mode)
        self.num_fix_frames = max(0, int(num_fix_frames))
        self.memory: list[MemoryEntry] = []

    @staticmethod
    def _prepare_audio_latent(audio_latent: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if audio_latent is None:
            return None
        if audio_latent.dim() != 3:
            raise ValueError(f"Expected audio_latent shape [B, T, C], got shape={tuple(audio_latent.shape)}")
        return audio_latent.detach().cpu().contiguous()

    @staticmethod
    def _normalize_waveform_channels(waveform: torch.Tensor, target_channels: int = 2) -> torch.Tensor:
        waveform = torch.as_tensor(waveform).detach().cpu().float()
        if waveform.ndim == 3:
            if waveform.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for waveform, got shape={tuple(waveform.shape)}")
            waveform = waveform[0]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform [C, T], got shape={tuple(waveform.shape)}")
        if waveform.shape[0] == target_channels:
            return waveform.contiguous()
        if waveform.shape[0] == 1:
            return waveform.repeat(target_channels, 1).contiguous()
        if waveform.shape[0] > target_channels:
            return waveform[:target_channels].contiguous()
        pad = target_channels - waveform.shape[0]
        return torch.cat([waveform, waveform[-1:].repeat(pad, 1)], dim=0).contiguous()

    @staticmethod
    def _select_audio_window(audio_latent: torch.Tensor, window_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
        total_frames = int(audio_latent.shape[1])
        window_size = max(1, int(window_size))
        window_len = min(total_frames, window_size)
        window_start = max((total_frames - window_len) // 2, 0)
        window_end = window_start + window_len
        metadata = {
            "audio_window_start": int(window_start),
            "audio_window_end": int(window_end),
            "audio_window_length": int(window_len),
            "audio_total_frames": int(total_frames),
        }
        return audio_latent[:, window_start:window_end].contiguous(), metadata

    @staticmethod
    def _select_audio_window_from_waveform(
        audio_latent: torch.Tensor,
        *,
        audio_waveform: torch.Tensor,
        audio_sample_rate: int,
        window_size: int,
        selection_mode: str,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
        downsample_factor: int,
        is_causal: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        waveform = PairedAudioVideoMemoryBank._normalize_waveform_channels(audio_waveform, target_channels=2)
        processor = AudioProcessor(
            target_sample_rate=int(audio_sample_rate),
            mel_bins=int(mel_bins),
            mel_hop_length=int(mel_hop_length),
            n_fft=int(n_fft),
        )
        mel_spectrogram = processor.waveform_to_mel(
            Audio(waveform=waveform.unsqueeze(0), sampling_rate=int(audio_sample_rate))
        )
        pixel_window_size = latent_window_size_to_pixel_window_size(
            int(window_size),
            downsample_factor=int(downsample_factor),
            is_causal=bool(is_causal),
        )
        _, window_start_indices, window_end_indices = select_audio_window_with_bounds(
            mel_spectrogram,
            pixel_window_size,
            mode=str(selection_mode).lower(),
        )
        mel_start = int(window_start_indices[0].item())
        mel_end = int(window_end_indices[0].item())
        start_time_sec, end_time_sec = mel_window_bounds_to_seconds(
            mel_start,
            mel_end,
            hop_length=int(mel_hop_length),
            sample_rate=int(audio_sample_rate),
        )

        total_frames = int(audio_latent.shape[1])
        window_len = min(total_frames, max(1, int(window_size)))
        duration_sec = max(float(waveform.shape[-1]) / float(audio_sample_rate), 1e-6)
        center_time_sec = max(0.0, min(0.5 * (start_time_sec + end_time_sec), duration_sec))
        center_latent = int(round(center_time_sec / duration_sec * float(max(total_frames - 1, 0))))
        window_start = max(0, min(center_latent - window_len // 2, max(total_frames - window_len, 0)))
        window_end = window_start + window_len
        metadata = {
            "audio_window_selection_mode": str(selection_mode).lower(),
            "audio_window_start": int(window_start),
            "audio_window_end": int(window_end),
            "audio_window_length": int(window_len),
            "audio_total_frames": int(total_frames),
            "mel_window_start": int(mel_start),
            "mel_window_end": int(mel_end),
            "audio_window_start_time_sec": float(start_time_sec),
            "audio_window_end_time_sec": float(end_time_sec),
        }
        return audio_latent[:, window_start:window_end].contiguous(), metadata

    @staticmethod
    def _select_video_clip_around_frame(
        frames: list[Image.Image],
        *,
        center_frame: int,
        video_clip_num_frames: int,
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        video_clip_num_frames = max(1, int(video_clip_num_frames))
        center_frame = max(0, min(int(center_frame), len(frames) - 1))
        left_context = (video_clip_num_frames - 1) // 2
        clip_start = max(0, min(center_frame - left_context, max(len(frames) - video_clip_num_frames, 0)))
        clip_end = min(clip_start + video_clip_num_frames, len(frames))
        clip = list(frames[clip_start:clip_end])
        if clip and len(clip) < video_clip_num_frames:
            clip.extend([clip[-1]] * (video_clip_num_frames - len(clip)))
        metadata = {
            "video_clip_start": int(clip_start),
            "video_clip_end": int(clip_end),
            "video_clip_length": int(len(clip)),
            "video_clip_center_frame": int(center_frame),
            "video_total_frames": int(len(frames)),
        }
        return clip, metadata

    @staticmethod
    def _select_video_clip_for_audio_window(
        frames: list[Image.Image],
        *,
        audio_window_start: int,
        audio_window_end: int,
        audio_total_frames: int,
        video_clip_num_frames: int,
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        audio_total_frames = max(1, int(audio_total_frames))
        window_center = (float(audio_window_start) + float(audio_window_end - 1)) * 0.5
        center_ratio = window_center / float(max(audio_total_frames - 1, 1))
        center_frame = int(round(center_ratio * float(max(len(frames) - 1, 0))))
        return PairedAudioVideoMemoryBank._select_video_clip_around_frame(
            frames,
            center_frame=center_frame,
            video_clip_num_frames=video_clip_num_frames,
        )

    def _trim(self) -> None:
        if self.max_size <= 0 or len(self.memory) <= self.max_size:
            return
        # NOTE: keep_tail==0 must yield NO tail entries. `tail[-0:]` is `tail[0:]`
        # i.e. the WHOLE list, so the original `tail[-keep_tail:]` silently turned
        # the trim into a no-op whenever max_size <= num_fix_frames (e.g. the
        # common max_size=3 / num_fix_frames=3 combo): the bank then grew
        # unbounded (memory_size climbing 0,1,2,...,N-1 in the console) and every
        # shot conditioned on EVERY prior shot, compounding generational drift.
        # Also clamp the anchor block so it can never exceed max_size.
        n_fixed = min(self.num_fix_frames, self.max_size)
        fixed = self.memory[:n_fixed]
        tail = self.memory[self.num_fix_frames :]
        keep_tail = self.max_size - len(fixed)
        if keep_tail <= 0:
            self.memory = fixed
            return

        # Per-character reservation. The plain recency window lets a talkative
        # character's slots evict a quieter one's ONLY voice exemplar, after
        # which that character has nothing to be continued from and re-renders
        # with a new voice. Guarantee every character seen in the tail keeps
        # their most recent slot, and only evict from characters that still
        # have another slot left.
        #
        # Indices are used throughout rather than entry objects: MemoryEntry is
        # an eq=True dataclass holding tensors, so `entry in list` would run a
        # tensor comparison and raise.
        kept = list(range(len(tail)))[-keep_tail:]
        last_by_char: dict[str, int] = {}
        for i, entry in enumerate(tail):
            owner = entry.metadata.get("character")
            if owner:
                last_by_char[owner] = i          # later shot wins = most recent

        for reserved in sorted(last_by_char.values()):
            if reserved in kept:
                continue
            counts: dict[Any, int] = {}
            for i in kept:
                owner = tail[i].metadata.get("character")
                counts[owner] = counts.get(owner, 0) + 1
            victim = next(
                (i for i in kept
                 if tail[i].metadata.get("character") is not None
                 and counts[tail[i].metadata.get("character")] > 1),
                None,
            )
            if victim is None:
                break                            # nothing safe to drop; leave as-is
            kept.remove(victim)
            kept.append(reserved)
            kept.sort()

        self.memory = fixed + [tail[i] for i in kept]

    def save_memory_slot(
        self,
        frames: list[Image.Image],
        audio_latent: torch.Tensor,
        *,
        audio_window_size: int,
        video_clip_num_frames: int,
        audio_waveform: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 16000,
        video_fps: float = 24.0,
        audio_window_selection_mode: str = "center",
        video_frame_selection_mode: str = "center",
        audio_memory_mel_bins: int = 128,
        audio_memory_mel_hop_length: int = 160,
        audio_memory_n_fft: int = 1024,
        audio_memory_downsample_factor: int = 4,
        audio_memory_is_causal: bool = True,
        character: Optional[str] = None,
    ) -> dict[str, Any]:
        audio_latent = self._prepare_audio_latent(audio_latent)
        if audio_latent is None:
            raise ValueError("paired audio memory slot requires audio_latent")

        selection_mode = str(audio_window_selection_mode).lower()
        if audio_waveform is not None and selection_mode != "center":
            try:
                window_latent, audio_metadata = self._select_audio_window_from_waveform(
                    audio_latent,
                    audio_waveform=audio_waveform,
                    audio_sample_rate=audio_sample_rate,
                    window_size=audio_window_size,
                    selection_mode=selection_mode,
                    mel_bins=audio_memory_mel_bins,
                    mel_hop_length=audio_memory_mel_hop_length,
                    n_fft=audio_memory_n_fft,
                    downsample_factor=audio_memory_downsample_factor,
                    is_causal=audio_memory_is_causal,
                )
                selected_frame = select_video_frame_indices_from_time_range(
                    num_frames=len(frames),
                    fps=float(video_fps),
                    start_time_sec=float(audio_metadata["audio_window_start_time_sec"]),
                    end_time_sec=float(audio_metadata["audio_window_end_time_sec"]),
                    count=1,
                    mode=str(video_frame_selection_mode).lower(),
                )[0]
                video_clip, video_metadata = self._select_video_clip_around_frame(
                    frames,
                    center_frame=int(selected_frame),
                    video_clip_num_frames=video_clip_num_frames,
                )
            except Exception as exc:
                window_latent, audio_metadata = self._select_audio_window(audio_latent, audio_window_size)
                audio_metadata["audio_window_selection_mode"] = "center"
                audio_metadata["selection_fallback"] = f"{selection_mode}: {exc}"
                video_clip, video_metadata = self._select_video_clip_for_audio_window(
                    frames,
                    audio_window_start=int(audio_metadata["audio_window_start"]),
                    audio_window_end=int(audio_metadata["audio_window_end"]),
                    audio_total_frames=int(audio_metadata["audio_total_frames"]),
                    video_clip_num_frames=video_clip_num_frames,
                )
        else:
            window_latent, audio_metadata = self._select_audio_window(audio_latent, audio_window_size)
            audio_metadata["audio_window_selection_mode"] = "center"
            video_clip, video_metadata = self._select_video_clip_for_audio_window(
                frames,
                audio_window_start=int(audio_metadata["audio_window_start"]),
                audio_window_end=int(audio_metadata["audio_window_end"]),
                audio_total_frames=int(audio_metadata["audio_total_frames"]),
                video_clip_num_frames=video_clip_num_frames,
            )

        # "last": take the video half of the slot from the END of the shot
        # instead of from wherever the loudest speech happened.
        #
        # The audio window is still chosen by max_response - that is the right
        # exemplar for the VOICE. But pairing it with a mid-shot video clip
        # makes the next shot resume from mid-shot, which reads on screen as the
        # take rewinding a second or two before continuing. Decoupling the two
        # keeps the best voice reference AND gives the next shot a genuine
        # continuation point.
        if str(video_frame_selection_mode).lower() == "last" and frames:
            n = max(1, int(video_clip_num_frames))
            clip_start = max(0, len(frames) - n)
            video_clip = list(frames[clip_start:])
            if len(video_clip) < n:
                video_clip.extend([video_clip[-1]] * (n - len(video_clip)))
            video_metadata = {
                "video_clip_start": int(clip_start),
                "video_clip_end": int(len(frames)),
                "video_clip_length": int(len(video_clip)),
                "video_clip_center_frame": int(len(frames) - 1),
                "video_total_frames": int(len(frames)),
                "video_frame_selection_mode": "last",
            }

        metadata = {"selection_mode": "paired_audio_window", **audio_metadata, **video_metadata}
        if character:
            metadata["character"] = str(character)
        entry = MemoryEntry(frame=video_clip, audio_latent=window_latent, metadata=metadata)
        fixed = self.memory[: self.num_fix_frames]
        free = self.memory[self.num_fix_frames :]
        free.append(entry)
        self.memory = fixed + free
        self._trim()
        return metadata

    def get_memory_frames(self) -> list[Image.Image | list[Image.Image]]:
        return [entry.frame for entry in self.memory]

    def get_memory_metadata(self) -> list[dict[str, Any]]:
        return [dict(entry.metadata) for entry in self.memory]

    def get_memory_audio(self, speaker: Optional[str] = None) -> Optional[torch.Tensor]:
        """Concatenated memory audio, optionally scoped to ONE speaker.

        Slots are written one per shot and the bank is character-blind, so with
        two characters alternating shots the audio lane carries BOTH voices into
        every shot. The model then has two equally valid voices to continue and
        picks either - the same-gender voice merge.

        When `speaker` is given, slots belonging to a DIFFERENT named character
        are zeroed instead of dropped. Zeroing (not dropping) is deliberate: the
        slot count, the per-slot lengths and therefore
        `_build_paired_memory_cross_mask`'s positional pairing all stay exactly
        as they were, so this cannot re-introduce the even-split desync that
        shredded stored voices. The video half is untouched, so the non-speaker
        keeps full face continuity - their slot simply pairs their face with
        silence, which is what was true in that shot anyway.

        Untagged slots (character is None) are always kept: a script that does
        not declare speakers behaves exactly as before.
        """
        # COLD START. If this speaker owns no slot yet, filtering would zero
        # EVERY slot and hand them a bank of pure silence to continue from -
        # strictly worse than no filtering at all. Measured on the first attempt
        # at this feature: a character's first line generated against an all-silent
        # bank and her pitch wandered 219 -> 276 -> 258Hz across three lines,
        # where with filtering off she was identical on all three. That single
        # regression is what got per-character memory shelved.
        #
        # So: a character's FIRST line falls back to the unfiltered bank, which
        # is exactly the pre-per-character behaviour for that one shot. From
        # their second line on they have an exemplar of their own and get full
        # isolation. The failure mode this removes (silence) is worse than the
        # one it briefly tolerates (one shot of shared bank).
        owns_a_slot = speaker is not None and any(
            entry.metadata.get("character") == speaker for entry in self.memory)
        effective_speaker = speaker if owns_a_slot else None

        # ANCHOR + LATEST (2026-07-28). A speaker's audio context is their
        # anchor slot(s) plus their MOST RECENT generated slot - older generated
        # slots are zeroed like other characters'. Measured motivation: shot 5's
        # line flipped the speaker's accent against a context of three American
        # exemplars (text pull beat the whole bank), and then shots 6-7 stayed
        # flipped because the drifted slot outnumbered nothing - consistency
        # machinery faithfully propagated the poison. With this policy a drifted
        # shot contributes exactly one slot against the permanent anchor instead
        # of accumulating a majority, so a single bad roll cannot own the rest
        # of the video. Untagged slots and no-speaker calls are untouched.
        latest_gen_idx = -1
        if effective_speaker is not None:
            for _i, entry in enumerate(self.memory):
                if (entry.metadata.get("character") == effective_speaker
                        and not entry.metadata.get("voice_anchor")):
                    latest_gen_idx = _i

        audio_latents = []
        for _i, entry in enumerate(self.memory):
            audio_latent = entry.audio_latent
            if audio_latent is not None and effective_speaker is not None:
                owner = entry.metadata.get("character")
                if owner is not None and owner != effective_speaker:
                    audio_latent = torch.zeros_like(audio_latent)
                elif (owner == effective_speaker
                        and not entry.metadata.get("voice_anchor")
                        and _i != latest_gen_idx):
                    audio_latent = torch.zeros_like(audio_latent)
            audio_latents.append(audio_latent)
        if not audio_latents or any(audio_latent is None for audio_latent in audio_latents):
            return None
        first = audio_latents[0]
        assert first is not None
        batch_size = first.shape[0]
        channels = first.shape[2]
        for audio_latent in audio_latents:
            assert audio_latent is not None
            if audio_latent.shape[0] != batch_size or audio_latent.shape[2] != channels:
                raise ValueError(
                    "All memory audio latents must share batch and channel dimensions, "
                    f"got first={tuple(first.shape)} current={tuple(audio_latent.shape)}"
                )
        # Harmonize dtypes before the cat: a pre-seeded voice-anchor slot may
        # carry a different dtype than the pipeline's generated latents, and
        # torch.cat requires one dtype. The LAST slot is the most recently
        # generated one, so its dtype is the pipeline's - cast everything to it.
        target_dtype = audio_latents[-1].dtype
        return torch.cat([al.to(target_dtype) for al in audio_latents],
                         dim=1).contiguous()

    def get_memory_audio_segment_lengths(self) -> tuple[tuple[int, ...], ...]:
        audio_latents = [entry.audio_latent for entry in self.memory]
        if not audio_latents or any(audio_latent is None for audio_latent in audio_latents):
            return ()
        return (tuple(int(audio_latent.shape[1]) for audio_latent in audio_latents if audio_latent is not None),)

    def __len__(self) -> int:
        return len(self.memory)


def video_uint8_to_pil_frames(video_uint8: torch.Tensor) -> list[Image.Image]:
    if video_uint8.ndim != 4:
        raise ValueError(f"Expected [F, H, W, C] uint8 video, got shape={tuple(video_uint8.shape)}")
    if video_uint8.shape[-1] != 3:
        raise ValueError(f"Expected RGB video with trailing channel dim 3, got shape={tuple(video_uint8.shape)}")
    video_uint8 = video_uint8.detach().cpu().contiguous()
    return [Image.fromarray(frame.numpy()) for frame in video_uint8]
