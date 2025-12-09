import os
import math
import numpy as np
import librosa
import gradio as gr

from typing import Callable, Optional
from pathlib import Path

import torch
from functools import partial
from audiotools import AudioSignal
from audiotools.core import util
from tria.model.tria import TRIA
from tria.pipelines.tokenizer import Tokenizer, TokenSequence
from tria.features import rhythm_features
from tria.constants import PRETRAINED_DIR, ASSETS_DIR


################################################################################
# Run inference with trained TRIA models
################################################################################


# Spectrogram plots
SPEC_MELS = 128
SMALL_WIDTH = 900  # Timbre/rhythm prompts
LARGE_WIDTH = 1200  # Generated outputs

# Device
DEVICE = None

# Batched outputs
N_OUTPUTS = 1

# Loaded configuration
LOADED = dict(
    name=None, 
    model=None, 
    tokenizer=None, 
    feature_fn=None, 
    infer_cfg=None, 
    sample_rate=None, 
    max_duration=None,
)

# Pretrained models
MODEL_ZOO = {
    "small_musdb_moises_2b": {
        "checkpoint": "pretrained/tria/small_musdb_moises_2b/80000/model.pt",
        "model_cfg": {
            "codebook_size": 1024,
            "n_codebooks": 9,
            "n_channels": 512,
            "n_feats": 2,
            "n_heads": 8,
            "n_layers": 12,
            "mult": 4,
            "p_dropout": 0.0,
            "bias": True,
            "max_len": 2000,
            "pos_enc": "rope",
            "qk_norm": True,
            "use_sdpa": True,
            "interp": "nearest",
            "share_emb": True,
        },
        "tokenizer_cfg": {"name": "dac"},
        "feature_cfg": {
            "sample_rate": 16_000,
            "n_bands": 2,
            "n_mels": 40,
            "window_length": 384,
            "hop_length": 192,
            "quantization_levels": 5,
            "slow_ma_ms": 200,
            "post_smooth_ms": 100,
            "legacy_normalize": False,
            "clamp_max": 50.0,
            "normalize_quantile": 0.98,
        },
        "infer_cfg": {
            "top_p": 0.95,
            "top_k": None,
            "temp": 1.0,
            "mask_temp": 10.5,
            "iterations": [8, 8, 8, 8, 4, 4, 4, 4, 4],
            "guidance_scale": 2.0,
            "causal_bias": 1.0,
        },
        "max_duration": 12.0,
    },
}


# Example audio
# Use absolute path relative to this file
BASE_DIR = Path(__file__).parent
TIMBRE_EXAMPLE_DIR = BASE_DIR / "example_timbre"
RHYTHM_EXAMPLE = ASSETS_DIR / "beatbox" / "beatbox_1.wav"

########################################
# Audio utilities
########################################


def db_to_linear(db: float):
    return 10.0 ** (db / 20.0)


def bandpass(signal: AudioSignal):
    signal = signal.clone().high_pass(250)
    signal = signal.low_pass(8000)
    signal.ensure_max_of_audio()
    
    return signal


def to_gradio_audio(signal: AudioSignal):
    return(
        signal.sample_rate,
        signal.clone().to_mono().audio_data.flatten().numpy().astype(np.float32)
    )


def from_gradio_audio(sample_rate: int, x: np.ndarray):
    x = np.asarray(x)
    if x.ndim == 1:
        x = torch.from_numpy(x[None, None, :].astype(np.float32))
    elif x.ndim == 2:
        x = torch.from_numpy(x.T[None, :, :].astype(np.float32))

    return AudioSignal(x, sample_rate=int(sample_rate))


########################################
# Spectrogram plotting
########################################


def _strip_silent_borders_uint8(img2d: np.ndarray, thr: int = 2):
    """
    Trim spectrogram columns.
    """
    assert img2d.ndim == 2 and img2d.dtype == np.uint8
    colmax = img2d.max(axis=0)
    left = 0
    while left < colmax.size and colmax[left] <= thr:
        left += 1
    right = colmax.size
    while right > left and colmax[right - 1] <= thr:
        right -= 1
    if right - left < min(8, img2d.shape[1] // 10):
        return img2d
    return img2d[:, left:right]


def _resize_width_linear_uint8(img2d: np.ndarray, target_w: int):
    """
    Resize spectrogram along time dimension.
    """
    H, W = img2d.shape
    if W == target_w:
        return img2d
    x_old = np.linspace(0.0, 1.0, W, endpoint=True)
    x_new = np.linspace(0.0, 1.0, target_w, endpoint=True)
    idx = np.searchsorted(x_old, x_new, side="left")
    idx = np.clip(idx, 1, W - 1)
    x0 = x_old[idx - 1][None, :]
    x1 = x_old[idx][None, :]
    y0 = img2d[:, idx - 1].astype(np.float32)
    y1 = img2d[:, idx].astype(np.float32)
    w = (x_new[None, :] - x0) / (x1 - x0 + 1e-12)
    out = y0 * (1.0 - w) + y1 * w
    return out.clip(0, 255).astype(np.uint8)


def _to_rgb_uint8(img2d: np.ndarray):
    return np.repeat(img2d[:, :, None], 3, axis=2)


def spectrogram_fast(signal: AudioSignal, small: bool = False):
    """
    Fast log-mel spectrogram.
    """
    target_w = SMALL_WIDTH if small else LARGE_WIDTH
    
    signal = signal.clone().to_mono()
    y = signal.audio_data.flatten().numpy()

    if y.size == 0:
        return np.zeros((SPEC_MELS, target_w, 3), dtype=np.uint8)

    n_fft = 512 if small else 1024
    hop   = 128 if small else 256
    n_mels = SPEC_MELS

    S = librosa.feature.melspectrogram(
        y=y, sr=signal.sample_rate, n_fft=n_fft, 
        hop_length=hop, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = np.clip(S_db, -80.0, 0.0)
    img = (255.0 * (S_db + 80.0) / 80.0).astype(np.uint8)  # (n_mels, n_frames)
    img = np.flipud(img)

    img = _strip_silent_borders_uint8(img, thr=2)
    img = _resize_width_linear_uint8(img, target_w=target_w)
    return _to_rgb_uint8(img)


########################################
# Inference utilities
########################################


def _pick_device():
    global DEVICE
    if DEVICE is None:
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cpu")
    return DEVICE


def load_model_by_name(name: str):
    global LOADED
    device = _pick_device()
    cfg = MODEL_ZOO[name]

    model = TRIA(**cfg["model_cfg"])
    # PyTorch 2.6부터 torch.load 기본값이 weights_only=True로 바뀌어
    # 예전 방식으로 저장된 체크포인트를 읽을 때 UnpicklingError가 발생하므로,
    # 신뢰할 수 있는 파일에 한해 weights_only=False를 명시적으로 사용한다.
    sd = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    tokenizer = Tokenizer(**cfg["tokenizer_cfg"])
    tokenizer = tokenizer.to(device)

    feat_fn = partial(rhythm_features, **cfg.get("feature_cfg", {}))

    LOADED.update(
        dict(name=name, model=model, tokenizer=tokenizer, 
             feature_fn=feat_fn, infer_cfg=cfg["infer_cfg"],
             sample_rate=tokenizer.sample_rate, 
             max_duration=cfg["max_duration"]))
    return f"Loaded model '{name}' on {device}"


def _to_channels(signal: AudioSignal, n: int):
    if n == 1:
        return signal.to_mono()
    else:
        if signal.num_channels == n:
            return signal
        else:
            assert signal.num_channels == 1
            signal.audio_data = signal.audio_data(1, n, 1)
        return signal
        

def _prepare_inputs(
    timbre_prompt: AudioSignal, 
    rhythm_prompt: AudioSignal, 
    prefix_dur: float, 
    buffer_dur: float,
    sample_rate: int,
    feat_fn: Callable,
    tokenizer: Tokenizer,
    interp: str,
    device: str,
):
    assert timbre_prompt.batch_size == rhythm_prompt.batch_size
    n_channels = tokenizer.n_channels

    # Prepare audio
    timbre_prompt = timbre_prompt.clone().resample(sample_rate)
    rhythm_prompt = rhythm_prompt.clone().resample(sample_rate)

    timbre_prompt = _to_channels(timbre_prompt, n_channels)
    rhythm_prompt = _to_channels(rhythm_prompt, n_channels)
    
    timbre_prompt = timbre_prompt.truncate_samples(int(prefix_dur * sample_rate))
    rhythm_prompt = rhythm_prompt.truncate_samples(
        int(buffer_dur * sample_rate) - timbre_prompt.signal_length
    )

    # Tokenize
    timbre_tokens = tokenizer.encode(timbre_prompt)
    rhythm_tokens = tokenizer.encode(rhythm_prompt)

    tokens = torch.cat(
        [timbre_tokens.tokens, rhythm_tokens.tokens], dim=-1
    )
    n_batch, n_codebooks, n_frames = tokens.shape
    prefix_frames = timbre_tokens.tokens.shape[-1]
    
    # Extract features
    _feats = feat_fn(rhythm_prompt)
    _feats = torch.nn.functional.interpolate(
        _feats, n_frames-prefix_frames, mode=interp
    )
    feats = torch.zeros(
        n_batch, 
        _feats.shape[1], 
        n_frames, 
        device=device
    )
    feats[..., prefix_frames:] = _feats

    # Construct masks
    prefix_mask = torch.arange(
        n_frames, device=device
    )[None, :].repeat(n_batch, 1) < prefix_frames                    # (n_batch, n_frames)
    tokens_mask = prefix_mask[:, None, :].repeat(1, n_codebooks, 1)  # (n_batch, n_codebooks, n_frames)
    feats_mask = ~prefix_mask                                        # (n_batch, n_frames)

    return tokens, feats, tokens_mask, feats_mask, timbre_tokens, rhythm_tokens, prefix_frames


@torch.no_grad()
def _inference(
    timbre_prompt: AudioSignal,
    rhythm_prompt: AudioSignal,
    sampling_ctrls, inference_ctrls, 
    audio_ctrls, schedule_ctrls,
    seed,
    override_buffer_dur: Optional[float] = None,
    override_prefix_dur: Optional[float] = None,
):
   
    if LOADED["model"] is None or LOADED["tokenizer"] is None or LOADED["feature_fn"] is None:
        raise RuntimeError(f"No model loaded")

    device = _pick_device()

    feat_fn = LOADED["feature_fn"]
    tokenizer = LOADED["tokenizer"]
    # Use overridden buffer duration if provided, else use model default
    buffer_dur = override_buffer_dur if override_buffer_dur is not None else LOADED["max_duration"]
    sample_rate = LOADED["sample_rate"]
    tokenizer = LOADED["tokenizer"]
    model = LOADED["model"]
    interp = model.interp

    # Prefix mode
    if override_prefix_dur is not None:
        prefix_dur = override_prefix_dur
    else:
        # Default to max length of 1/3 buffer
        prefix_dur = int(buffer_dur / 3)
    
    # Optionally filter rhythm audio
    if audio_ctrls["filter_inputs"]:
        rhythm_prompt = bandpass(rhythm_prompt)
    
    timbre_prompt = timbre_prompt.clone().to(device)
    rhythm_prompt = rhythm_prompt.clone().to(device)
    
    timbre_prompt.ensure_max_of_audio()
    rhythm_prompt.ensure_max_of_audio()

    (
        buffer, feats, buffer_mask, feats_mask, 
        timbre_tokens, rhythm_tokens, prefix_frames,
    ) = _prepare_inputs(
        timbre_prompt, rhythm_prompt, prefix_dur, buffer_dur,
        sample_rate, feat_fn, tokenizer, interp, device,
    )

    # Random seed
    util.seed(seed[0])

    # Cleanup
    if sampling_ctrls["top_p"] in [0., 1.0]:
        sampling_ctrls["top_p"] = None    
    if sampling_ctrls["top_k"] in [0]:
        sampling_ctrls["top_k"] = None
    if inference_ctrls["cfg_scale"] in [0.]:
        inference_ctrls["cfg_scale"] = None
    
    # Inference
    generated = model.inference(
        buffer, feats, buffer_mask.clone(), feats_mask,
        top_p=sampling_ctrls["top_p"],
        top_k=sampling_ctrls["top_k"],
        temp=sampling_ctrls["temperature"],
        mask_temp=sampling_ctrls["mask_temperature"],
        iterations=[int(v) for v in schedule_ctrls],
        guidance_scale=inference_ctrls["cfg_scale"],
        causal_bias=inference_ctrls["causal_bias"],
        seed=seed,
    )[..., prefix_frames:]

    # Write result
    assert generated.shape == rhythm_tokens.tokens.shape
    rhythm_tokens.tokens = generated

    # Decode to audio
    out = tokenizer.decode(rhythm_tokens)
    out.normalize(audio_ctrls["loudness_db"])
    out.ensure_max_of_audio()

    # Echo parameters
    msg = (
        f"Model='{LOADED['name']}' on {device} | sample_rate={sample_rate}, "
        f"prefix_dur={min(prefix_dur, timbre_prompt.duration):0.3f} | "
        f"top_p={sampling_ctrls['top_p']}, "
        f"top_k={sampling_ctrls['top_k']}, "
        f"temp={sampling_ctrls['temperature']}, "
        f"mask_temp={sampling_ctrls['mask_temperature']}; "
        f"seed={int(seed[0])}, "
        f"causal_bias={inference_ctrls['causal_bias']}, "
        f"cfg={inference_ctrls['cfg_scale']}"
    )
    return out, msg


########################################
# Gradio event handlers
########################################


def handle_audio_change(audio_tuple):
    if audio_tuple is None or audio_tuple[1] is None:
        return np.zeros((SPEC_MELS, SMALL_WIDTH, 3), dtype=np.uint8)
    signal = from_gradio_audio(*audio_tuple)
    return spectrogram_fast(signal, small=True)


def scan_example_timbres():
    """
    Scan example timbres from directory.
    """
    if not TIMBRE_EXAMPLE_DIR.exists():
        return []
    
    files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        files.extend(list(TIMBRE_EXAMPLE_DIR.glob(ext)))
    
    # Sort by name
    files = sorted(files, key=lambda x: x.name)
    return [str(f) for f in files]


def on_refresh_examples():
    """
    Refresh example timbres list.
    """
    files = scan_example_timbres()
    # Return updated choices for the dropdown
    return gr.Dropdown(choices=files, value=None)


def on_load_example_from_dropdown(filepath):
    """
    Load audio from selected dropdown file.
    """
    if not filepath:
        return None, None
    
    signal = AudioSignal(filepath)
    return to_gradio_audio(signal), spectrogram_fast(signal, small=True)


def load_example_rhythm():
    """
    Return a gr.Audio value and a spectrogram image.
    """
    signal = AudioSignal(RHYTHM_EXAMPLE)
    return to_gradio_audio(signal), spectrogram_fast(signal, small=True)


def on_select_model(name: str):
    msg = load_model_by_name(name)
    return msg


def on_generate(
    model_name,
    timbre_prompt, rhythm_prompt,
    top_p, top_k, temperature, mask_temperature,
    seed, causal_bias, cfg_scale,
    loudness_db, filter_inputs,
    *schedule_vals,
    progress=gr.Progress()
):
    # Ensure model up-to-date
    if LOADED["name"] != model_name:
        _ = load_model_by_name(model_name)
    sample_rate = LOADED["sample_rate"]

    sampling_ctrls = dict(
        top_p=float(top_p), top_k=int(top_k), 
        temperature=float(temperature),
        mask_temperature=float(mask_temperature))
    inference_ctrls = dict(
        causal_bias=float(causal_bias), 
        cfg_scale=float(cfg_scale))
    audio_ctrls = dict(
        loudness_db=float(loudness_db), 
        filter_inputs=bool(filter_inputs))
    schedule_ctrls = [int(v) for v in schedule_vals]

    seed = [seed + i for i in range(N_OUTPUTS)]

    # Convert to AudioSignal
    assert isinstance(timbre_prompt, tuple)
    assert isinstance(rhythm_prompt, tuple)
    timbre_prompt = from_gradio_audio(*timbre_prompt)
    rhythm_prompt = from_gradio_audio(*rhythm_prompt)
    
    # === Overlap-Add Logic ===
    
    # Settings for overlap-add
    SEGMENT_DUR = 8.0  # Total buffer size for generation
    OVERLAP_DUR = 1.0  # Overlap between segments
    PREFIX_DUR = 2.0   # Timbre prompt duration within the segment
    
    # Effective duration generated per step (excluding prefix)
    # The model takes prefix + rhythm_chunk.
    # Total buffer = PREFIX_DUR + RHYTHM_CHUNK_DUR
    # We want SEGMENT_DUR = 8.0
    # So Rhythm Chunk size = 6.0
    RHYTHM_CHUNK_DUR = SEGMENT_DUR - PREFIX_DUR
    
    # Step size for moving through the input rhythm
    # We want overlap of 1.0 sec in the output.
    # So we advance by RHYTHM_CHUNK_DUR - OVERLAP_DUR
    STEP_SIZE = RHYTHM_CHUNK_DUR - OVERLAP_DUR
    
    total_duration = rhythm_prompt.duration
    
    # If total duration is short enough, just run once
    if total_duration <= RHYTHM_CHUNK_DUR:
         # Batch inputs
        timbre_prompt_batch = AudioSignal.batch([timbre_prompt]*N_OUTPUTS)
        rhythm_prompt_batch = AudioSignal.batch([rhythm_prompt]*N_OUTPUTS)
        
        # We use a custom buffer duration here to match the exact length needed + prefix
        # But actually _inference truncates rhythm prompt to fit buffer - prefix.
        # So we can just set buffer_dur = PREFIX + total_duration
        current_buffer_dur = PREFIX_DUR + total_duration
        
        out, msg = _inference(
            timbre_prompt_batch, rhythm_prompt_batch,
            sampling_ctrls, inference_ctrls, 
            audio_ctrls, schedule_ctrls, seed,
            override_buffer_dur=current_buffer_dur,
            override_prefix_dur=PREFIX_DUR
        )
        final_output = out.cpu()
        
    else:
        # Long audio: Split and Overlap-Add
        progress(0, desc="Starting generation...")
        
        # Resample upfront to avoid issues
        rhythm_prompt = rhythm_prompt.resample(sample_rate)
        timbre_prompt = timbre_prompt.resample(sample_rate)
        
        # Calculate number of segments
        # We need to cover total_duration
        # First segment covers [0, RHYTHM_CHUNK_DUR]
        # Next starts at STEP_SIZE...
        
        num_segments = math.ceil((total_duration - OVERLAP_DUR) / STEP_SIZE)
        if num_segments < 1: num_segments = 1
        
        full_output = torch.zeros(
            N_OUTPUTS, 1, int(total_duration * sample_rate) + int(sample_rate), # slightly larger buffer
            device='cpu'
        )
        # Weight buffer for normalization
        weight_buffer = torch.zeros(
             1, 1, full_output.shape[-1],
             device='cpu'
        )
        
        # Create crossfade window
        # Overlap region is OVERLAP_DUR seconds
        overlap_samples = int(OVERLAP_DUR * sample_rate)
        fade_in = torch.linspace(0, 1, overlap_samples)
        fade_out = torch.linspace(1, 0, overlap_samples)
        
        # Main loop
        current_time = 0.0
        
        for i in range(num_segments):
            progress(i / num_segments, desc=f"Generating segment {i+1}/{num_segments}")
            
            # Extract chunk
            # If it's the last chunk, take what's left, but minimum length constraints apply
            chunk_start = i * STEP_SIZE
            chunk_end = min(chunk_start + RHYTHM_CHUNK_DUR, total_duration)
            
            # Ensure we have enough audio for the chunk, padding if necessary (though min duration handles this)
            # Actually AudioSignal truncate handles cropping.
            
            # We need to pass a rhythm prompt that starts at chunk_start
            # and has duration = chunk_end - chunk_start
            
            # Slice audio
            start_sample = int(chunk_start * sample_rate)
            end_sample = int(chunk_end * sample_rate)
            
            rhythm_chunk = rhythm_prompt.clone()
            rhythm_chunk.audio_data = rhythm_chunk.audio_data[..., start_sample:end_sample]
            
            chunk_dur = rhythm_chunk.duration
            
            # Prepare batch
            timbre_batch = AudioSignal.batch([timbre_prompt]*N_OUTPUTS)
            rhythm_batch = AudioSignal.batch([rhythm_chunk]*N_OUTPUTS)
            
            # Inference
            # Buffer duration for this call = PREFIX + chunk_dur
            current_buffer_dur = PREFIX_DUR + chunk_dur
            
            # Ensure it's not too small for the model (though our chunks are large enough usually)
            
            out_chunk, msg = _inference(
                timbre_batch, rhythm_batch,
                sampling_ctrls, inference_ctrls, 
                audio_ctrls, schedule_ctrls, seed,
                override_buffer_dur=current_buffer_dur,
                override_prefix_dur=PREFIX_DUR
            )
            out_chunk = out_chunk.cpu()
            
            # Add to full output
            # out_chunk contains the generated rhythm (corresponding to rhythm_chunk)
            # Its length should match rhythm_chunk length in samples approximately
            
            chunk_data = out_chunk.audio_data # (B, 1, T)
            T_chunk = chunk_data.shape[-1]
            
            # Define global position
            global_start = int(chunk_start * sample_rate)
            global_end = global_start + T_chunk
            
            # Apply windowing for crossfade if not first/last
            # Ideally we window the overlap regions.
            # Simple OLA: standard trapezoidal window or similar
            # But here we specified fade-in/fade-out in overlap.
            
            # Construct a window for this chunk
            window = torch.ones(1, 1, T_chunk)
            
            # Fade in (if not first segment)
            if i > 0:
                window[..., :overlap_samples] = fade_in
            
            # Fade out (if not last segment)
            if i < num_segments - 1:
                # The end of this chunk overlaps with the next
                # The overlap starts at T_chunk - overlap_samples
                window[..., -overlap_samples:] = fade_out
            
            # Add
            full_output[..., global_start:global_end] += chunk_data * window
            weight_buffer[..., global_start:global_end] += window
            
        
        # Normalize by weight
        # Avoid division by zero
        mask = weight_buffer > 1e-6
        full_output[mask] /= weight_buffer[mask]
        
        # Create final AudioSignal
        # Trim to exact length
        final_len = int(total_duration * sample_rate)
        full_output = full_output[..., :final_len]
        
        final_output = AudioSignal(full_output, sample_rate)
        final_output.normalize(audio_ctrls["loudness_db"])
        final_output.ensure_max_of_audio()

    out_audio = [to_gradio_audio(final_output)] # Single output
    out_specs = [
        spectrogram_fast(final_output, small=False)
    ]

    return out_audio + out_specs + [msg]


########################################
# Interface
########################################

example_files = scan_example_timbres()

with gr.Blocks(title="Voice to Drum") as demo:
    gr.Markdown("# 🥁 Voice to Drum 🥁")
    gr.Markdown(
        "Select a **Model**, load **Timbre** + **Rhythm** prompts (record or upload). "
        "Click **Generate**."
    )

    # Inputs row
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Timbre Prompt")
            timbre_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="numpy",
                label="Timbre Audio",
            )
            
            # Dynamic Example Selection
            with gr.Row():
                example_dropdown = gr.Dropdown(
                    label="Example Timbres", 
                    choices=example_files,
                    value=None,
                    scale=3
                )
                refresh_btn = gr.Button("🔄", scale=0)
            
            timbre_spec = gr.Image(label="Timbre Spectrogram")
            
        with gr.Column():
            gr.Markdown("### Rhythm Prompt")
            rhythm_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="numpy",
                label="Rhythm Audio",
            )
            rhythm_spec = gr.Image(label="Rhythm Spectrogram")
            btn_rhythm_ex = gr.Button("Load Example Rhythm", variant="secondary")

    # Outputs row
    gr.Markdown("### Generated Output")
    generate_btn = gr.Button("Generate", variant="primary")

    
    with gr.Row():
        with gr.Column():
            out_audio_1 = gr.Audio(
                type="numpy",
                label="Generated Audio",
                interactive=False,
            )
            out_spec_1  = gr.Image(label="Spectrogram")
        
        # Removed extra outputs
    
    params_text = gr.Markdown("")

    # Controls bottom: 2 columns (left: Model + Sampling + Audio; right: Schedule)
    with gr.Row():
        with gr.Column():
            # Model selector above everything on left
            model_names = list(MODEL_ZOO.keys())
            model_name = gr.Dropdown(choices=model_names, value=model_names[0], label="Model")
            model_status = gr.Markdown("")

            with gr.Accordion("Sampling", open=False):
                with gr.Row():
                    top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top P")
                    top_k = gr.Slider(0, 1024, value=0, step=1, label="Top K")
                with gr.Row():
                    temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Temperature")
                    mask_temperature = gr.Slider(0.0, 50.0, value=10.5, step=0.1, label="Mask Temperature")
                with gr.Row():
                    seed = gr.Slider(0, 1000, value=0, step=1, label="Random Seed")
                    causal_bias = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Causal Bias")
                    cfg_scale = gr.Slider(0.0, 10.0, value=2.0, step=0.1, label="CFG Scale")

            with gr.Accordion("Audio", open=False):
                with gr.Row():
                    loudness_db = gr.Slider(-80.0, 0.0, value=-20.0, step=0.5, label="Loudness (dB)")
                    filter_inputs = gr.Checkbox(value=False, label="Filter Inputs")

        with gr.Column():
            with gr.Accordion("Schedule", open=False):
                schedule_sliders = []
                for i in range(1, 10):
                    schedule_sliders.append(
                        gr.Slider(0, 32, value=8, step=1, label=f"Codebook {i} Iterations")
                    )

    # Automatically update plots on audio upload
    timbre_audio.change(fn=handle_audio_change, inputs=timbre_audio, outputs=[timbre_spec], show_progress="hidden")
    rhythm_audio.change(fn=handle_audio_change, inputs=rhythm_audio, outputs=[rhythm_spec], show_progress="hidden")

    # Example Refresh & Load
    refresh_btn.click(fn=on_refresh_examples, inputs=None, outputs=example_dropdown)
    example_dropdown.change(
        fn=on_load_example_from_dropdown, 
        inputs=example_dropdown, 
        outputs=[timbre_audio, timbre_spec],
        show_progress="hidden"
    )

    # Default examples
    btn_rhythm_ex.click(fn=load_example_rhythm, inputs=None, outputs=[rhythm_audio, rhythm_spec], show_progress="hidden")

    # Load model and utilities
    model_name.change(fn=on_select_model, inputs=model_name, outputs=model_status, show_progress="hidden")
    demo.load(fn=on_select_model, inputs=model_name, outputs=model_status)

    generate_btn.click(
        fn=on_generate,
        inputs=[model_name, timbre_audio, rhythm_audio,
                top_p, top_k, temperature, mask_temperature,
                seed, causal_bias, cfg_scale,
                loudness_db, filter_inputs, *schedule_sliders],
        outputs=[out_audio_1,
                 out_spec_1,
                 params_text],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
