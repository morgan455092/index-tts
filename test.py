import argparse
from indextts.infer_v2 import IndexTTS2

def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 CLI")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--text", type=str, required=True, help="Input text for TTS")
    parser.add_argument("--spk_audio_prompt", type=str, default="examples/voice_07.wav", help="Reference speaker audio")
    parser.add_argument("--emo_audio_prompt", type=str, default=None, help="Optional emotion audio")
    parser.add_argument("--output_path", type=str, default="gen.wav", help="Output wav file path")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Directory of model checkpoints")
    parser.add_argument("--use_fp16", action="store_true", help="Enable FP16")
    parser.add_argument("--use_cuda_kernel", action="store_true", help="Enable CUDA kernel")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    
    args = parser.parse_args()

    # 初始化 TTS
    tts = IndexTTS2(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        use_fp16=args.use_fp16,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed
    )

    # 執行推理
    tts.infer(
        spk_audio_prompt=args.spk_audio_prompt,
        text=args.text,
        output_path=args.output_path,
        emo_audio_prompt=args.emo_audio_prompt,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
